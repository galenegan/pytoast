import numpy as np
import scipy.signal as sig
from scipy.stats import linregress
from typing import Optional, Union, List, Dict, Any
from utils.despike_utils import threshold, goring_nikora, recursive_gaussian
from utils.spectral_utils import psd, csd, get_frequency_range
from utils.base_instrument import BaseInstrument
from utils.constants import GRAVITATIONAL_ACCELERATION as g
from utils.rotate_utils import apply_flow_rotation


class Sonic(BaseInstrument):
    """Class for processing data from Sonic anemometers.

    Contains methods for: - Loading data from source files
    - Preprocessing (despiking, flow-dependent rotations)
    - Calculating turbulence statistics: TKE dissipation, Reynolds stress, TKE, buoyancy flux
    """

    def __init__(
        self,
        files: Union[str, List],
        name_map: dict,
        deployment_type: str = "fixed",
        fs: Optional[float] = None,
        z: Optional[Union[float, List[float]]] = None,
        data_keys: Optional[Union[str, List[str]]] = None,
        path_length: float = 0.15,
    ):
        """Initialize a Sonic object.

        Parameters
        ----------
        files : str or List[str]
            Path(s) to data files. If a list, each element is treated as a file containing data from
            an individual burst period. Supported formats: .npy (saved as a dict), .mat (saved as a
            MATLAB struct), .csv (variables in columns). If variables are two-dimensional, the larger
            dimension is assumed to be time and the shorter dimension a vertical coordinate.
        name_map : dict
            Mapping of standard variable names to names in the data files, e.g.:
            {
                "u1": "x-velocity variable name" or ["var 1", "var 2", ...],
                "u2": "y-velocity variable name" or ["var 1", "var 2", ...],
                "u3": "z-velocity variable name" or ["var 1", "var 2", ...],
                "Ts": "sonic temperature variable name" or ["var 1", "var 2", ...],
                "time": "time variable name" or ["var 1", "var 2", ...],
            }
            "Ts" and "time" are optional, but an error is raised if "time" is absent and `fs` is
            also not provided. Lists are used when data from multiple instruments are stored in
            separate variables rather than a 2-D array.
        deployment_type : str, optional
            One of {"fixed", "cast"} depending on how the instrument is deployed. Default is "fixed", in which case
            self.z will be converted to a constant numpy array of instrument deployment depths or measurement cell
            heights. If "cast", self.z will be set to None and vertical coordinates will be calculated as a data
            variable within individual measurement bursts.
        fs : float, optional
            Sampling frequency (Hz). If not provided, it will be inferred (and rounded to 2 decimal places) from the
            `time` variable
        z : float or List[float], optional
            Mean height above the surface (m) for each instrument. Defaults to integer indices if not
            specified.
        data_keys : str or List[str], optional
            One or more nested keys to traverse after loading the file (e.g. "Data" if the variables
            in name_map are stored at `burst["Data"]["variable_name"]`).
        path_length : float, optional
            Sonic path length (m). Used in the Henjes correction to the spectral curve fit in
            `Sonic.dissipation`. Defaults to 0.15.

        Returns
        -------
        Sonic
        """
        self.path_length = path_length
        files_list = files if isinstance(files, list) else [files]
        Sonic.validate_inputs(files_list, name_map, deployment_type, fs, z, data_keys, path_length)
        super().__init__(files, name_map, deployment_type=deployment_type, fs=fs, z=z, data_keys=data_keys)

    @staticmethod
    def validate_inputs(
        files: Union[str, List],
        name_map: dict,
        deployment_type: str = "fixed",
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[float, int, List[Union[float, int]]]] = None,
        data_keys: Optional[Union[str, List[str]]] = None,
        path_length: float = 0.15,
    ):

        # General validation
        BaseInstrument.validate_common_inputs(files, name_map, deployment_type, fs, z, data_keys)

        # Instrument-specific requirements
        required_keys = ["u1", "u2", "u3"]

        for key in required_keys:
            if key not in name_map:
                raise ValueError(f"`name_map` must include a mapping for '{key}'")

        if not isinstance(path_length, float):
            raise TypeError("`path length` must be a float")

    def set_preprocess_opts(self, opts: Dict[str, Any]):
        """Enable preprocessing for all subsequent burst loads using the
        options defined in the input dictionary.

        Parameters
        ----------
        opts : dict
            Preprocessing options. Supported keys:

            despike : dict, optional
                Options for despiking. If not specified, no despiking is applied. Supported keys:

                method : {'threshold', 'goring_nikora', 'recursive_gaussian'}
                    If `threshold`, data is despiked by replacing any samples with a magnitude outside a specified
                    range. If `goring_nikora`, data is despiked using the Goring & Nikora (2002) algorithm. If
                    `recursive_gaussian`, data is despiked using a recursive Gaussian filter.

                If ``{'method': 'goring_nikora', ...}``, additional keys can be (see `goring_nikora` docstring):
                    remaining_spikes : int
                    max_iter : int
                    robust_statistics : bool

                If ``{'method': 'threshold', ...}``, additional keys can be:
                    threshold_min : float
                    threshold_max : float

                If ``{'method': 'recursive_gaussian', ...}``, additional keys can be:
                    alpha : float
                    max_iter : int

            rotate : dict, optional
                Options for rotations. If not specified, no rotations applied. Supported keys:
                    flow_rotation : str or Tuple[float], optional.
                        One of {`align_principal`, `align_streamwise`, or (horizontal_angle_degrees,
                        vertical_angle_degrees)}. If `align_principal`, then the velocity will be rotated to align with
                        the principal axes of the flow. If `align_streamwise`, then the velocity will be rotated to
                        align with the horizontal wind magnitude sqrt(u^2 + v^2). In both cases, the vertical velocity
                        will be minimized. If float angles are specified in a tuple, the flow will be rotated by those
                        angles in the horizontal and vertical planes.
        """
        self._preprocess_opts = opts
        self._preprocess_enabled = True

        self._despike = opts.get("despike", {})
        if self._despike:
            self._despike_method = self._despike.get("method")
            self._despike_opts = {key: val for key, val in self._despike.items() if key != "method"}

        self._rotate = opts.get("rotate", {})
        self._cached_idx = None
        self._cached_data = None

    def _apply_preprocessing(self, burst_data):

        if not self._preprocess_enabled:
            return burst_data

        if self._despike:
            despike_fn = {
                "goring_nikora": goring_nikora,
                "threshold": threshold,
                "recursive_gaussian": recursive_gaussian,
            }.get(self._despike_method)
            if despike_fn is None:
                raise ValueError(f"Invalid despiking method '{self._despike_method}'")
            for key in ["u1", "u2", "u3"]:
                burst_data[key] = despike_fn(burst_data[key], **self._despike_opts)

        if self._rotate:
            flow_rotation = self._rotate.get("flow_rotation")
            if flow_rotation:
                burst_data = apply_flow_rotation(burst_data, flow_rotation)

        return burst_data

    def dissipation(
        self,
        burst_data: dict,
        f_low: float,
        f_high: float,
        henjes_correction: bool,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the dissipation rate of TKE via spectral curve fit to the
        streamwise wavenumber spectrum.

        Choice of constant is consistent with Edson and Fairall (1998), and the path length correction of Henjes et al (1999) can
        be optionally applied as well.

        Parameters
        ----------
        burst_data : dict
            Burst data dictionary containing `u1` key. This should correspond to the streamwise velocity (e.g., by
            specifying `align_streamwise` in the preprocessing options) but this is not explicitly enforced.
        f_low : float
            Lower bound (Hz) of inertial subrange where the curve fit is carried out
        f_high : float
            Upper bound (Hz) of inertial subrange where the curve fit is carried out
        henjes_correction : bool
            If True, apply the Henjes et al. path length correction to the spectral curve fit
        kwargs : dict
            Additional keyword arguments to pass to `spectral_utils.psd`

        Returns
        -------
        eps : np.ndarray
            Dissipation rate of TKE at each height

        References
        ----------
        Edson, J. B., & Fairall, C. W. (1998). Similarity relationships in the marine atmospheric surface layer for
            terms in the TKE and scalar variance budgets. Journal of the atmospheric sciences, 55(13), 2311-2328.
        Henjes, K., Taylor, P. K., & Yelland, M. J. (1999). Effect of pulse averaging on sonic anemometer spectra.
            Journal of Atmospheric and Oceanic Technology, 16(1), 181-184.
        """

        def spectral_fit(
            u: np.ndarray,
            f_low: float,
            f_high: float,
            henjes_correction: bool = True,
            **kwargs,
        ) -> np.ndarray:
            c1 = 0.53
            u_prime = sig.detrend(u, type="linear")
            u_bar = np.nanmean(u)
            f, S = psd(u_prime, fs=self.fs, onesided=True, **kwargs)

            if henjes_correction:
                fs = self.fs
                L = self.path_length
                delta_t = 1 / fs

                att1 = (np.sin(np.pi * f * delta_t) / (np.pi * f * delta_t)) ** 2
                att2 = ((f / (fs - f)) ** (5 / 3)) * (
                    np.sin(np.pi * (fs - f) * delta_t) / (np.pi * (fs - f) * delta_t)
                ) ** 2
                L1 = np.sin(np.pi * f * L / u_bar) ** 2 / ((np.pi * f * L / u_bar) ** 2)
                L2 = np.sin(np.pi * (fs - f) * L / u_bar) ** 2 / ((np.pi * (fs - f) * L / u_bar) ** 2)
                S = S / (L1 * att1 + L2 * att2)

            # Converting to wavenumber spectrum
            G = S * u_bar / (2 * np.pi)
            k = 2 * np.pi * f / u_bar

            # Fit range
            good_data = (f > f_low) & (f < f_high)

            # Doing the fit
            if (np.sum(np.isnan(G)) > len(G) / 2) or (sum(good_data) < 20):
                eps = np.nan
            else:
                X = c1 * k[good_data] ** (-5 / 3)
                y = G[good_data]
                slope, *_ = linregress(X, y)
                eps23 = slope
                if eps23 < 0:
                    eps = np.nan
                else:
                    eps = eps23 ** (3 / 2)
            return eps

        n_heights = self.n_heights
        eps = np.empty((n_heights,))
        for height_idx in range(n_heights):
            u = burst_data["u1"][height_idx, :]
            eps[height_idx] = spectral_fit(
                u,
                henjes_correction=henjes_correction,
                f_low=f_low,
                f_high=f_high,
                **kwargs,
            )

        return eps

    def covariance(
        self,
        burst_data: Dict[str, np.ndarray],
        method: str = "cov",
        f_low: Optional[float] = None,
        f_high: Optional[float] = None,
        **kwargs,
    ):
        """Calculate components of the covariance matrix (i.e., the Reynolds
        stress)

        Parameters
        ----------
        burst_data : dict
            Burst data dictionary.
        method : str
            Method to calculate covariances. Options are:
            - `cov`: Standard covariance calculation using the built-in `np.cov`
            - `spectral_integral`: Integrate the cross-spectrum over a specified frequency range
        f_low : float, optional
            Lower frequency bound (Hz) for spectral integration, by default None
        f_high : float, optional
            Upper frequency bound (Hz) for spectral integration, by default None
        **kwargs
            Additional arguments passed to spectral calculations

        Returns
        -------
        out : dict
            Dictionary containing covariance components.
        """
        out = {}
        n_heights = self.n_heights
        if method == "cov":
            u_bar = np.mean(burst_data["u1"], axis=1, keepdims=True)
            v_bar = np.mean(burst_data["u2"], axis=1, keepdims=True)
            w_bar = np.mean(burst_data["u3"], axis=1, keepdims=True)
            u_prime = burst_data["u1"] - u_bar
            v_prime = burst_data["u2"] - v_bar
            w_prime = burst_data["u3"] - w_bar

            out["uu"] = np.mean(u_prime**2, axis=1)
            out["vv"] = np.mean(v_prime**2, axis=1)
            out["ww"] = np.mean(w_prime**2, axis=1)
            out["uw"] = np.mean(u_prime * w_prime, axis=1)
            out["vw"] = np.mean(v_prime * w_prime, axis=1)
            out["uv"] = np.mean(u_prime * v_prime, axis=1)

        elif method == "spectral_integral":
            out["uu"] = np.empty((n_heights,))
            out["vv"] = np.empty((n_heights,))
            out["ww"] = np.empty((n_heights,))
            out["uw"] = np.empty((n_heights,))
            out["vw"] = np.empty((n_heights,))
            out["uv"] = np.empty((n_heights,))

            for height_idx in range(n_heights):
                u = burst_data["u1"][height_idx, :]
                v = burst_data["u2"][height_idx, :]
                w = burst_data["u3"][height_idx, :]

                # Power spectral densities
                f, S_uu = psd(u, fs=self.fs, **kwargs)
                f, S_vv = psd(v, fs=self.fs, **kwargs)
                f, S_ww = psd(w, fs=self.fs, **kwargs)
                f, S_uw = csd(u, w, fs=self.fs, **kwargs)
                f, S_vw = csd(v, w, fs=self.fs, **kwargs)
                f, S_uv = csd(u, v, fs=self.fs, **kwargs)

                start_index, end_index = get_frequency_range(f, f_low, f_high)
                df = np.nanmax(np.diff(f))

                out["uu"][height_idx] = np.sum(np.real(S_uu[start_index:end_index]) * df)
                out["vv"][height_idx] = np.sum(np.real(S_vv[start_index:end_index]) * df)
                out["ww"][height_idx] = np.sum(np.real(S_ww[start_index:end_index]) * df)
                out["uw"][height_idx] = np.sum(np.real(S_uw[start_index:end_index]) * df)
                out["vw"][height_idx] = np.sum(np.real(S_vw[start_index:end_index]) * df)
                out["uv"][height_idx] = np.sum(np.real(S_uv[start_index:end_index]) * df)
        else:
            raise ValueError(f"Invalid covariance method '{method}'")

        return out

    def tke(self, burst_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculates turbulent kinetic energy.

        Parameters
        ----------
        burst_data : dict
            Burst data dictionary containing velocity components u1/u2/u3

        Returns
        -------
        tke_out : np.ndarray
            TKE at each measurement height
        """
        u_bar = np.mean(burst_data["u1"], axis=1, keepdims=True)
        v_bar = np.mean(burst_data["u2"], axis=1, keepdims=True)
        w_bar = np.mean(burst_data["u3"], axis=1, keepdims=True)

        u_prime = burst_data["u1"] - u_bar
        v_prime = burst_data["u2"] - v_bar
        w_prime = burst_data["u3"] - w_bar

        tke_prime = 0.5 * (u_prime**2 + v_prime**2 + w_prime**2)
        tke_out = np.mean(tke_prime, axis=1)

        return tke_out

    def buoyancy_flux(self, burst_data: Dict[str, np.ndarray]):
        """Buoyancy flux from the sonic temperature/vertical velocity
        covariance (e.g., Liu et al., (2001)).

        Parameters
        ----------
        burst_data : dict
            Burst data dictionary containing u3 and Ts

        Returns
        -------
        B : np.ndarray
            Buoyancy flux at each measurement height

        References
        ----------
        Liu, H., Peters, G., & Foken, T. (2001). New equations for sonic temperature variance and buoyancy heat flux
            with an omnidirectional sonic anemometer. Boundary-Layer Meteorology, 100(3), 459-468.
        """
        if "Ts" not in burst_data:
            raise ValueError("Cannot compute buoyancy flux without sonic temperature data")

        Ts_bar = np.mean(burst_data["Ts"], axis=1, keepdims=True)
        w_bar = np.mean(burst_data["u3"], axis=1, keepdims=True)
        Ts_prime = burst_data["Ts"] - Ts_bar
        w_prime = burst_data["u3"] - w_bar
        B = g * np.mean(Ts_prime * w_prime, axis=1) / (Ts_bar + 273.15)
        return B

    def subsample(self, start_idx, end_idx):
        """Subsample the Sonic object between files[start_idx] and
        files[end_idx].

        Parameters
        ----------
        start_idx : int
            First file to include in subsampling
        end_idx : int
            Upper bound (exclusive) on file index in subsampling

        Returns
        -------
        new_sonic : Sonic
            Subsampled Sonic object
        """
        new_sonic = self.__class__(
            files=self.files[start_idx:end_idx],
            name_map=self.name_map,
            deployment_type=self.deployment_type,
            fs=self.fs,
            z=self.z,
            data_keys=self.data_keys,
            path_length=self.path_length,
        )
        if self._preprocess_enabled:
            new_sonic.set_preprocess_opts(self._preprocess_opts)
        return new_sonic
