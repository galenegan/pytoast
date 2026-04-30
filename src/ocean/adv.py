import numpy as np
import scipy.signal as sig
from typing import Optional, Union, List, Dict, Any
from utils.base_instrument import BaseInstrument
from utils.despike_utils import threshold, goring_nikora, recursive_gaussian
from utils.wave_utils import get_wavenumber, get_cg, jones_monismith_correction
from scipy.stats import linregress

from utils.spectral_utils import psd, csd, get_frequency_range
from utils.constants import GRAVITATIONAL_ACCELERATION as g
from utils.rotate_utils import (
    coord_transform_3_beam_nortek,
    apply_flow_rotation,
)


def _find_wave_band(f: np.ndarray, P_uu: np.ndarray, df: float) -> np.ndarray:
    """
    Locate the wave peak in the u-component auto-spectrum and return indices spanning
    0.35 * f_peak below to 0.8 * f_peak above, clipped to the valid range of `f`.
    """
    f_offset = 0.07  # Assume wave peak is above this value
    width_ratio_low = 0.35
    width_ratio_high = 0.8
    search = (f > f_offset) & (f < 1)
    peak_idx = int(np.flatnonzero(search)[np.argmax(P_uu[search])])
    f_peak = f[peak_idx]
    n_below = int((f_peak * width_ratio_low) // df)
    n_above = int((f_peak * width_ratio_high) // df)
    lo = max(peak_idx - n_below, 0)
    hi = min(peak_idx + n_above, len(f) - 1)
    return np.arange(lo, hi)


class ADV(BaseInstrument):
    """Class for processing data from Acoustic Doppler Velocimeter (ADV)
    instruments.

    Contains methods for:

    - Loading data from source files
    - Preprocessing (despiking, coordinate transformations, flow-dependent rotations)
    - Calculating turbulence statistics: TKE, TKE dissipation, Reynolds stress (including wave-turbulence decomposed)
    - Calculating directional wave statistics
    """

    def __init__(
        self,
        files: Union[str, List],
        name_map: dict,
        deployment_type: str = "fixed",
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[List[Union[float, int]], np.ndarray]] = None,
        data_keys: Optional[Union[str, List[str]]] = None,
        source_coords: str = "xyz",
        orientation: str = "up",
    ):
        """Initialize an ADV object.

        Parameters
        ----------
        files : str or List[str]
            Path(s) to data files. If a list, each element is treated as a file containing data from an individual burst
            period. Supported formats: `.npy` (saved as a dict), `.mat` (saved as a MATLAB struct), `.csv` (variables in
            columns). If variables are two-dimensional, the larger dimension is assumed to be time and the shorter
            dimension is assumed to be a vertical coordinate.
        name_map : dict
            Mapping of standard variable names to names in the data files, e.g.:
            {
                "u1": "first velocity variable name" or ["var 1", "var 2", ...],
                "u2": "second velocity variable name" or ["var 1", "var 2", ...],
                "u3": "third velocity variable name" or ["var 1", "var 2", ...],
                "p": "pressure variable name" or ["var 1", "var 2", ...],
                "time": "time variable name" or ["var 1", "var 2", ...],
                "heading": "heading variable name" or ["var 1", "var 2", ...],
                "pitch": "pitch variable name" or ["var 1", "var 2", ...],
                "roll": "roll variable name" or ["var 1", "var 2", ...],
            }
            `p` and `time` are optional, but an error is raised if `time` is absent and `fs` is also not provided.
            `heading`, `pitch`, and `roll` are also optional but required for ENU coordinate transformations. Lists are
            used when data from multiple instruments are stored in separate variables rather than a 2-D array.
        deployment_type : str, optional
            One of {"fixed", "cast"} depending on how the instrument is deployed. Default is "fixed", in which case
            self.z will be converted to a constant numpy array of instrument deployment depths or measurement cell
            heights. If "cast", self.z will be set to None and vertical coordinates will be calculated as a data
            variable within individual measurement bursts.
        fs : int or float, optional
            Sampling frequency (Hz). If not provided, it will be inferred (and rounded to 2 decimal places) from the
            `time` variable
        z : float, List[float, int], or np.ndarray, optional
            Mean height above the bed (m) for each instrument. If not provided, it will default to integer indices, in
            which case certain functionality (e.g., wave statistics) will not be available. Unlike the ADCP class,
            interpretation of `z` will not vary depending on `orientation`: input must be in meters above bed for
            depth-dependent calculations to work correctly.
        data_keys : str or List[str], optional
            One or more nested keys to traverse after loading the file (e.g. "Data" if the variables in name_map are
            stored at `burst["Data"]["variable_name"]`).
        source_coords : str, optional
            Velocity coordinate system in the source files. One of {`xyz`, `enu`, `beam`}.
            Defaults to `xyz`.
        orientation : str, optional
            Orientation of the ADV probe. One of {`up`, `down`}. Defaults to `up`. For Nortek Vector ADVs, this
            corresponds to the end cap pointing up and the probe pointing down (see
            https://support.nortekgroup.com/hc/en-us/articles/360029507712-What-do-the-Error-and-Status-codes-mean)

        Returns
        -------
        ADV
        """
        self.source_coords = source_coords
        self.orientation = orientation
        files_list = files if isinstance(files, list) else [files]
        ADV.validate_inputs(files_list, name_map, deployment_type, fs, z, data_keys, source_coords, orientation)
        super().__init__(files, name_map, deployment_type=deployment_type, fs=fs, z=z, data_keys=data_keys)

    @staticmethod
    def validate_inputs(
        files: Union[str, List],
        name_map: dict,
        deployment_type: str = "fixed",
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[float, int, List[Union[float, int]], np.ndarray]] = None,
        data_keys: Optional[Union[str, List[str]]] = None,
        source_coords: Optional[str] = "xyz",
        orientation: Optional[str] = "up",
    ):

        # General validation
        BaseInstrument.validate_common_inputs(files, name_map, deployment_type, fs, z, data_keys)

        # Instrument-specific requirements
        required_keys = ["u1", "u2", "u3"]

        for key in required_keys:
            if key not in name_map:
                raise ValueError(f"`name_map` must include a mapping for '{key}'")

        if source_coords not in ["xyz", "enu", "beam"]:
            raise ValueError(
                f"Invalid value for `source_coords`: {source_coords}. Must be one of ['xyz', 'enu', 'beam']"
            )

        if orientation not in ["down", "up"]:
            raise ValueError(f"Invalid value for `orientation`: {orientation}. Must be one of ['down', 'up']")

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
                Options for rotations and coordinate transformations. If not specified, no rotations applied.
                Supported keys:

                coords_out : str, optional
                    Coordinates for burst data to be transformed to. One of {`beam`, `xyz`, `enu`}.

                transformation_matrices : List[np.ndarray], optional
                    Transformation matrices for the instruments. Length must match ADV.n_heights.

                declination : float, optional
                    Magnetic declination in degrees. Added to heading for coordinate transformations.

                constant_hpr : List[Tuple[float]], optional
                    Constant heading, pitch, and roll angles to apply at each instrument in lieu of full heading, pitch
                    and roll arrays in the burst. Length of the list must match ADV.n_heights

                flow_rotation : str or Tuple[float], optional.
                    One of {`align_principal`, `align_streamwise`, or (horizontal_angle_degrees, vertical_angle_degrees)}.
                    If `align_principal`, then the velocity will be rotated to align with the principal axes of the
                    flow. If `align_streamwise`, then the velocity will be rotated to align with the horizontal current
                    magnitude sqrt(u^2 + v^2). In both cases, the vertical velocity will be minimized. If float angles
                    are specified in a tuple, the flow will be rotated by those angles in the horizontal and vertical
                    planes. Specifying any option will throw an error if `burst["coords"] == "beam"`, unless a
                    coordinate system change to `xyz` or `enu` is also requested.
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
        """Applies preprocessing to a burst data dictionary during loading."""
        burst_data["coords"] = self.source_coords
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
            coords_out = self._rotate.get("coords_out")
            if coords_out:
                burst_data = self._apply_coord_transform(burst_data, coords_out)

            flow_rotation = self._rotate.get("flow_rotation")
            if flow_rotation:
                if burst_data["coords"] == "beam":
                    raise ValueError(
                        "Cannot apply flow rotation in beam coordinates. Specify 'coords_out' "
                        "as 'xyz' or 'enu' in rotate options."
                    )
                burst_data = apply_flow_rotation(burst_data, flow_rotation)

        return burst_data

    def _apply_coord_transform(self, burst_data, coords_out):
        """Transform velocity components between coordinate systems.

        Uses configuration stored in self._rotate. Can be called from _apply_preprocessing during standard burst
        loading, or directly from analysis methods when transformation is needed.

        Parameters
        ----------
        burst_data : dict
            Burst data dictionary. `burst_data["coords"]` must reflect the current
            coordinate system of u1/u2/u3.
        coords_out : str
            Target coordinate system. One of {`beam`, `xyz`, `enu`}.

        Returns
        -------
        dict
            burst_data with velocity components transformed in-place and
            burst_data["coords"] updated to coords_out.
        """
        coords_in = burst_data["coords"]
        n_heights = self.n_heights

        transformation_matrices = self._rotate.get("transformation_matrices")
        if transformation_matrices is None:
            raise ValueError("A transformation matrix must be provided for each instrument")
        if len(transformation_matrices) != n_heights:
            raise ValueError(f"Expected {n_heights} transformation matrices, got {len(transformation_matrices)}")

        heading = burst_data.get("heading")
        pitch = burst_data.get("pitch")
        roll = burst_data.get("roll")

        if ((coords_in == "enu") or (coords_out == "enu")) and ((heading is None) or (pitch is None) or (roll is None)):
            constant_hpr = self._rotate.get("constant_hpr")
            if constant_hpr:
                if len(constant_hpr) != n_heights:
                    raise ValueError("A (heading, pitch, roll) tuple must be provided for each instrument")
                heading = np.array([constant_hpr[i][0] for i in range(n_heights)]).reshape(-1, 1)
                pitch = np.array([constant_hpr[i][1] for i in range(n_heights)]).reshape(-1, 1)
                roll = np.array([constant_hpr[i][2] for i in range(n_heights)]).reshape(-1, 1)
            else:
                raise ValueError(
                    "Heading, pitch, and roll must be provided for any coordinate transformation to/from ENU"
                )

        # Each instrument in the array may have a different orientation, so HPR
        # is indexed per instrument (height_idx).
        for height_idx in range(n_heights):
            u1_new, u2_new, u3_new = coord_transform_3_beam_nortek(
                u1=burst_data["u1"][height_idx, :],
                u2=burst_data["u2"][height_idx, :],
                u3=burst_data["u3"][height_idx, :],
                heading=heading[height_idx, :] if heading is not None else None,
                pitch=pitch[height_idx, :] if pitch is not None else None,
                roll=roll[height_idx, :] if roll is not None else None,
                transformation_matrix=transformation_matrices[height_idx],
                declination=self._rotate.get("declination", 0.0),
                orientation=self.orientation,
                coords_in=coords_in,
                coords_out=coords_out,
            )
            burst_data["u1"][height_idx, :] = u1_new
            burst_data["u2"][height_idx, :] = u2_new
            burst_data["u3"][height_idx, :] = u3_new

        burst_data["coords"] = coords_out
        return burst_data

    def benilov_decomposition(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        p: np.ndarray,
        mab: float,
        rho: float,
        f_low: Optional[float] = None,
        f_high: Optional[float] = None,
        **kwargs,
    ) -> dict[str, float]:
        """Benilov wave-turbulence decomposition to estimate wave and
        turbulence components of the Reynolds stress. (Benilov & Filyushkin,
        1970)

        Parameters
        ----------
        u : np.ndarray
            x-component of velocity (m/s)
        v : np.ndarray
            y-component of velocity (m/s)
        w : np.ndarray
            z-component of velocity (m/s)
        p: : np.ndarray
            pressure (dbar)
        mab : float
            meters above bed for pressure sensor
        rho : float
            fluid density (kg / m^3)
        f_low : float, optional
            lower frequency bound of spectral sum
        f_high : float, optional
            upper frequency bound of spectral sum
        kwargs: Additional arguments passed to spectral_utils.psd/csd.
                See spectral_utils.psd for parameter definitions.

        Returns
        -------
        Dictionary of turbulent and wave momentum flux components

        References
        ----------
        Benilov, A. Y., & Filyushkin, B. N. (1970). Application of methods of linear filtration to an analysis of
            fluctuations in the surface layer of the sea. Izv., Acad. Sci., USSR, Atmos. Oceanic Phys, 68, 810-819.
        """
        if not self._physical_z:
            raise ValueError("`z` values must be specified during initialization for Benilov decomposition.")

        h = 1e4 * np.nanmean(p) / (rho * g) + mab  # Average water depth before detrending

        u = sig.detrend(u)
        v = sig.detrend(v)
        w = sig.detrend(w)
        p = sig.detrend(p)

        # Getting sea surface elevation spectrum
        f, P_pp = psd(p, self.fs, **kwargs)
        df = np.max(np.diff(f))
        omega = 2 * np.pi * f
        k = get_wavenumber(omega, h)
        attenuation_correction = 1e4 * np.cosh(k * h) / (rho * g * np.cosh(k * mab))
        P_etaeta = P_pp * (attenuation_correction**2)

        # All the velocity components
        _, P_uu = psd(u, self.fs, **kwargs)
        _, P_up = csd(u, p, self.fs, **kwargs)
        P_ueta = P_up * attenuation_correction

        _, P_vv = psd(v, self.fs, **kwargs)
        _, P_vp = csd(v, p, self.fs, **kwargs)
        P_veta = P_vp * attenuation_correction

        _, P_ww = psd(w, self.fs, **kwargs)
        _, P_wp = csd(w, p, self.fs, **kwargs)
        P_weta = P_wp * attenuation_correction

        # Velocity cross spectra
        _, P_uw = csd(u, w, self.fs, **kwargs)
        _, P_vw = csd(v, w, self.fs, **kwargs)
        _, P_uv = csd(u, v, self.fs, **kwargs)

        # Defining frequency range
        start_index, end_index = get_frequency_range(f, f_low, f_high)

        # Calculating wave spectra
        P_uwave_uwave = P_ueta * np.conj(P_ueta) / P_etaeta
        P_vwave_vwave = P_veta * np.conj(P_veta) / P_etaeta
        P_wwave_wwave = P_weta * np.conj(P_weta) / P_etaeta
        P_uwave_wwave = P_ueta * np.conj(P_weta) / P_etaeta
        P_uwave_vwave = P_ueta * np.conj(P_veta) / P_etaeta
        P_vwave_wwave = P_veta * np.conj(P_weta) / P_etaeta

        # Calculating turbulent spectra
        P_ut_ut = P_uu - P_uwave_uwave
        P_ut_wt = P_uw - P_uwave_wwave
        P_ut_vt = P_uv - P_uwave_vwave
        P_vt_vt = P_vv - P_vwave_vwave
        P_vt_wt = P_vw - P_vwave_wwave
        P_wt_wt = P_ww - P_wwave_wwave

        # Summing them to get Reynolds stresses
        out = {}
        out["uu_turb"] = np.nansum(np.real(P_ut_ut[start_index:end_index]) * df)
        out["uu_wave"] = np.nansum(np.real(P_uwave_uwave[start_index:end_index]) * df)
        out["vv_turb"] = np.nansum(np.real(P_vt_vt[start_index:end_index]) * df)
        out["vv_wave"] = np.nansum(np.real(P_vwave_vwave[start_index:end_index]) * df)
        out["ww_turb"] = np.nansum(np.real(P_wt_wt[start_index:end_index]) * df)
        out["ww_wave"] = np.nansum(np.real(P_wwave_wwave[start_index:end_index]) * df)
        out["uw_turb"] = np.nansum(np.real(P_ut_wt[start_index:end_index]) * df)
        out["uw_wave"] = np.nansum(np.real(P_uwave_wwave[start_index:end_index]) * df)
        out["vw_turb"] = np.nansum(np.real(P_vt_wt[start_index:end_index]) * df)
        out["vw_wave"] = np.nansum(np.real(P_vwave_wwave[start_index:end_index]) * df)
        out["uv_turb"] = np.nansum(np.real(P_ut_vt[start_index:end_index]) * df)
        out["uv_wave"] = np.nansum(np.real(P_uwave_vwave[start_index:end_index]) * df)

        return out

    def phase_decomposition(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        f_low: Optional[float] = None,
        f_high: Optional[float] = None,
        f_wave_low: Optional[float] = None,
        f_wave_high: Optional[float] = None,
        **kwargs,
    ):
        """Bricker & Monismith (2007) phase method for wave-turbulence
        decomposition.

        Unlike the Benilov method, no pressure data are required.

        Parameters
        ----------
        u : np.ndarray
            x-component of velocity (m/s)
        v : np.ndarray
            y-component of velocity (m/s)
        w : np.ndarray
            z-component of velocity (m/s)
        f_low : float, optional
            lower frequency bound of spectral sum
        f_high : float, optional
            upper frequency bound of spectral sum
        f_wave_low : float, optional
            lower frequency bound of wave range. If not specified, the range is assumed to start at 0.35 f_max below
            the wave peak f_max
        f_wave_high : float, optional
            upper frequency bound of the wave range. If not specified, the range is assumed to end at 0.8 f_max above
            the wave peak f_max
        kwargs: Additional arguments passed to spectral_utils.psd/csd.
                See spectral_utils.psd for parameter definitions.

        Returns
        -------
        Dictionary of turbulent and wave momentum flux components

        References
        ----------
        Bricker, J. D., & Monismith, S. G. (2007). Spectral wave–turbulence decomposition. Journal of Atmospheric and
            Oceanic Technology, 24(8), 1479-1487.
        """
        out = {}

        u = sig.detrend(u)
        v = sig.detrend(v)
        w = sig.detrend(w)

        # Auto-spectra
        _, P_uu = psd(u, fs=self.fs, **kwargs)
        _, P_vv = psd(v, fs=self.fs, **kwargs)
        _, P_ww = psd(w, fs=self.fs, **kwargs)

        # Cross-spectra
        _, P_uw = csd(u, w, fs=self.fs, **kwargs)
        _, P_vw = csd(v, w, fs=self.fs, **kwargs)
        f, P_uv = csd(u, v, fs=self.fs, **kwargs)

        phase_uw = np.arctan2(np.imag(P_uw), np.real(P_uw))
        phase_vw = np.arctan2(np.imag(P_vw), np.real(P_vw))
        phase_uv = np.arctan2(np.imag(P_uv), np.real(P_uv))
        df = np.nanmax(np.diff(f))

        # Wave-band indices (explicit range if given, else locate peak in P_uu)
        if f_wave_low and f_wave_high:
            waverange = np.flatnonzero((f > f_wave_low) & (f < f_wave_high))
        else:
            waverange = _find_wave_band(f, P_uu, df)

        # Turbulent band: 0 < f < 1 Hz, wave band excluded. Used to fit the inertial subrange.
        interp_end = int(np.nanargmin(np.abs(f - 1)))
        turb_mask = np.zeros_like(f, dtype=bool)
        turb_mask[1:interp_end] = True
        turb_mask[waverange] = False
        log_f_turb = np.log(f[turb_mask])
        log_f_all = np.log(f)

        # Log-log linear fit of the turbulent spectrum, evaluated over the full frequency range
        fits = {}
        for name, P in (("uu", P_uu), ("vv", P_vv), ("ww", P_ww)):
            coefs = np.polyfit(log_f_turb, np.log(P[turb_mask]), deg=1)
            fits[name] = np.exp(np.polyval(coefs, log_f_all))

        # Wave spectra = excess over the turbulent fit within the wave band (clipped at 0)
        Puu_wave = np.clip(P_uu[waverange] - fits["uu"][waverange], 0, None)
        Pvv_wave = np.clip(P_vv[waverange] - fits["vv"][waverange], 0, None)
        Pww_wave = np.clip(P_ww[waverange] - fits["ww"][waverange], 0, None)

        # Wave Fourier amplitudes
        Um_wave = np.sqrt(Puu_wave * df)
        Vm_wave = np.sqrt(Pvv_wave * df)
        Wm_wave = np.sqrt(Pww_wave * df)

        out["uu_wave"] = np.nansum(Puu_wave * df)
        out["vv_wave"] = np.nansum(Pvv_wave * df)
        out["ww_wave"] = np.nansum(Pww_wave * df)
        out["uw_wave"] = np.nansum(Um_wave * Wm_wave * np.cos(phase_uw[waverange]))
        out["uv_wave"] = np.nansum(Um_wave * Vm_wave * np.cos(phase_uv[waverange]))
        out["vw_wave"] = np.nansum(Vm_wave * Wm_wave * np.cos(phase_vw[waverange]))

        # Full Reynolds stresses, then subtract wave contribution to get the turbulent part
        start_index, end_index = get_frequency_range(f, f_low, f_high)
        totals = {
            "uu": np.nansum(np.real(P_uu[start_index:end_index]) * df),
            "vv": np.nansum(np.real(P_vv[start_index:end_index]) * df),
            "ww": np.nansum(np.real(P_ww[start_index:end_index]) * df),
            "uw": np.nansum(np.real(P_uw[start_index:end_index]) * df),
            "vw": np.nansum(np.real(P_vw[start_index:end_index]) * df),
            "uv": np.nansum(np.real(P_uv[start_index:end_index]) * df),
        }
        for key, total in totals.items():
            out[f"{key}_turb"] = total - out[f"{key}_wave"]

        return out

    def dmd(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        f_wave_low: float,
        f_wave_high: float,
        rank_truncation: Union[int, float] = 0.05,
        time_delay_size: Optional[int] = None,
        return_time_series: bool = False,
    ) -> dict:
        """Estimate Reynolds stresses with the DMD-based wave-turbulence
        decomposition of Chavez-Dorado et al.

        (2025). This function is a Python port (with various simplifications) of the MATLAB implementation found here:
        https://github.com/DiBenedettoLab/Wave-Turbulence_DMD

        Parameters
        ----------
        u : np.ndarray
            x-component of velocity (m/s)
        v : np.ndarray
            y-component of velocity (m/s)
        w : np.ndarray
            z-component of velocity (m/s)
        f_wave_low : float
            Lower bound (Hz) of the wave frequency band.
        f_wave_high : float
            Upper bound (Hz) of the wave frequency band.
        rank_truncation : int or float, optional
            Controls how many DMD modes are retained after SVD. If a float, modes are kept if their corresponding
            singular value is at least `rank_truncation` times the largest singular value. If an int, exactly that many
            modes are kept. Defaults to 0.05.
        time_delay_size : int, optional
            Number of time-lag rows in the Hankel embedding matrix. Defaults to `N // 5`.
        return_time_series : bool, optional
            If True, also return the decomposed wave and turbulence time series for each velocity component. Defaults to
            False.

        Returns
        -------
        out : dict
            Dictionary with turbulence Reynolds stress components: `uu_turb`, `vv_turb`, `ww_turb`,
            `uw_turb`, `vw_turb`, `uv_turb`, each a scalar. If `return_time_series` is True, also
            includes `u_wave`, `u_turb`, `v_wave`, `v_turb`, `w_wave`, `w_turb`, each length N-1.

        References
        ----------
        Schmid, P. J. (2010). Dynamic mode decomposition of numerical and experimental data. Journal of fluid mechanics,
            656, 5-28.

        Chávez-Dorado, J., Scherl, I., & DiBenedetto, M. (2025). Wave and turbulence separation using dynamic mode
            decomposition. Journal of Atmospheric and Oceanic Technology, 42(5), 509-526.
        """

        def _decompose(signal: np.ndarray, n: int) -> tuple:
            """Run DMD on a 1-D signal of length N.

            Returns (u_wave, u_turb), each length N-1.
                        Parameters
                        ----------
                        signal : np.ndarray
                            1-D signal (usually velocity) of length N.
                        n : int
                            Number of time-lag rows for Hankel embedding.
            """
            raw = signal - np.mean(signal)
            N = len(raw)
            m = N - n

            # Build Hankel matrices: X1[j,i] = raw[i+j], X2[j,i] = raw[i+j+1]
            # Shape (m, n): rows are snapshot indices, columns are lag indices
            idx = np.arange(m)[:, None] + np.arange(n)[None, :]  # (m, n)
            X1 = raw[idx]
            X2 = raw[idx + 1]

            # SVD of X1
            U, s, Vh = np.linalg.svd(X1, full_matrices=False)
            V = Vh.T

            # Rank truncation based on singular values
            if isinstance(rank_truncation, float):
                r = int(np.sum(s >= rank_truncation * s[0]))
                r = max(r, 1)
            else:
                r = int(rank_truncation)

            Ur = U[:, :r]
            Sr_diag = s[:r]
            Vr = V[:, :r]
            Sr_inv = np.diag(1.0 / Sr_diag)

            # Low-rank dynamics matrix and eigendecomposition
            Atilde = Ur.T @ X2 @ Vr @ Sr_inv  # (r, r)
            lambda_, Wr = np.linalg.eig(Atilde)

            # DMD modes and amplitudes
            Phi = X2 @ Vr @ Sr_inv @ Wr  # (m, r)
            alpha1 = Sr_diag * Vr[0, :]  # (r,) first Hankel row scaled by singular values
            b = np.linalg.solve(Wr @ np.diag(lambda_), alpha1)  # (r,)

            # Frequencies from eigenvalues
            dt = 1.0 / self.fs
            f2 = np.imag(np.log(lambda_)) / (2.0 * np.pi * dt)

            # Select wave modes (both positive and negative frequencies)
            wave_mask = (np.abs(f2) >= f_wave_low) & (np.abs(f2) <= f_wave_high)
            b_wave = b[wave_mask]
            Phi_wave = Phi[:, wave_mask]
            omega_wave = lambda_[wave_mask]

            # Reconstruct wave Hankel block: time_dynamics[k, t] = b_wave[k] * omega_wave[k]^t
            t = np.arange(n)
            time_dynamics = b_wave[:, None] * (omega_wave[:, None] ** t)  # (r_wave, n)
            Xdmd_wave = Phi_wave @ time_dynamics  # (m, n)

            # Unfold Hankel structure: first row (n samples) + last-column tail (m-1 samples)
            u_wave = np.real(np.concatenate([Xdmd_wave[0, :], Xdmd_wave[1:, -1]]))
            u_turb = raw[:-1] - u_wave  # length N-1

            return u_wave, u_turb

        N = len(u)
        n = time_delay_size if time_delay_size is not None else N // 5

        u_wave, u_turb = _decompose(u, n)
        v_wave, v_turb = _decompose(v, n)
        w_wave, w_turb = _decompose(w, n)

        # Reynolds stresses from turbulence components
        out = {
            "uu_turb": np.mean(u_turb**2),
            "vv_turb": np.mean(v_turb**2),
            "ww_turb": np.mean(w_turb**2),
            "uw_turb": np.mean(u_turb * w_turb),
            "vw_turb": np.mean(v_turb * w_turb),
            "uv_turb": np.mean(u_turb * v_turb),
            "uu_wave": np.mean(u_wave**2),
            "vv_wave": np.mean(v_wave**2),
            "ww_wave": np.mean(w_wave**2),
            "uw_wave": np.mean(u_wave * w_wave),
            "vw_wave": np.mean(v_wave * w_wave),
            "uv_wave": np.mean(u_wave * v_wave),
        }

        if return_time_series:
            out["u_wave"] = u_wave
            out["u_turb"] = u_turb
            out["v_wave"] = v_wave
            out["v_turb"] = v_turb
            out["w_wave"] = w_wave
            out["w_turb"] = w_turb

        return out

    def covariance(
        self,
        burst_data: Dict[str, np.ndarray],
        method: str = "cov",
        f_low: Optional[float] = None,
        f_high: Optional[float] = None,
        rho: Optional[float] = 1020,
        phase_kwargs: Optional[dict] = None,
        dmd_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        """Calculate components of the covariance matrix (i.e., the Reynolds
        stress)

        Parameters
        ----------
        burst_data : dict
            Burst data dictionary. Must be in non-beam coordinates.
        method : str
            Method to calculate covariances. Options are:
            - `cov`: Standard covariance calculation using the built-in `np.cov`
            - `spectral_integral`: Integrate the cross-spectrum over a specified frequency range
            - `benilov`: Benilov wave-turbulence decomposition
            - `phase`:  Bricker & Monismith phase-method wave-turbulence decomposition
            - `dmd`: Chavez-Dorado et al. DMD wave-turbulence decomposition.
        f_low : float, optional
            Lower frequency bound (Hz) for spectral integration, by default None
        f_high : float, optional
            Upper frequency bound (Hz) for spectral integration, by default None
        rho : float, optional
            Water density (kg/m^3), by default 1020
        phase_kwargs : dict, optional
            Additional arguments specific to phase decomposition method, by default None. If specified, should include
            keys `f_wave_low` and `f_wave_high` to define the frequency range of the wave band.
        dmd_kwargs : dict, optional
            Additional arguments specific to DMD decomposition method, by default None. If specified, should include
            keys...
        **kwargs
            Additional arguments passed to spectral calculations

        Returns
        -------
        dict
            Dictionary containing covariance components. For method='cov' or
            'spectral_integral', keys are velocity component pairs
            (e.g. 'uu','uv','uw'). For wave decomposition methods, keys include
            turbulence and wave components (e.g. 'uu_turb', 'uu_wave').
        """
        if burst_data["coords"] == "beam":
            raise ValueError(
                "Reynolds stress is not implemented for beam coordinates."
                " Switch to either xyz or enu as a preprocessing step"
            )

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
                f, P_uu = psd(u, fs=self.fs, **kwargs)
                f, P_vv = psd(v, fs=self.fs, **kwargs)
                f, P_ww = psd(w, fs=self.fs, **kwargs)
                f, P_uw = csd(u, w, fs=self.fs, **kwargs)
                f, P_vw = csd(v, w, fs=self.fs, **kwargs)
                f, P_uv = csd(u, v, fs=self.fs, **kwargs)

                start_index, end_index = get_frequency_range(f, f_low, f_high)
                df = np.nanmax(np.diff(f))

                out["uu"][height_idx] = np.sum(np.real(P_uu[start_index:end_index]) * df)
                out["vv"][height_idx] = np.sum(np.real(P_vv[start_index:end_index]) * df)
                out["ww"][height_idx] = np.sum(np.real(P_ww[start_index:end_index]) * df)
                out["uw"][height_idx] = np.sum(np.real(P_uw[start_index:end_index]) * df)
                out["vw"][height_idx] = np.sum(np.real(P_vw[start_index:end_index]) * df)
                out["uv"][height_idx] = np.sum(np.real(P_uv[start_index:end_index]) * df)
        elif method == "benilov":
            out["uu_turb"] = np.empty((n_heights,))
            out["vv_turb"] = np.empty((n_heights,))
            out["ww_turb"] = np.empty((n_heights,))
            out["uw_turb"] = np.empty((n_heights,))
            out["vw_turb"] = np.empty((n_heights,))
            out["uv_turb"] = np.empty((n_heights,))

            out["uu_wave"] = np.empty((n_heights,))
            out["vv_wave"] = np.empty((n_heights,))
            out["ww_wave"] = np.empty((n_heights,))
            out["uw_wave"] = np.empty((n_heights,))
            out["vw_wave"] = np.empty((n_heights,))
            out["uv_wave"] = np.empty((n_heights,))

            for height_idx in range(n_heights):
                u = burst_data["u1"][height_idx, :]
                v = burst_data["u2"][height_idx, :]
                w = burst_data["u3"][height_idx, :]
                p = burst_data["p"][height_idx, :]

                b_out = self.benilov_decomposition(
                    u=u,
                    v=v,
                    w=w,
                    p=p,
                    mab=self.z[height_idx],
                    rho=rho,
                    f_low=f_low,
                    f_high=f_high,
                    **kwargs,
                )

                out["uu_turb"][height_idx] = b_out["uu_turb"]
                out["vv_turb"][height_idx] = b_out["vv_turb"]
                out["ww_turb"][height_idx] = b_out["ww_turb"]
                out["uw_turb"][height_idx] = b_out["uw_turb"]
                out["vw_turb"][height_idx] = b_out["vw_turb"]
                out["uv_turb"][height_idx] = b_out["uv_turb"]

                out["uu_wave"][height_idx] = b_out["uu_wave"]
                out["vv_wave"][height_idx] = b_out["vv_wave"]
                out["ww_wave"][height_idx] = b_out["ww_wave"]
                out["uw_wave"][height_idx] = b_out["uw_wave"]
                out["vw_wave"][height_idx] = b_out["vw_wave"]
                out["uv_wave"][height_idx] = b_out["uv_wave"]

        elif method == "phase":
            # Extract phase method-specific kwargs with error handling
            phase_kwargs = phase_kwargs or {}
            f_wave_low = phase_kwargs.get("f_wave_low", None)
            f_wave_high = phase_kwargs.get("f_wave_high", None)

            out["uu_turb"] = np.empty((n_heights,))
            out["vv_turb"] = np.empty((n_heights,))
            out["ww_turb"] = np.empty((n_heights,))
            out["uw_turb"] = np.empty((n_heights,))
            out["vw_turb"] = np.empty((n_heights,))
            out["uv_turb"] = np.empty((n_heights,))

            out["uu_wave"] = np.empty((n_heights,))
            out["vv_wave"] = np.empty((n_heights,))
            out["ww_wave"] = np.empty((n_heights,))
            out["uw_wave"] = np.empty((n_heights,))
            out["vw_wave"] = np.empty((n_heights,))
            out["uv_wave"] = np.empty((n_heights,))

            for height_idx in range(n_heights):
                u = burst_data["u1"][height_idx, :]
                v = burst_data["u2"][height_idx, :]
                w = burst_data["u3"][height_idx, :]

                p_out = self.phase_decomposition(
                    u=u,
                    v=v,
                    w=w,
                    f_low=f_low,
                    f_high=f_high,
                    f_wave_low=f_wave_low,
                    f_wave_high=f_wave_high,
                    **kwargs,
                )

                out["uu_turb"][height_idx] = p_out["uu_turb"]
                out["vv_turb"][height_idx] = p_out["vv_turb"]
                out["ww_turb"][height_idx] = p_out["ww_turb"]
                out["uw_turb"][height_idx] = p_out["uw_turb"]
                out["vw_turb"][height_idx] = p_out["vw_turb"]
                out["uv_turb"][height_idx] = p_out["uv_turb"]

                out["uu_wave"][height_idx] = p_out["uu_wave"]
                out["vv_wave"][height_idx] = p_out["vv_wave"]
                out["ww_wave"][height_idx] = p_out["ww_wave"]
                out["uw_wave"][height_idx] = p_out["uw_wave"]
                out["vw_wave"][height_idx] = p_out["vw_wave"]
                out["uv_wave"][height_idx] = p_out["uv_wave"]
        elif method == "dmd":
            # Extract dmd method-specific kwargs with error handling
            dmd_kwargs = dmd_kwargs or {}
            f_wave_low = dmd_kwargs.get("f_wave_low", None)
            f_wave_high = dmd_kwargs.get("f_wave_high", None)
            rank_truncation = dmd_kwargs.get("rank_truncation", 0.05)
            time_delay_size = dmd_kwargs.get("time_delay_size", None)
            return_time_series = dmd_kwargs.get("return_time_series", False)

            out["uu_turb"] = np.empty((n_heights,))
            out["vv_turb"] = np.empty((n_heights,))
            out["ww_turb"] = np.empty((n_heights,))
            out["uw_turb"] = np.empty((n_heights,))
            out["vw_turb"] = np.empty((n_heights,))
            out["uv_turb"] = np.empty((n_heights,))

            out["uu_wave"] = np.empty((n_heights,))
            out["vv_wave"] = np.empty((n_heights,))
            out["ww_wave"] = np.empty((n_heights,))
            out["uw_wave"] = np.empty((n_heights,))
            out["vw_wave"] = np.empty((n_heights,))
            out["uv_wave"] = np.empty((n_heights,))

            if return_time_series:
                N = burst_data["u1"].shape[1]
                out["u_turb"] = np.empty((n_heights, N))
                out["v_turb"] = np.empty((n_heights, N))
                out["w_turb"] = np.empty((n_heights, N))
                out["u_wave"] = np.empty((n_heights, N))
                out["v_wave"] = np.empty((n_heights, N))
                out["w_wave"] = np.empty((n_heights, N))

            for height_idx in range(n_heights):
                u = burst_data["u1"][height_idx, :]
                v = burst_data["u2"][height_idx, :]
                w = burst_data["u3"][height_idx, :]

                d_out = self.dmd(
                    u=u,
                    v=v,
                    w=w,
                    f_wave_low=f_wave_low,
                    f_wave_high=f_wave_high,
                    rank_truncation=rank_truncation,
                    time_delay_size=time_delay_size,
                    return_time_series=return_time_series,
                )

                out["uu_turb"][height_idx] = d_out["uu_turb"]
                out["vv_turb"][height_idx] = d_out["vv_turb"]
                out["ww_turb"][height_idx] = d_out["ww_turb"]
                out["uw_turb"][height_idx] = d_out["uw_turb"]
                out["vw_turb"][height_idx] = d_out["vw_turb"]
                out["uv_turb"][height_idx] = d_out["uv_turb"]

                out["uu_wave"][height_idx] = d_out["uu_wave"]
                out["vv_wave"][height_idx] = d_out["vv_wave"]
                out["ww_wave"][height_idx] = d_out["ww_wave"]
                out["uw_wave"][height_idx] = d_out["uw_wave"]
                out["vw_wave"][height_idx] = d_out["vw_wave"]
                out["uv_wave"][height_idx] = d_out["uv_wave"]

                if return_time_series:
                    out["u_turb"][height_idx, :] = d_out["u_turb"]
                    out["v_turb"][height_idx, :] = d_out["v_turb"]
                    out["w_turb"][height_idx, :] = d_out["w_turb"]
                    out["u_wave"][height_idx, :] = d_out["u_wave"]
                    out["v_wave"][height_idx, :] = d_out["v_wave"]
                    out["w_wave"][height_idx, :] = d_out["w_wave"]
        else:
            raise IOError(f"Unrecognized method {method}")

        return out

    @staticmethod
    def _calcJii(sig1, sig2, sig3, u1, u2):
        """Calculates J11, J22, and J33, the diagonal elements of equation A.13
        in Gerbi et al.

        (2009)
        """
        # Initializing coordinate arrays
        r_len = 120
        r = np.logspace(-2, 4, r_len)
        R = 1 / r
        theta = np.linspace(0, np.pi, r_len // 4)
        phi = np.linspace(0, 2 * np.pi, r_len // 4)

        # Precompute trigonometric functions and associated variables
        cos_theta = np.cos(theta)  # (Ntheta,)
        sin_theta = np.sin(theta)  # (Ntheta,)
        cos_phi = np.cos(phi)  # (Nphi,)
        sin_phi = np.sin(phi)  # (Nphi,)

        # Want shape (Ntheta, Nphi)
        G_squared = (sin_theta**2)[:, np.newaxis] * (cos_phi**2 / sig1**2 + sin_phi**2 / sig2**2)[np.newaxis, :] + (
            cos_theta**2
        )[:, np.newaxis] / sig3**2

        # Also shape (Ntheta, Nphi)
        P11 = (1 / G_squared) * (
            (sin_theta**2)[:, np.newaxis] * (sin_phi**2)[np.newaxis, :] / sig2**2
            + (cos_theta**2)[:, np.newaxis] / sig3**2
        )
        P22 = (1 / G_squared) * (
            (sin_theta**2)[:, np.newaxis] * (cos_phi**2)[np.newaxis, :] / sig1**2
            + (cos_theta**2)[:, np.newaxis] / sig3**2
        )
        P33 = ((sin_theta**2)[:, np.newaxis] / G_squared) * (cos_phi**2 / sig1**2 + sin_phi**2 / sig2**2)[np.newaxis, :]
        P11_3 = P11[..., np.newaxis]
        P22_3 = P22[..., np.newaxis]
        P33_3 = P33[..., np.newaxis]

        # Defining k_squared (Ntheta, Nphi, Nr)
        R_3 = R[np.newaxis, np.newaxis, :]
        G_squared_3 = G_squared[..., np.newaxis]

        # (Ntheta, Nphi)
        R0 = (u1 / sig1) * sin_theta[:, np.newaxis] * cos_phi[np.newaxis, :] + (u2 / sig2) * sin_theta[
            :, np.newaxis
        ] * sin_phi[np.newaxis, :]
        R0_3 = R0[..., np.newaxis]  # (Ntheta, Nphi, 1)

        # Innermost integral
        I3 = R_3 ** (2 / 3) * np.exp(-((R0_3 - R_3) ** 2) / 2)

        # Middle integral
        # Gets a negative sign so that we go from R = 0 -> infinity rather than R = infinity -> zero
        I2_11 = -np.trapezoid(
            G_squared_3 ** (-11 / 6) * sin_theta[:, np.newaxis, np.newaxis] * P11_3 * I3,
            R,
            axis=2,
        )
        I2_22 = -np.trapezoid(
            G_squared_3 ** (-11 / 6) * sin_theta[:, np.newaxis, np.newaxis] * P22_3 * I3,
            R,
            axis=2,
        )
        I2_33 = -np.trapezoid(
            G_squared_3 ** (-11 / 6) * sin_theta[:, np.newaxis, np.newaxis] * P33_3 * I3,
            R,
            axis=2,
        )

        # Outer integral
        I1_11 = np.trapezoid(I2_11, phi, axis=-1)
        I1_22 = np.trapezoid(I2_22, phi, axis=-1)
        I1_33 = np.trapezoid(I2_33, phi, axis=-1)

        J11 = (1 / (2 * (2 * np.pi) ** (3 / 2))) * (1 / (sig1 * sig2 * sig3)) * np.trapezoid(I1_11, theta, axis=-1)
        J22 = (1 / (2 * (2 * np.pi) ** (3 / 2))) * (1 / (sig1 * sig2 * sig3)) * np.trapezoid(I1_22, theta, axis=-1)
        J33 = (1 / (2 * (2 * np.pi) ** (3 / 2))) * (1 / (sig1 * sig2 * sig3)) * np.trapezoid(I1_33, theta, axis=-1)

        return J11, J22, J33

    def dissipation(
        self, burst_data: Dict[str, np.ndarray], f_low: float, f_high: float, **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Estimate the dissipation rate of TKE using the Gerbi et al. (2009) spectral curve fitting method. This is nearly
        equivalent to the Feddersen et al. (2007) method, but it uses a more efficient numerical integration and
        estimates dissipation with a least squares fit rather than a mean over the inertial range.

        Parameters
        ----------
        f_low : float
            Lower frequency bound (Hz) for inertial subrange where -5/3 law applies
        f_high : float
            Upper frequency bound (Hz) for inertial subrange where -5/3 law applies
        **kwargs
            Additional arguments passed to spectral_utils.psd.
            See spectral_utils.psd for parameter definitions.

        Returns
        -------
        Dictionary with the following keys/values at each height:
            eps : float
                dissipation rate of TKE (m^2/s^3)
            noise : float
                intercept from dissipation linear regression
            quality_flag : int
                1 for good eps estimate, 0 for bad eps estimate.
                Defined based on Gerbi Eq. 11

        References
        ----------
        Feddersen, F., Trowbridge, J. H., & Williams, A. J. (2007). Vertical structure of dissipation in the nearshore.
            Journal of Physical Oceanography, 37(7), 1764-1777.

        Gerbi, G. P., Trowbridge, J. H., Terray, E. A., Plueddemann, A. J., & Kukulka, T. (2009). Observations of
            turbulence in the ocean surface boundary layer: Energetics and transport. Journal of Physical Oceanography,
            39(5), 1077-1096.
        """

        def spectral_fit(u, v, w, f_low, f_high, **kwargs):
            """Carries out the spectral curve fit."""
            if np.all(np.isnan(u)) or np.all(np.isnan(v)) or np.all(np.isnan(w)):
                return np.nan, np.nan, 0
            omega_range = [2 * np.pi * f_low, 2 * np.pi * f_high]
            alpha = 1.5

            w_prime = sig.detrend(w, type="linear")
            fw, Pw_f = psd(w_prime, self.fs, onesided=False, **kwargs)

            omega = 2 * np.pi * fw
            Pw_omega = Pw_f / (2 * np.pi)

            inertial_indices = (omega >= omega_range[0]) & (omega <= omega_range[1])
            omega_inertial = omega[inertial_indices]
            Pw_inertial = Pw_omega[inertial_indices]

            sig1 = np.nanstd(u)
            sig2 = np.nanstd(v)
            sig3 = np.nanstd(w)

            u1 = np.nanmean(u)
            u2 = np.nanmean(v)

            _, _, J33 = self._calcJii(sig1, sig2, sig3, u1, u2)

            # linear regression
            X = J33 * alpha * (omega_inertial ** (-5 / 3))
            y = Pw_inertial
            slope, intercept, *_ = linregress(X, y)
            eps23 = slope
            noise = intercept

            if eps23 < 0:
                return np.nan, np.nan, 0
            else:
                eps = eps23 ** (3 / 2)
                if noise < J33 * alpha * (eps ** (2 / 3)) * (omega_range[0] ** (-5 / 3)):
                    quality_flag = 1
                else:
                    quality_flag = 0

            return eps, noise, quality_flag

        if burst_data["coords"] == "beam":
            raise ValueError(
                "Dissipation is not implemented for beam coordinates."
                " Switch to either xyz or enu as a preprocessing step"
            )

        out = {}
        n_heights = self.n_heights
        out["eps"] = np.empty((n_heights,))
        out["noise"] = np.empty((n_heights,))
        out["quality_flag"] = np.empty((n_heights,), dtype=int)
        for height_idx in range(n_heights):
            u = burst_data["u1"][height_idx, :]
            v = burst_data["u2"][height_idx, :]
            w = burst_data["u3"][height_idx, :]
            (eps, noise, quality_flag) = spectral_fit(u, v, w, f_low, f_high, **kwargs)
            out["eps"][height_idx] = eps
            out["noise"][height_idx] = noise
            out["quality_flag"][height_idx] = quality_flag

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
        if burst_data["coords"] == "beam":
            raise ValueError(
                "TKE is not implemented for beam coordinates. Switch to either xyz or enu as a preprocessing step"
            )

        u1_bar = np.mean(burst_data["u1"], axis=1, keepdims=True)
        u2_bar = np.mean(burst_data["u2"], axis=1, keepdims=True)
        u3_bar = np.mean(burst_data["u3"], axis=1, keepdims=True)

        u1_prime = burst_data["u1"] - u1_bar
        u2_prime = burst_data["u2"] - u2_bar
        u3_prime = burst_data["u3"] - u3_bar

        tke_prime = 0.5 * (u1_prime**2 + u2_prime**2 + u3_prime**2)
        tke_out = np.mean(tke_prime, axis=1)
        return tke_out

    def directional_wave_statistics(
        self,
        burst_data: dict,
        band_definitions: Optional[dict] = None,
        sea_correction: Optional[bool] = True,
        f_cutoff: Optional[float] = 1.0,
        rho: Optional[float] = 1020,
        **kwargs,
    ) -> dict:
        """Calculate directional wave statistics from velocity and pressure
        measurements.

        Parameters
        ----------
        burst_data : dict
            Burst dictionary containing u1/u2 velocities and pressure. Cannot be in beam coordinates.
        band_definitions : dict, optional
            Dictionary defining frequency bands for spectral sums of the form
             `{"infragravity": (f_low, f_high), "swell": (f_low, f_high), "sea": (f_low, f_high)}`
             If None, uses default bands:
            - infragravity: 1/250 to 1/25 Hz
            - swell: 1/25 to 0.2 Hz
            - sea: 0.2 to 0.5 Hz
            Statistics for the full frequency range (`all`) will be calculated as well.
        sea_correction : bool, optional
            Whether to apply Jones-Monismith correction for sea waves, by default True
        f_cutoff : float, optional
            Upper bound for spectral integration to avoid high frequency noise. Defaults to 1.0 Hz.
        rho : float
            Water density (kg/m^3)
        **kwargs
            Additional arguments passed to spectral analysis functions

        Returns
        -------
        dict
            Dictionary of wave statistics. Scalar variables (e.g. `Hsig_all`) have shape
            `(n_heights,)`; spectral variables (e.g. `P_uu`) have shape `(n_heights, n_freqs)`.

        References
        ----------
        Herbers, T. H. C., Elgar, S., & Guza, R. T. (1999). Directional spreading of waves in the nearshore. Journal of
            Geophysical Research: Oceans, 104(C4), 7683-7693.

        Jones, N. L., & Monismith, S. G. (2007). Measuring short‐period wind waves in a tidally forced environment with
            a subsurface pressure gauge. Limnology and Oceanography: Methods, 5(10), 317-327.

        Kumar, N., Cahl, D. L., Crosby, S. C., & Voulgaris, G. (2017). Bulk versus spectral wave parameters:
            Implications on stokes drift estimates, regional wave modeling, and HF radars applications. Journal of
            Physical Oceanography, 47(6), 1413-1431.

        Madsen, O. S. (1994). Spectral wave-current bottom boundary layer flows. In Coastal engineering 1994 (pp.
            384-398).

        Mei, C. C., Stiassnie, M. A., & Yue, D. K. P. (2005). Theory and applications of ocean surface waves: Part 1:
            linear aspects.

        Wiberg, P. L., & Sherwood, C. R. (2008). Calculating wave-generated bottom orbital velocities from surface-wave
            parameters. Computers & Geosciences, 34(10), 1243-1262.
        """
        if "p" not in burst_data.keys():
            raise ValueError("Pressure must be included in dataset to calculate directional wave statistics")

        if burst_data["coords"] == "beam":
            raise ValueError("Directional wave statistics not implemented for beam coordinates.")

        n_heights = self.n_heights
        results = []
        for height_idx in range(n_heights):
            u = burst_data["u1"][height_idx, :]
            v = burst_data["u2"][height_idx, :]
            p = burst_data["p"][height_idx, :]
            results.append(
                self._wave_worker(
                    u=u,
                    v=v,
                    p=p,
                    mab=self.z[height_idx],
                    rho=rho,
                    band_definitions=band_definitions,
                    sea_correction=sea_correction,
                    f_cutoff=f_cutoff,
                    **kwargs,
                )
            )

        return {key: np.array([r[key] for r in results]) for key in results[0]}

    def _wave_worker(
        self,
        u: np.ndarray,
        v: np.ndarray,
        p: np.ndarray,
        mab: float,
        rho: float,
        band_definitions: Optional[dict] = None,
        sea_correction: Optional[bool] = True,
        f_cutoff: Optional[float] = 1.0,
        **kwargs,
    ) -> dict:
        """Helper function for directional wave statistics."""
        # Calculate water depth (m) prior to detrending
        h = 1e4 * np.nanmean(p) / (rho * g) + mab

        u = sig.detrend(u, type="linear")
        v = sig.detrend(v, type="linear")
        p = sig.detrend(p, type="linear")

        # Calculating spectra
        f, P_uu = psd(u, fs=self.fs, **kwargs)
        f, P_vv = psd(v, fs=self.fs, **kwargs)
        f, P_pp = psd(p, fs=self.fs, **kwargs)
        f, P_uv = csd(u, v, fs=self.fs, **kwargs)
        f, P_pu = csd(p, u, fs=self.fs, **kwargs)
        f, P_pv = csd(p, v, fs=self.fs, **kwargs)
        df = np.max(np.diff(f))

        # Frequency band definitions
        if band_definitions is None:
            band_definitions = {
                "infragravity": (1 / 250, 1 / 25),
                "swell": (1 / 25, 1 / 5),
                "sea": (1 / 5, 1 / 2),
            }

        f_bands = {}
        for band_name, (f_low, f_high) in band_definitions.items():
            f_bands[band_name] = ((f >= f_low) & (f < f_high) & (f < f_cutoff))
        f_bands["all"] = ((f > 0) & (f < f_cutoff))

        # Getting sea surface elevation spectrum
        omega = 2 * np.pi * f
        k = get_wavenumber(omega, h)
        attenuation_correction = 1e4 * np.cosh(k * h) / (rho * g * np.cosh(k * mab))
        P_etaeta = P_pp * (attenuation_correction**2)

        if sea_correction:
            P_etaeta = jones_monismith_correction(P_etaeta, P_pp, f)

        # Directional moments (Herbers et al., 1999, Appendix)
        a1 = np.real(P_pu / np.sqrt(P_pp * (P_uu + P_vv)))
        b1 = np.real(P_pv / np.sqrt(P_pp * (P_uu + P_vv)))
        dir1 = np.degrees(np.arctan2(b1, a1))
        spread1 = np.degrees(np.sqrt(2 * (1 - (a1 * np.cos(np.radians(dir1)) + b1 * np.sin(np.radians(dir1))))))

        a2 = np.real((P_uu - P_vv) / (P_uu + P_vv))
        b2 = np.real(2 * P_uv / (P_uu + P_vv))
        dir2 = np.degrees(np.arctan2(b2, a2) / 2)
        spread2 = np.degrees(
            np.sqrt(0.5 * (1 - (a2 * np.cos(2 * np.radians(dir2)) + b2 * np.sin(2 * np.radians(dir2)))))
        )

        # Phase and group velocity
        cp = omega / k
        cg = get_cg(k, h)

        # Radiation stress -- Mei et al. Ch 11.3
        dir_rad = np.deg2rad(dir1)
        E = rho * g * P_etaeta
        n = cg / cp
        Sxx = (E / 2) * (2 * n * np.cos(dir_rad) ** 2 + (2 * n - 1))
        Syy = (E / 2) * (2 * n * np.sin(dir_rad) ** 2 + (2 * n - 1))
        Sxy = E * n * np.sin(dir_rad) * np.cos(dir_rad)

        # Orbital velocity, basically following Wiberg & Sherwood (2008) but excluding
        # the factor of sqrt(2) (see Madsen 1994)
        # Time domain calculation
        u_prime = u - np.nanmean(u)
        v_prime = v - np.nanmean(v)
        u_orb_var = np.sqrt((np.nanvar(u_prime) + np.nanvar(v_prime)))

        # Spectral calculation
        u_orb_spec = np.sqrt(np.sum((P_uu + P_vv) * df))

        # Setting up output dictionary and storing the spectral output
        out = {}
        out["f"] = f
        out["df"] = df
        out["P_uu"] = P_uu
        out["P_vv"] = P_vv
        out["P_pp"] = P_pp
        out["P_uv"] = P_uv
        out["P_pu"] = P_pu
        out["P_pv"] = P_pv
        out["P_etaeta"] = P_etaeta
        out["a1"] = a1
        out["b1"] = b1
        out["a2"] = a2
        out["b2"] = b2
        out["dir1"] = dir1
        out["spread1"] = spread1
        out["dir2"] = dir2
        out["spread2"] = spread2
        out["Sxx"] = Sxx
        out["Syy"] = Syy
        out["Sxy"] = Sxy
        out["cp"] = cp
        out["cg"] = cg
        out["u_orb_var"] = u_orb_var
        out["u_orb_spec"] = u_orb_spec

        # Looping over the frequency bands and adding bulk (integrated) parameters
        for band_name, band_indices in f_bands.items():
            # Significant and rms wave height
            out[f"Hsig_{band_name}"] = 4 * np.sqrt(np.sum(P_etaeta[band_indices] * df))
            out[f"Hrms_{band_name}"] = np.sqrt(8 * np.sum(P_etaeta[band_indices] * df))

            # Mean frequency and period
            out[f"fm_{band_name}"] = np.sum(f[band_indices] * P_etaeta[band_indices]) / np.sum(P_etaeta[band_indices])
            out[f"Tm_{band_name}"] = 1 / out[f"fm_{band_name}"]

            # Peak frequency and period
            out[f"fp_{band_name}"] = f[band_indices][np.argmax(P_etaeta[band_indices])]
            out[f"Tp_{band_name}"] = 1 / out[f"fp_{band_name}"]

            # Directions
            out[f"a1_{band_name}"] = np.sum(a1[band_indices] * P_etaeta[band_indices]) / np.sum(P_etaeta[band_indices])
            out[f"b1_{band_name}"] = np.sum(b1[band_indices] * P_etaeta[band_indices]) / np.sum(P_etaeta[band_indices])
            out[f"a2_{band_name}"] = np.sum(a2[band_indices] * P_etaeta[band_indices]) / np.sum(P_etaeta[band_indices])
            out[f"b2_{band_name}"] = np.sum(b2[band_indices] * P_etaeta[band_indices]) / np.sum(P_etaeta[band_indices])

            out[f"dir1_{band_name}"] = np.degrees(np.arctan2(out[f"b1_{band_name}"], out[f"a1_{band_name}"]))
            out[f"dir2_{band_name}"] = np.degrees(np.arctan2(out[f"b2_{band_name}"], out[f"a2_{band_name}"]) / 2)
            out[f"spread1_{band_name}"] = np.degrees(
                np.sqrt(
                    2
                    * (
                        1
                        - (
                            out[f"a1_{band_name}"] * np.cos(np.deg2rad(out[f"dir1_{band_name}"]))
                            + out[f"b1_{band_name}"] * np.sin(np.deg2rad(out[f"dir1_{band_name}"]))
                        )
                    )
                )
            )
            out[f"spread2_{band_name}"] = np.degrees(
                np.sqrt(
                    0.5
                    * (
                        1
                        - (
                            out[f"a2_{band_name}"] * np.cos(2 * np.deg2rad(out[f"dir2_{band_name}"]))
                            + out[f"b2_{band_name}"] * np.sin(2 * np.deg2rad(out[f"dir2_{band_name}"]))
                        )
                    )
                )
            )

            # Radiation stress
            out[f"Sxx_{band_name}"] = np.sum(Sxx[band_indices] * df)
            out[f"Syy_{band_name}"] = np.sum(Syy[band_indices] * df)
            out[f"Sxy_{band_name}"] = np.sum(Sxy[band_indices] * df)

            # Stokes drift, both bulk and spectral (unfortunately different). See Kumar et al. 2017, Appendix.
            omega_peak = 2 * np.pi / out[f"Tp_{band_name}"]
            k_peak = get_wavenumber(omega_peak, h)
            out[f"Us_bulk_{band_name}"] = (
                (out[f"Hsig_{band_name}"] ** 2 * omega_peak * k_peak / 16)
                * np.cosh(2 * k_peak * mab)
                / (np.sinh(k_peak * h) ** 2)
                * np.cos(np.radians(out[f"dir1_{band_name}"]))
            )
            out[f"Vs_bulk_{band_name}"] = (
                (out[f"Hsig_{band_name}"] ** 2 * omega_peak * k_peak / 16)
                * np.cosh(2 * k_peak * mab)
                / (np.sinh(k_peak * h) ** 2)
                * np.sin(np.radians(out[f"dir1_{band_name}"]))
            )

            # Spectral Stokes drift
            out[f"Us_spec_{band_name}"] = np.sum(
                P_etaeta[band_indices]
                * omega[band_indices]
                * k[band_indices]
                * (np.cosh(2 * k[band_indices] * mab) / (np.sinh(k[band_indices] * h) ** 2))
                * np.cos(np.radians(dir1[band_indices]))
                * df
            )
            out[f"Vs_spec_{band_name}"] = np.sum(
                P_etaeta[band_indices]
                * omega[band_indices]
                * k[band_indices]
                * (np.cosh(2 * k[band_indices] * mab) / (np.sinh(k[band_indices] * h) ** 2))
                * np.sin(np.radians(dir1[band_indices]))
                * df
            )
        return out

    def subsample(self, start_idx: int, end_idx: int):
        """Subsample the ADV object between files[start_idx] and
        files[end_idx].

        Parameters
        ----------
        start_idx : int
            First file to include in subsampling
        end_idx : int
            Upper bound (exclusive) on file index in subsampling


        Returns
        -------
        new_adv : ADV
            Subsampled ADV object
        """
        new_adv = self.__class__(
            files=self.files[start_idx:end_idx],
            name_map=self.name_map,
            deployment_type=self.deployment_type,
            fs=self.fs,
            z=self.z,
            data_keys=self.data_keys,
            source_coords=self.source_coords,
            orientation=self.orientation,
        )
        if self._preprocess_enabled:
            new_adv.set_preprocess_opts(self._preprocess_opts)
        return new_adv

    @property
    def output_coords(self):
        if self._preprocess_enabled and self._rotate:
            return self._rotate.get("coords_out", self.source_coords)
        return self.source_coords
