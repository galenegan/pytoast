import numpy as np
import scipy.signal as sig
from scipy.stats import linregress
from typing import Optional, Union, List, Dict, Any
from src.utils.despike_utils import apply_threshold_despike
from src.utils.spectral_utils import psd, csd, get_frequency_range
from src.utils.base_instrument import BaseInstrument
from src.utils.constants import GRAVITATIONAL_ACCELERATION as g
from src.utils.rotate_utils import (
    align_with_principal_axis,
    align_with_flow,
    rotate_velocity_by_theta,
)


class Sonic(BaseInstrument):
    def __init__(
        self,
        files: Union[str, List],
        name_map: dict,
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[float, int, List[Union[float, int]]]] = None,
        data_keys: Optional[Union[str, List[str]]] = None,
        path_length: float = 0.15,
    ):
        """
        Initialize a Sonic anemometer data manager.

        Parameters
        ----------
        files : str or List[str]
            Path(s) to data files. If a list, each element is treated as a file containing data from
            an individual burst period. Supported formats: .npy (saved as a dict), .mat (saved as a
            MATLAB struct), .csv (variables in columns). If variables are two-dimensional, the larger
            dimension is assumed to be time and the shorter dimension a vertical coordinate; in that
            case a matching `z` list must be provided.
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
        fs : int or float, optional
            Sampling frequency (Hz). Inferred (rounded to 2 decimal places) from the "time" variable
            if not provided.
        z : float, int, or List[float, int], optional
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
        Sonic.validate_inputs(files_list, name_map, fs, z, data_keys, path_length)
        super().__init__(files, name_map, fs, z, data_keys)

    @staticmethod
    def validate_inputs(
        files: Union[str, List],
        name_map: dict,
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[float, int, List[Union[float, int]]]] = None,
        data_keys: Optional[Union[str, List[str]]] = None,
        path_length: Optional[float] = 0.15,
    ):

        # General validation
        BaseInstrument.validate_common_inputs(files, name_map, fs, z)

        # Instrument-specific requirements
        required_keys = ["u1", "u2", "u3"]

        for key in required_keys:
            if key not in name_map:
                raise ValueError(f"`name_map` must include a mapping for '{key}'")

        if not isinstance(path_length, float):
            raise TypeError("`path length` must be a float")

    def set_preprocess_opts(self, opts: Dict[str, Any]):
        """Enable preprocessing for all subsequent burst loads using the options defined in the input dictionary.

        Parameters
        ----------
        opts : Dict[str, Any]
            Options for preprocessing. Currently supports the following keys/values
            {
                "rotate": "align_wind", "align_principal", or (horizontal_angle(s), vertical_angle(s))
                "despike": bool
            }
        """

        self._preprocess_opts = opts
        self._preprocess_enabled = True

        self._despike = opts.get("despike", {})
        if self._despike:
            self._despike_opts = {key: val for key, val in self._despike.items()}

        self._rotate = opts.get("rotate", {})
        self._cached_idx = None
        self._cached_data = None

    def _apply_preprocessing(self, burst_data):

        if not self._preprocess_enabled:
            return burst_data

        if self._despike:
            for key in self.beam_keys:
                burst_data[key] = apply_threshold_despike(burst_data[key], **self._despike_opts)

        if self._rotate:
            flow_rotation = self._rotate.get("flow_rotation")
            if flow_rotation:
                burst_data = self._apply_flow_rotation(burst_data, flow_rotation)

        return burst_data

    def _apply_flow_rotation(self, burst_data, flow_rotation):
        """
        Rotate u1/u2/u3 to align with the burst-mean flow direction.

        Parameters
        ----------
        burst_data : dict
            Burst data dictionary. Must be in non-beam coordinates.
        flow_rotation : str or tuple
            "align_principal", "align_current", or (theta_h_deg, theta_v_deg).

        Returns
        -------
        dict
            burst_data with u1/u2/u3 rotated and burst_data["rotation"] set.
        """
        if isinstance(flow_rotation, str):
            if flow_rotation == "align_principal":
                theta_h, theta_v = align_with_principal_axis(burst_data["u1"], burst_data["u2"], burst_data["u3"])
            elif flow_rotation == "align_current":
                theta_h, theta_v = align_with_flow(burst_data["u1"], burst_data["u2"], burst_data["u3"])
            else:
                raise ValueError(
                    f"Invalid flow_rotation '{flow_rotation}'. Must be 'align_principal' or 'align_current'."
                )
        elif isinstance(flow_rotation, tuple):
            theta_h, theta_v = flow_rotation
        else:
            raise TypeError(f"flow_rotation must be a str or tuple, got {type(flow_rotation)}")

        u1_new, u2_new, u3_new = rotate_velocity_by_theta(
            burst_data["u1"], burst_data["u2"], burst_data["u3"], theta_h, theta_v
        )
        burst_data["u1"] = u1_new
        burst_data["u2"] = u2_new
        burst_data["u3"] = u3_new
        burst_data["rotation"] = flow_rotation
        return burst_data

    def dissipation(
        self,
        burst_data: dict,
        f_low: float,
        f_high: float,
        henjes_correction: bool,
        **kwargs,
    ) -> np.ndarray:
        """

        Parameters
        ----------
        f_low
        f_high
        henjes_correction
        kwargs

        Returns
        -------

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
            f, S = psd(u_prime, fs=self.fs, onesided=False, **kwargs)

            if henjes_correction:
                fs = kwargs.get("fs", self.fs)
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
            u = burst_data["u"][height_idx, :]
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
        out = {}
        n_heights = self.n_heights
        if method == "cov":
            u_bar = np.mean(burst_data["u"], axis=1, keepdims=True)
            v_bar = np.mean(burst_data["v"], axis=1, keepdims=True)
            w_bar = np.mean(burst_data["w"], axis=1, keepdims=True)
            u_prime = burst_data["u"] - u_bar
            v_prime = burst_data["v"] - v_bar
            w_prime = burst_data["w"] - w_bar

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
                u = burst_data["u"][height_idx, :]
                v = burst_data["v"][height_idx, :]
                w = burst_data["w"][height_idx, :]

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

    def tke(self, burst_data: Dict[str, np.ndarray]):
        u_bar = np.mean(burst_data["u"], axis=1, keepdims=True)
        v_bar = np.mean(burst_data["v"], axis=1, keepdims=True)
        w_bar = np.mean(burst_data["w"], axis=1, keepdims=True)

        u_prime = burst_data["u"] - u_bar
        v_prime = burst_data["v"] - v_bar
        w_prime = burst_data["w"] - w_bar

        tke_prime = 0.5 * (u_prime**2 + v_prime**2 + w_prime**2)
        tke_out = np.mean(tke_prime, axis=1)
        return tke_out

    def buoyancy_flux(self, burst_data: Dict[str, np.ndarray]):
        if "Ts" not in burst_data:
            raise ValueError("Cannot compute buoyancy flux without sonic temperature data")

        Ts_bar = np.mean(burst_data["Ts"], axis=1, keepdims=True)
        w_bar = np.mean(burst_data["w"], axis=1, keepdims=True)
        Ts_prime = burst_data["Ts"] - Ts_bar
        w_prime = burst_data["w"] - w_bar
        B = g * np.mean(Ts_prime * w_prime, axis=1) / (Ts_bar + 273.15)
        return B

    def subsample(self, start_idx, end_idx):
        new_sonic = self.__class__(
            self.files[start_idx:end_idx],
            self.name_map,
            self.fs,
            self.z,
            self.data_keys,
            self.path_length,
        )
        if self._preprocess_enabled:
            new_sonic.set_preprocess_opts(self._preprocess_opts)
        return new_sonic
