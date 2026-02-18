import numpy as np
import os
import scipy.signal as sig
import xarray as xr
from sklearn.linear_model import LinearRegression
from typing import Optional, Union, List, Dict, Any, Tuple
from src.utils.spectral_utils import psd, csd, get_frequency_range
from src.utils.base_instrument import BaseInstrument
from src.utils.constants import GRAVITATIONAL_ACCELERATION as g
from src.utils.rotate_utils import align_with_principal_axis, align_with_flow, rotate_velocity_by_theta

class Sonic(BaseInstrument):

    def __init__(
        self,
        files: Union[str, List],
        name_map: dict,
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[float, int, List[Union[float, int]]]] = None,
        L: float = 0.15,
    ):
        self.path_length = L
        super().__init__(files, name_map, fs, z)

    @staticmethod
    def validate_inputs(
        files: Union[str, List],
        name_map: dict,
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[float, int, List[Union[float, int]]]] = None,
        L: Optional[float] = 0.15,
    ):

        # General validation
        BaseInstrument.validate_common_inputs(files, name_map, fs, z)

        # Instrument-specific requirements
        required_keys = ["u", "v", "w"]

        for key in required_keys:
            if key not in name_map:
                raise ValueError(f"`name_map` must include a mapping for '{key}'")

        if not isinstance(L, float):
            raise TypeError("path length `L` must be a float")

    @classmethod
    def from_raw(
        cls,
        files: Union[str, List],
        name_map: dict,
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[float, int, List[Union[float, int]]]] = None,
        L: Optional[float] = 0.15,
    ):
        """
        Initializes a new Sonic object from data files.

        Parameters
        ----------
        files : str or List[str]
            If str, must be a path to a netCDF file, .mat file, or zarr file store that contains the entire dataset
            you wish to load. If list, the elements of the list will be interpreted as files containing data from
            individual measurement burst periods. Supported burst file types are .npy (assuming it was saved as a
            dictionary), .mat (assuming it was saved as a Matlab Struct) and .csv (with variables in separate
            columns). If the variables associated with a particular name are two-dimensional, then the larger
            dimension is assumed to be time and the shorter dimension is assumed to be a vertical coordinate. In this
            case, a "z" list must be passed as an argument with a length matching the size of the shorter dimension

        name_map : dict
            a dictionary of the form:
            {
                "u": "x-velocity variable name" or ["var 1", "var 2", etc.],
                "v": "y-velocity variable name" or ["var 1", "var 2", etc.],
                "w": "z-velocity variable name" or ["var 1", "var 2", etc.],
                "Ts": "sonic temperature name" or ["var 1", "var 2", etc.],
                "time": "time variable name" or ["var 1", "var 2", etc.],
            }
            Of these, "Ts" and "time" are optional, but an error will be raised if "time" is not specified and a
            sampling frequency is also not specified. Lists should be provided if data from multiple instruments is
            stored in multiple variables (as opposed to a 2d array).

        z : float, int or List[float, int], optional
            mean height above the surface (m) for each instrument. If not specified, the height coordinate in the
            resulting Sonic object will be integer indices.

        fs : int or float, optional
            sampling frequency (Hz). If not specified, will be inferred (and rounded to 2 decimal places)
            from name_map["time"] values

        L : float, optional
            path length (m). Required to implement the Henjes Correction to the spectral curve fit in
            Sonic.dissipation. Defaults to 0.15 if not specified


        Returns
        -------
        Sonic object

        """
        Sonic.validate_inputs(files, name_map, fs, z, L)

        return cls(files, name_map, fs, z, L)

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

        self._preprocess_enabled = True

        self._rotate = opts.get("rotate", "align_wind")
        self._cached_idx = None
        self._cached_data = None

    def _apply_preprocessing(self, burst_data):

        if isinstance(self._rotate, str):
            if self._rotate == "align_principal":
                theta_h, theta_v = align_with_principal_axis(burst_data)
            elif self._rotate == "align_wind":
                theta_h, theta_v = align_with_flow(burst_data)
            else:
                raise ValueError(f"Invalid rotation option '{self._rotate}'")
        else:
            theta_h, theta_v = self._rotate

        if np.sum(np.abs(theta_v)) != 0.0 or np.sum(np.abs(theta_h)) != 0.0:
            burst_data = rotate_velocity_by_theta(burst_data, theta_h, theta_v)

        return burst_data


    def dissipation(
        self, burst_data: dict, f_low: float, f_high: float, henjes_correction: bool, **kwargs
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
            u: np.ndarray, f_low: float, f_high: float, henjes_correction: bool = True, **kwargs
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
                reg = LinearRegression().fit(X.reshape(-1, 1), y)
                eps23 = reg.coef_[0].item()
                if eps23 < 0:
                    eps = np.nan
                else:
                    eps = eps23 ** (3 / 2)
            return eps

        n_heights = self.n_heights
        eps = np.empty((n_heights,))
        for height_idx in range(n_heights):
            u = burst_data["u"][height_idx, :]
            eps[height_idx] = spectral_fit(u, henjes_correction=henjes_correction, f_low=f_low, f_high=f_high, **kwargs)

        return eps

    def covariance(
        self,
        burst_data: Dict[str, np.ndarray],
        method: str = "cov",
        f_low: Optional[float] = None,
        f_high: Optional[float] = None,
        **kwargs
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

            out["uu"] = np.mean(u_prime ** 2, axis=1)
            out["vv"] = np.mean(v_prime ** 2, axis=1)
            out["ww"] = np.mean(w_prime ** 2, axis=1)
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
        files = self.files[start_idx:end_idx]
        return self.__class__(files, self.name_map, self.fs, self.z, self.path_length)


if __name__ == "__main__":
    import time

    t0 = time.time()
    # Testing this out
    file_path = "/Users/ea-gegan/Documents/gitrepos/tke-budget/src/reprocess_pw/processed_data/5-13/sonic_end_based.npy"

    # Name map:
    name_map = {"u": "E", "v": "N", "w": "w", "p": "P2", "time": "dn"}
    sonic = Sonic.from_raw(files, name_map, fs=32, z=mabs, zarr_save_path="~/Desktop/adv_zarr_test")
