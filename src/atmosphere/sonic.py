import glob
import itertools
import numpy as np
import os
import scipy.signal as sig
import xarray as xr
from scipy.stats import median_abs_deviation
from sklearn.linear_model import LinearRegression
from typing import Optional, Union, List, Tuple
from src.utils.parsing_utils import DatasetParser
from src.utils.interp_utils import naninterp_pd
from src.utils.spectral_utils import psd, csd
from src.utils.base_instrument import BaseInstrument


class Sonic(BaseInstrument):

    @staticmethod
    def validate_inputs(
        files: Union[str, List],
        name_map: dict,
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[float, int, List[Union[float, int]]]] = None,
        zarr_save_path: Optional[str] = None,
        overwrite: Optional[bool] = False,
        L: Optional[float] = 0.15,
    ):

        # General validation
        BaseInstrument.validate_common_inputs(files, name_map, fs, z, zarr_save_path, overwrite)

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
        zarr_save_path: Optional[str] = None,
        overwrite: Optional[bool] = False,
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

        # General parser
        parser = DatasetParser(files, name_map, fs, z)
        ds = parser.parse_input()

        # Adding sonic-specific attribute
        ds.attrs["path_length"] = L

        if zarr_save_path and (overwrite or not os.path.exists(os.path.expanduser(zarr_save_path))):
            ds.to_zarr(zarr_save_path, consolidated=True)

        return cls(ds)


    def align_velocity_with_wind(self) -> (xr.DataArray, xr.DataArray, xr.DataArray):
        """
        Rotates u, v, w velocities to minimize the burst-averaged v and w.

        Parameters
        ----------


        Returns
        -------
        u_rot: DataArray
            Major axis horizontal velocity

        v_rot: DataArray
            Minor axis horizontal velocity

        w_rot: DataArray
            Zero-mean vertical velocity
        """

        u_bar = self.u.mean()
        v_bar = self.v.mean()
        w_bar = self.w.mean()
        s_bar = np.sqrt(u_bar**2 + v_bar**2)
        alpha = np.arctan2(v_bar, u_bar)
        beta = np.arctan2(w_bar, s_bar)
        u_rot = self.u * np.cos(alpha) * np.cos(beta) + self.v * np.sin(alpha) * np.cos(beta) + self.w * np.sin(beta)
        v_rot = -self.u * np.sin(alpha) + self.v * np.cos(alpha)
        w_rot = -self.u * np.cos(alpha) * np.sin(beta) - self.v * np.sin(alpha) * np.sin(beta) + self.w * np.cos(beta)

        return u_rot, v_rot, w_rot

    def dissipation(
        self, f_low: float, f_high: float, henjes_correction: bool, chunk_size: int = 100, **kwargs
    ) -> float:
        """

        Parameters
        ----------
        f_low
        f_high
        henjes_correction
        chunk_size
        kwargs

        Returns
        -------

        """

        def spectral_fit(u: np.ndarray, henjes_correction: bool = True) -> float:
            c1 = 0.53
            u_prime = sig.detrend(u, type="linear")
            u_bar = np.nanmean(u)
            f, S = psd(u_prime, onesided=False, **kwargs)

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

        ds_chunked = self.chunk({"burst": chunk_size})
        eps = xr.apply_ufunc(
            spectral_fit,
            ds_chunked.u,
            f_low,
            f_high,
            henjes_correction,
            input_core_dims=[["time"], [], [], []],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[self.w.dtype],
        )

        return eps


if __name__ == "__main__":
    import time

    t0 = time.time()
    # Testing this out
    file_path = "/Users/ea-gegan/Documents/gitrepos/tke-budget/src/reprocess_pw/processed_data/5-13/sonic_end_based.npy"

    # Name map:
    name_map = {"u": "E", "v": "N", "w": "w", "p": "P2", "time": "dn"}
    sonic = Sonic.from_raw(files, name_map, fs=32, z=mabs, zarr_save_path="~/Desktop/adv_zarr_test")
