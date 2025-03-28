import glob
import numpy as np
import os
import scipy.signal as sig
import xarray as xr
from sklearn.linear_model import LinearRegression
from typing import Optional, Union, List
from utils.parsing_utils import DatasetParser
from utils.interp_utils import naninterp_pd
from utils.spectral_utils import psd, csd


class ADV:
    def __init__(self, dataset: xr.Dataset):
        super().__setattr__("ds", dataset)

    def __getattr__(self, name):
        if name in self.ds.variables:
            return self.ds[name]
        return getattr(self.ds, name)

    def __setattr__(self, name, value):
        if "ds" in self.__dict__ and name in self.ds.variables:
            # Assign to dataset variable if it already exists
            self.ds[name] = value
        else:
            # Otherwise, assign normally (e.g., during __init__)
            super().__setattr__(name, value)

    @staticmethod
    def validate_inputs(files, name_map, fs, z):
        # TODO: validate all inputs to from_raw
        pass

    @classmethod
    def from_raw(
        cls,
        files: Union[str, List],
        name_map: dict,
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[float, List[float]]] = None,
        zarr_save_path: Optional[str] = None,
        overwrite: Optional[bool] = False,
    ):
        """
        Initializes a new ADV object from data files.

        Parameters
        ----------
        files : str or List[str]
            If str, must be a path to a netCDF file, .mat file, or zarr file store that contains the
            entire dataset you wish to load. If list, the elements of the list will be interpreted as files containing data
            from individual measurement burst periods. Supported burst file types are .npy (assuming it was saved as a
            dictionary), .mat (assuming it was saved as a Matlab Struct) and .csv (with variables in separate columns).
            If the variables associated with a particular name are two-dimensional, then the larger dimension is
            assumed to be time and the shorter dimension is assumed to be a vertical coordinate. In this case,
            a "z" list must be passed as an argument with a length matching the size of the shorter dimension

        name_map : dict
            a dictionary of the form:
            {
                "u": "x-velocity variable name" or ["var 1", "var 2", etc.],
                "v": "y-velocity variable name" or ["var 1", "var 2", etc.],
                "w": "z-velocity variable name" or ["var 1", "var 2", etc.],
                "p": "pressure variable name" or ["var 1", "var 2", etc.],
                "time": "time variable name" or ["var 1", "var 2", etc.],
            }
            Of these, "p" and "time" are optional, but an error will be raised if "time" is not specified and a sampling
            frequency is also not specified. Lists should be provided if data from multiple instruments is stored
            in multiple variables (as opposed to a 2d array).

         z : float or List[float]
            mean height above the bed (m) for each instrument

        fs : int or float
            sampling frequency (Hz), will be inferred (and rounded to an integer) from name_map["time"] values
            if not specified

        """
        ADV.validate_inputs(files, name_map, fs, z)
        parser = DatasetParser(files, name_map, fs, z)
        ds = parser.parse_input()
        if zarr_save_path and (overwrite or not os.path.exists(os.path.expanduser(zarr_save_path))):
            ds.to_zarr(zarr_save_path, consolidated=True)

        return cls(ds)

    @classmethod
    def from_saved_zarr(cls, zarr_path: str):
        ds = xr.open_zarr(zarr_path)
        return cls(ds)

    def despike(self, threshold: int = 5, max_iter: int = 10, chunk_size: int = 100):
        """
        Implements the Goring & Nikora (2002) phase-space de-spiking algorithm, modifying self.u, self.v, and self.w
        in-place.

        Parameters
        ----------
        threshold : int
            Iterations will stop once there are threshold or fewer bad samples

        max_iter : int
            Maximum number of iterations

        chunk_size : int
            Chunk size for Dask parallelization

        Returns:
            None
        """

        def flag_bad_indices(u: np.ndarray) -> np.ndarray:
            # Initializing gradient arrays
            du = np.gradient(u) / 2
            du2 = np.gradient(du) / 2

            # Standard deviation
            sigma_u = np.nanstd(u)
            sigma_du = np.nanstd(du)
            sigma_du2 = np.nanstd(du2)

            # Mean
            u_bar = np.nanmean(u)
            du_bar = np.nanmean(du)
            du2_bar = np.nanmean(du2)

            # Expected absolute maximum
            n = len(u)
            lam = np.sqrt(2 * np.log(n))

            # Calculating axes of the 3 ellipses
            theta = np.arctan(np.nansum(u * du2) / np.nansum(u**2))

            # u vs du
            a1 = lam * sigma_u
            b1 = lam * sigma_du

            # du vs du2
            a3 = lam * sigma_du
            b3 = lam * sigma_du2

            # u vs du2
            A = np.array([[np.cos(theta) ** 2, np.sin(theta) ** 2], [np.sin(theta) ** 2, np.cos(theta) ** 2]])
            b = np.array([(lam * sigma_u) ** 2, (lam * sigma_du2) ** 2])
            x = np.linalg.solve(A, b)
            a2 = np.sqrt(x[0])
            b2 = np.sqrt(x[1])

            # Finding the indices of the elements outside the three ellipses

            # u vs du
            bad_index_u_du = (u - u_bar) ** 2 / a1**2 + (du - du_bar) ** 2 / b1**2 > 1

            # u vs du2
            bad_index_u_du2 = (
                (np.cos(theta) * (u - u_bar) + np.sin(theta) * (du2 - du2_bar)) ** 2 / a2**2
                + (np.sin(theta) * (u - u_bar) - np.cos(theta) * (du2 - du2_bar)) ** 2 / b2**2
            ) > 1

            # du vs du2
            bad_index_du_du2 = (du - du_bar) ** 2 / a3**2 + (du2 - du2_bar) ** 2 / b3**2 > 1

            # Combining all of them
            bad_index_total = bad_index_u_du | bad_index_u_du2 | bad_index_du_du2

            return bad_index_total

        def despike_worker(u: np.ndarray) -> np.ndarray:
            u_out = u.copy()
            iterations = 0
            # First pass
            bad_index_u = flag_bad_indices(u_out)
            total_bad_u = sum(bad_index_u)

            while ((total_bad_u > threshold)) and (iterations < max_iter):
                u_out[bad_index_u] = np.nan
                u_out = naninterp_pd(u_out)

                if total_bad_u > threshold:
                    bad_index_u = flag_bad_indices(u_out)

                total_bad_u = sum(bad_index_u)
                iterations += 1

            u_out = naninterp_pd(u_out)
            return u_out

        ds_chunked = self.chunk({"burst": chunk_size})

        self.u = xr.apply_ufunc(
            despike_worker,
            ds_chunked.u,
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[self.u.dtype],
        )
        self.v = xr.apply_ufunc(
            despike_worker,
            ds_chunked.v,
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[self.v.dtype],
        )
        self.w = xr.apply_ufunc(
            despike_worker,
            ds_chunked.w,
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[self.w.dtype],
        )

    def get_principal_axis(self) -> xr.DataArray:
        """
        Calculates the direction of maximum variance from the u and v velocities (Thomson & Emery, 4.52b).

        Parameters
        ----------

        Returns
        -------
        theta : DataArray
            (bursts, height) array containing the direction of maximum variance in degrees, CCW positive from east
            assuming that u = eastward velocity, v = northward velocity
        """
        # Covariance matrix
        u_var = self.u.var(dim="time")
        v_var = self.v.var(dim="time")
        cv = xr.cov(self.u, self.v, dim="time")

        # Direction of maximum variance
        theta = (180.0 / np.pi) * (0.5 * np.arctan2(2.0 * cv, (u_var - v_var)))

        return theta

    def rotate_velocity(self, theta: xr.DataArray) -> (xr.DataArray, xr.DataArray):
        """
        Rotates u, v velocities in direction theta

        Parameters
        ----------
        theta: DataArray
            Direction (degrees, CCW positive from east) in which velocites should be rotated.
            This is often the output of adv.get_principal_axis

        Returns
        -------
        vel_maj: DataArray
            Major axis velocity (m/s) in the direction of the first principal component

        vel_min: DataArray
            Minor axis velocity (m/s) in the direction of the second principal component
        """
        # Storing as complex variable
        U = self.u + 1j * self.v
        U_rotated = U * np.exp(-1j * theta * np.pi / 180)
        vel_maj = np.real(U_rotated)
        vel_min = np.imag(U_rotated)

        return vel_maj, vel_min

    def dissipation(
        self,
        f_low: float,
        f_high: float,
        chunk_size: int = 100,
        **kwargs
    ) -> (float, float, int):
        """
        Estimate the dissipation rate of TKE using the Gerbi et al. (2009)
        spectral curve fitting method. This is nearly equivalent to the
        Feddersen et al. (2007) method, but it uses a more efficient numerical
        integration and estimates dissipation with a least squares fit rather
        than a mean over the inertial range.

        Parameters
        ----------
        f_low : float
            Lower frequency bound (Hz) for inertial subrange where -5/3 law applies

        f_high : float
            Upper frequency bound (Hz) for inertial subrange where -5/3 law applies

        chunk_size : int
            Chunk size (in burst dimension) for Dask parallelization

        **kwargs
            Additional arguments passed to spectral_utils.psd.
            See spectral_utils.psd for parameter definitions.

        Returns
        -------
        eps : float
            dissipation rate of TKE (m^2/s^3)
        noise: float
            intercept from dissipation linear regression
        quality_flag: int
            1 for good eps estimate, 0 for bad eps estimate.
            Defined based on Gerbi Eq. X
        """

        def calcJ33(sig1, sig2, sig3, u1, u2):
            """
            Calculates J33, the output of equation A.13
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
            G_squared = (sin_theta**2)[:, np.newaxis] * (cos_phi**2 / sig1**2 + sin_phi**2 / sig2**2)[
                np.newaxis, :
            ] + (cos_theta**2)[:, np.newaxis] / sig3**2

            # Also shape (Ntheta, Nphi)
            P33 = ((sin_theta**2)[:, np.newaxis] / G_squared) * (cos_phi**2 / sig1**2 + sin_phi**2 / sig2**2)[
                np.newaxis, :
            ]
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
            I2 = -np.trapezoid(G_squared_3 ** (-11 / 6) * sin_theta[:, np.newaxis, np.newaxis] * P33_3 * I3, R, axis=2)

            # Outer integral
            I1 = np.trapezoid(I2, phi, axis=-1)

            J33 = (1 / (2 * (2 * np.pi) ** (3 / 2))) * (1 / (sig1 * sig2 * sig3)) * np.trapezoid(I1, theta, axis=-1)

            return J33

        def spectral_fit(u, v, w, fs):
            """
            Carries out the spectral curve fit, applied through apply_ufunc
            """
            if np.sum(np.isnan(u)) == len(u) or np.sum(np.isnan(v)) == len(v) or np.sum(np.isnan(w)) == len(w):
                return np.nan, np.nan, 0
            omega_range = [2 * np.pi * f_low, 2 * np.pi * f_high]
            alpha = 1.5

            w_prime = sig.detrend(w, type="linear")
            fw, Pw = psd(
                w_prime,
                fs,
                onesided=False,
                **kwargs
            )

            omega = 2 * np.pi * fw

            inertial_indices = (omega >= omega_range[0]) & (omega <= omega_range[1])
            omega_inertial = omega[inertial_indices]
            Pw_inertial = (Pw[inertial_indices]) / (2 * np.pi)

            sig1 = np.nanstd(u)
            sig2 = np.nanstd(v)
            sig3 = np.nanstd(w)

            u1 = np.nanmean(u)
            u2 = np.nanmean(v)

            J33 = calcJ33(sig1, sig2, sig3, u1, u2)

            # linear regression
            X = J33 * alpha * (omega_inertial ** (-5 / 3))
            y = Pw_inertial
            reg = LinearRegression().fit(X.reshape(-1, 1), y)
            eps = reg.coef_[0] ** (3 / 2)
            noise = reg.intercept_

            if noise < J33 * alpha * (eps ** (2 / 3)) * (omega_range[0] ** (-5 / 3)):
                quality_flag = 1
            else:
                quality_flag = 0

            return eps, noise, quality_flag

        ds_chunked = self.chunk({"burst": chunk_size})

        (eps, noise, quality_flag) = xr.apply_ufunc(
            spectral_fit,
            ds_chunked.u,
            ds_chunked.v,
            ds_chunked.w,
            ds_chunked.fs,
            input_core_dims=[["time"], ["time"], ["time"], []],
            output_core_dims=[[], [], []],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[self.w.dtype, self.w.dtype, int],
        )
        return eps, noise, quality_flag


if __name__ == "__main__":
    import time

    t0 = time.time()
    mean_depth = 13
    m_below_surface = np.linspace(1.8, 7.2, 6)
    mabs = [mean_depth - mbs for mbs in m_below_surface]

    # Testing this out
    files = glob.glob("/Users/ea-gegan/Documents/gitrepos/tke-budget/data/adv_fall/*.mat")
    files.sort()
    files = files[:20]

    # Name map:
    name_map = {"u": "E", "v": "N", "w": "w", "p": "P2", "time": "dn"}
    adv = ADV.from_raw(files, name_map, fs=32, z=mabs, zarr_save_path="~/Desktop/adv_zarr_test")
    adv.despike()
    theta = adv.get_principal_axis()
    vel_maj, vel_min = adv.rotate_velocity(theta)
    adv.u, adv.v = vel_maj, vel_min

    eps, noise, quality_flag = adv.dissipation(f_low=1.2, f_high=15)
    print(eps.values[:, 0])
    t1 = time.time()
    print(f"finished processing 20 files in {t1 - t0:.2f} seconds")
    # adv0 = ADV.from_saved_zarr("~/Desktop/adv_zarr_test")
    # test2_0 = adv0.u[9, 0, :].values

    # import matplotlib.pyplot as plt
    #
    # plt.plot(vel_maj[9, 0, :].values)
    # plt.plot(adv.u[9, 0, :].values)
    # plt.show()
