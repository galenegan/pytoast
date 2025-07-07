import glob
import itertools
import numpy as np
import os
import scipy.signal as sig
import xarray as xr
from scipy.stats import median_abs_deviation
from sklearn.linear_model import LinearRegression
from typing import Optional, Union, List, Tuple
from utils.parsing_utils import DatasetParser
from utils.interp_utils import naninterp_pd
from utils.spectral_utils import psd, csd
from utils.base_instrument import BaseInstrument


class ADV(BaseInstrument):

    @staticmethod
    def validate_inputs(
        files: Union[str, List],
        name_map: dict,
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[float, int, List[Union[float, int]]]] = None,
        zarr_save_path: Optional[str] = None,
        overwrite: Optional[bool] = False,
    ):

        # General validation
        BaseInstrument.validate_common_inputs(files, name_map, fs, z, zarr_save_path, overwrite)

        # Instrument-specific requirements
        required_keys = ["u", "v", "w"]

        for key in required_keys:
            if key not in name_map:
                raise ValueError(f"`name_map` must include a mapping for '{key}'")

    @classmethod
    def from_raw(
        cls,
        files: Union[str, List],
        name_map: dict,
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[float, int, List[Union[float, int]]]] = None,
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

         z : float, int or List[float, int], optional
            mean height above the bed (m) for each instrument. If not specified, the height coordinate in the resulting
            ADV object will be integer indices.

        fs : int or float, optional
            sampling frequency (Hz). If not specified, will be inferred (and rounded to an integer)
            from name_map["time"] values


        Returns
        -------
        ADV object

        """
        ADV.validate_inputs(files, name_map, fs, z)
        parser = DatasetParser(files, name_map, fs, z)
        ds = parser.parse_input()
        if zarr_save_path and (overwrite or not os.path.exists(os.path.expanduser(zarr_save_path))):
            ds.to_zarr(zarr_save_path, consolidated=True)

        return cls(ds)

    def despike(self, threshold: int = 5, max_iter: int = 10, chunk_size: int = 100, robust_statistics: bool = False):
        """
        Implements the Goring & Nikora (2002) phase-space de-spiking algorithm,
        modifying self.u, self.v, and self.w in-place.

        Parameters
        ----------
        threshold : int
            Iterations will stop once there are threshold or fewer bad samples

        max_iter : int
            Maximum number of iterations

        chunk_size : int
            Chunk size for Dask parallelization

        robust_statistics : bool
            If True, ellipsoid centers will be based on the median and axis lengths will be based on median absolute
            deviation as suggested by Wahl (2003). If False, mean and standard deviation are used, consistent with the
            original Goring & Nikora implementation.

        Returns
        -------
        None
        """

        def flag_bad_indices(u: np.ndarray) -> np.ndarray:
            # Initializing gradient arrays
            du = np.gradient(u) / 2
            du2 = np.gradient(du) / 2

            if robust_statistics:
                # Standard deviation
                sigma_u = median_abs_deviation(u, nan_policy="omit")
                sigma_du = median_abs_deviation(du, nan_policy="omit")
                sigma_du2 = median_abs_deviation(du2, nan_policy="omit")

                # Median
                u_bar = np.nanmedian(u)
                du_bar = np.nanmedian(du)
                du2_bar = np.nanmedian(du2)
            else:
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

    def get_wavenumber(self, omega: Union[float, np.ndarray], h: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # Returns wavenumber from the surface gravity wave dispersion relation
        # using Newton's method

        g = 9.81
        k = omega / np.sqrt(g * h)

        f = g * k * np.tanh(k * h) - omega**2

        while np.max(np.abs(f)) > 1e-10:
            dfdk = g * k * h * ((1 / np.cosh(k * h)) ** 2) + g * np.tanh(k * h)
            k = k - f / dfdk
            f = g * k * np.tanh(k * h) - omega**2
        k[omega == 0] = 0
        return k

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

    def dissipation(self, f_low: float, f_high: float, chunk_size: int = 100, **kwargs) -> (float, float, int):
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

        def spectral_fit(u, v, w):
            """
            Carries out the spectral curve fit, applied through apply_ufunc
            """
            if np.sum(np.isnan(u)) == len(u) or np.sum(np.isnan(v)) == len(v) or np.sum(np.isnan(w)) == len(w):
                return np.nan, np.nan, 0
            omega_range = [2 * np.pi * f_low, 2 * np.pi * f_high]
            alpha = 1.5

            w_prime = sig.detrend(w, type="linear")
            fw, Pw = psd(w_prime, onesided=False, **kwargs)

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
            input_core_dims=[["time"], ["time"], ["time"]],
            output_core_dims=[[], [], []],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[self.w.dtype, self.w.dtype, int],
        )
        return eps, noise, quality_flag

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
    ) -> Tuple[float]:
        """
        Benilov wave-turbulence decomposition to estimate wave and turbulence
        components of the Reynolds stress. (Benilov & Filyushkin, 1970)

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
        kwargs: Additional arguments passed to spectral_utils.psd.
                See spectral_utils.psd for parameter definitions.

        Returns
        -------
        Tuple of turbulent and wave momentum flux components
        """

        u = sig.detrend(u)
        v = sig.detrend(v)
        w = sig.detrend(w)
        p = sig.detrend(p)
        g = 9.81

        h = 1e4 * np.nanmean(p) / (rho * g) + mab  # Average water depth

        # Getting sea surface elevation spectrum
        f, S_pp = psd(p, **kwargs)
        df = np.max(np.diff(f))
        omega = 2 * np.pi * f
        k = self.get_wavenumber(omega, h)
        attenuation_correction = 1e4 * np.cosh(k * h) / (rho * g * np.cosh(k * mab))
        S_etaeta = S_pp * (attenuation_correction**2)

        # All the velocity components
        _, S_uu = psd(u, **kwargs)
        _, S_up = csd(u, p, **kwargs)
        S_ueta = S_up * attenuation_correction

        _, S_vv = psd(v, **kwargs)
        _, S_vp = csd(v, p, **kwargs)
        S_veta = S_vp * attenuation_correction

        _, S_ww = psd(w, **kwargs)
        _, S_wp = csd(w, p, **kwargs)
        S_weta = S_wp * attenuation_correction

        # Velocity cross spectra
        _, S_uw = csd(u, w, **kwargs)
        _, S_vw = csd(v, w, **kwargs)
        _, S_uv = csd(u, v, **kwargs)

        # Defining frequency range
        if f_low:
            start_index = np.argmin(np.abs(f - f_low))
        else:
            start_index = 0

        if f_high:
            end_index = np.argmin(np.abs(f - f_high))
        else:
            end_index = len(f)

        # Calculating wave spectra
        S_uwave_uwave = S_ueta * np.conj(S_ueta) / S_etaeta
        S_vwave_vwave = S_veta * np.conj(S_veta / S_etaeta)
        S_wwave_wwave = S_weta * np.conj(S_weta / S_etaeta)
        S_uwave_wwave = S_ueta * np.conj(S_weta) / S_etaeta
        S_uwave_vwave = S_ueta * np.conj(S_veta) / S_etaeta
        S_vwave_wwave = S_veta * np.conj(S_weta) / S_etaeta

        # Calculating turbulent spectra
        S_ut_ut = S_uu - S_uwave_uwave
        S_ut_wt = S_uw - S_uwave_wwave
        S_ut_vt = S_uv - S_uwave_vwave
        S_vt_vt = S_vv - S_vwave_vwave
        S_vt_wt = S_vw - S_vwave_wwave
        S_wt_wt = S_ww - S_wwave_wwave

        # Summing them to get Reynolds stresses
        uu_turb = np.nansum(np.real(S_ut_ut[start_index:end_index]) * df)
        uu_wave = np.nansum(np.real(S_uwave_uwave[start_index:end_index]) * df)
        vv_turb = np.nansum(np.real(S_vt_vt[start_index:end_index]) * df)
        vv_wave = np.nansum(np.real(S_vwave_vwave[start_index:end_index]) * df)
        ww_turb = np.nansum(np.real(S_wt_wt[start_index:end_index]) * df)
        ww_wave = np.nansum(np.real(S_wwave_wwave[start_index:end_index]) * df)
        uw_turb = np.nansum(np.real(S_ut_wt[start_index:end_index]) * df)
        uw_wave = np.nansum(np.real(S_uwave_wwave[start_index:end_index]) * df)
        vw_turb = np.nansum(np.real(S_vt_wt[start_index:end_index]) * df)
        vw_wave = np.nansum(np.real(S_vwave_wwave[start_index:end_index]) * df)
        uv_turb = np.nansum(np.real(S_ut_vt[start_index:end_index]) * df)
        uv_wave = np.nansum(np.real(S_uwave_vwave[start_index:end_index]) * df)

        # Tuple output which will get restructured in the call to ADV.covariances
        out = (
            uu_turb,
            uu_wave,
            vv_turb,
            vv_wave,
            ww_turb,
            ww_wave,
            uw_turb,
            uw_wave,
            vw_turb,
            vw_wave,
            uv_turb,
            uv_wave,
        )
        return out

    def phase_decomposition(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        f_low: Optional[float] = None,
        f_high: Optional[float] = None,
        **kwargs,
    ):

        u = sig.detrend(u)
        v = sig.detrend(v)
        w = sig.detrend(w)

        # All the velocity components
        _, S_uu = psd(u, **kwargs)
        _, S_vv = psd(v, **kwargs)
        _, S_ww = psd(w, **kwargs)

        # Velocity cross spectra
        _, S_uw = csd(u, w, **kwargs)
        _, S_vw = csd(v, w, **kwargs)
        f, S_uv = csd(u, v, **kwargs)

        phase_uw = np.arctan2(np.imag(S_uw), np.real(S_uw))
        phase_vw = np.arctan2(np.imag(S_vw), np.real(S_vw))
        phase_uv = np.arctan2(np.imag(S_uv), np.real(S_uv))
        df = np.nanmax(np.diff(f))

        # Searching for the wave peak within a reasonable range of frequencies --
        # adjust this for each data set
        if f_low and f_high:
            waverange = np.where(((f > f_low) & (f < f_high)))[0]
        else:
            f_offset = 0.07  # Assume wave peak is not below this value
            width_ratio_low = 0.35  # Wave range below the wave peak (fraction of peak frequency)
            width_ratio_high = 0.8  # Wave range above the peak
            offset = np.sum(f <= f_offset)
            u_idx_max = np.argmax(S_uu[(f > f_offset) & (f < 1)]) + offset
            f_max = f[u_idx_max]
            waverange = np.arange(
                max(u_idx_max - (f_max * width_ratio_low) // df, 0),
                min(u_idx_max + (f_max * width_ratio_high) // df, len(f) - 1),
            ).astype(int)
        interprange = np.arange(1, np.nanargmin(np.abs(f - 1))).astype(int)

        # Separating the turbulent portion from the full spectrum
        Suu_turb = S_uu[interprange]
        fuu = f[interprange]
        Suu_turb = np.delete(Suu_turb, waverange - interprange[0])
        fuu = np.delete(fuu, waverange - interprange[0])
        Suu_turb = Suu_turb[fuu > 0]
        fuu = fuu[fuu > 0]

        Svv_turb = S_vv[interprange]
        fvv = f[interprange]
        Svv_turb = np.delete(Svv_turb, waverange - interprange[0])
        fvv = np.delete(fvv, waverange - interprange[0])
        Svv_turb = Svv_turb[fvv > 0]
        fvv = fvv[fvv > 0]

        Sww_turb = S_ww[interprange]
        fww = f[interprange]
        Sww_turb = np.delete(Sww_turb, waverange - interprange[0])
        fww = np.delete(fww, waverange - interprange[0])
        Sww_turb = Sww_turb[fww > 0]
        fww = fww[fww > 0]

        # Linear interpolation over turbulent spectra
        F = np.log(fuu)
        S = np.log(Suu_turb)
        Puu = np.polyfit(F, S, deg=1)
        Puuhat = np.exp(np.polyval(Puu, np.log(f)))

        F = np.log(fvv)
        S = np.log(Svv_turb)
        Pvv = np.polyfit(F, S, deg=1)
        Pvvhat = np.exp(np.polyval(Pvv, np.log(f)))

        F = np.log(fww)
        S = np.log(Sww_turb)
        Pww = np.polyfit(F, S, deg=1)
        Pwwhat = np.exp(np.polyval(Pww, np.log(f)))

        # # Plotting to test the code
        # if plot:
        #     plt.figure()
        #     plt.loglog(fuu, Suu_turb, "k*")
        #     plt.loglog(f[waverange], Suu[waverange], "r-")
        #     plt.loglog(f, Puuhat, "b-")
        #     plt.title("Suu")
        #     plt.show()
        #
        #     plt.figure()
        #     plt.loglog(fww, Sww_turb, "k*")
        #     plt.loglog(f[waverange], Sww[waverange], "r-")
        #     plt.loglog(f, Pwwhat, "b-")
        #     plt.title("Sww")
        #     plt.show()

        # Wave spectra strictly above the interpolation line
        Suu_wave = S_uu[waverange] - Puuhat[waverange]
        Suu_wave[Suu_wave < 0] = 0
        Svv_wave = S_vv[waverange] - Pvvhat[waverange]
        Svv_wave[Svv_wave < 0] = 0
        Sww_wave = S_ww[waverange] - Pwwhat[waverange]
        Sww_wave[Sww_wave < 0] = 0

        # Wave Fourier components
        Amu_wave = np.sqrt((Suu_wave + 0j) * df)
        Amv_wave = np.sqrt((Svv_wave + 0j) * df)
        Amw_wave = np.sqrt((Sww_wave + 0j) * df)

        # Wave Magnitudes
        Um_wave = np.sqrt(np.real(Amu_wave) ** 2 + np.imag(Amu_wave) ** 2)
        Vm_wave = np.sqrt(np.real(Amv_wave) ** 2 + np.imag(Amv_wave) ** 2)
        wm_wave = np.sqrt(np.real(Amw_wave) ** 2 + np.imag(Amw_wave) ** 2)

        # Wave reynolds stress
        uw_wave = np.nansum(Um_wave * wm_wave * np.cos(phase_uw[waverange]))
        uv_wave = np.nansum(Um_wave * Vm_wave * np.cos(phase_uv[waverange]))
        vw_wave = np.nansum(Vm_wave * wm_wave * np.cos(phase_vw[waverange]))

        uu_wave = np.nansum(Suu_wave * df)
        vv_wave = np.nansum(Svv_wave * df)
        ww_wave = np.nansum(Sww_wave * df)

        # Full reynolds stresses
        uu = np.nansum(np.real(S_uu) * df)
        uv = np.nansum(np.real(S_uv) * df)
        uw = np.nansum(np.real(S_uw) * df)
        vv = np.nansum(np.real(S_vv) * df)
        vw = np.nansum(np.real(S_vw) * df)
        ww = np.nansum(np.real(S_ww) * df)

        # Turbulent reynolds stresses

        uu_turb = uu - uu_wave
        vv_turb = vv - vv_wave
        ww_turb = ww - ww_wave
        uw_turb = uw - uw_wave
        uv_turb = uv - uv_wave
        vw_turb = vw - vw_wave

        # Tuple output which will get restructured in the call to ADV.covariances
        out = (
            uu_turb,
            uu_wave,
            vv_turb,
            vv_wave,
            ww_turb,
            ww_wave,
            uw_turb,
            uw_wave,
            vw_turb,
            vw_wave,
            uv_turb,
            uv_wave,
        )
        return out

    def covariance(
        self,
        method: str = "cov",
        f_low: Optional[float] = None,
        f_high: Optional[float] = None,
        rho: Optional[float] = 1020,
        chunk_size: Optional[int] = 100,
        **kwargs,
    ) -> dict:
        """
        Calculate components of the covariance matrix.

        Parameters
        ----------
        method : str
            Method to calculate covariances. Options are:
            - 'cov': Standard covariance calculation
            - 'benilov': Benilov wave-turbulence decomposition
            - 'phase':  Bricker & Monismith phase-method wave-turbulence decomposition
        f_low : float, optional
            Lower frequency bound (Hz) for wave band, by default None
        f_high : float, optional 
            Upper frequency bound (Hz) for wave band, by default None
        rho : float, optional
            Water density (kg/m^3), by default 1020
        chunk_size : int, optional
            Size of chunks for parallel processing, by default 100
        **kwargs
            Additional arguments passed to spectral calculations

        Returns
        -------
        dict
            Dictionary containing covariance components. For method='cov', keys are
            velocity component pairs (e.g. 'uu','uv','uw'). For wave decomposition
            methods, keys include turbulent and wave components (e.g. 'uu_turb',
            'uu_wave').
        """
        if method == "cov":
            out = {}
            components_to_return = [elem for elem in itertools.combinations_with_replacement(("u", "v", "w"), 2)]
            for component_pair in components_to_return:
                key = component_pair[0] + component_pair[1]
                out[key] = xr.cov(self[component_pair[0]], self[component_pair[1]], dim="time")
            return out
        elif method == "benilov":
            ds_chunked = self.chunk({"burst": chunk_size})
            out = xr.apply_ufunc(
                self.benilov_decomposition,
                ds_chunked.u,
                ds_chunked.v,
                ds_chunked.w,
                ds_chunked.p,
                ds_chunked.height,
                rho,
                f_low,
                f_high,
                kwargs=kwargs,
                input_core_dims=[["time"], ["time"], ["time"], ["time"], [], [], [], []],
                output_core_dims=[[], [], [], [], [], [], [], [], [], [], [], []],
                output_dtypes=[float] * 12,
                vectorize=True,
                dask="parallelized",
            )
            # Turning into a dict to match format of "cov" method
            keys = [
                "uu_turb",
                "uu_wave",
                "vv_turb",
                "vv_wave",
                "ww_turb",
                "ww_wave",
                "uw_turb",
                "uw_wave",
                "vw_turb",
                "vw_wave",
                "uv_turb",
                "uv_wave",
            ]

            out_dict = {key: da for key, da in zip(keys, out)}
            return out_dict
        elif method == "phase":
            ds_chunked = self.chunk({"burst": chunk_size})
            out = xr.apply_ufunc(
                self.phase_decomposition,
                ds_chunked.u,
                ds_chunked.v,
                ds_chunked.w,
                f_low,
                f_high,
                kwargs=kwargs,
                input_core_dims=[["time"], ["time"], ["time"], [], []],
                output_core_dims=[[], [], [], [], [], [], [], [], [], [], [], []],
                output_dtypes=[float] * 12,
                vectorize=True,
                dask="parallelized",
            )
            # Turning into a dict to match format of "cov" method
            keys = [
                "uu_turb",
                "uu_wave",
                "vv_turb",
                "vv_wave",
                "ww_turb",
                "ww_wave",
                "uw_turb",
                "uw_wave",
                "vw_turb",
                "vw_wave",
                "uv_turb",
                "uv_wave",
            ]

            out_dict = {key: da for key, da in zip(keys, out)}
            return out_dict
        else:
            raise IOError(f"Unrecognized method {method}")


if __name__ == "__main__":
    import time

    t0 = time.time()
    mean_depth = 13
    m_below_surface = np.linspace(1.8, 7.2, 6)
    mabs = [mean_depth - mbs for mbs in m_below_surface]

    # Testing this out
    files = glob.glob("/Users/ea-gegan/Documents/gitrepos/tke-budget/data/adv_fall/*.mat")
    files.sort()
    files = files[:50]

    # Name map:
    name_map = {"u": "E", "v": "N", "w": "w", "p": "P2", "time": "dn"}
    adv = ADV.from_raw(files, name_map, fs=32, z=mabs, zarr_save_path="~/Desktop/adv_zarr_test")
    adv.despike()
    theta = adv.get_principal_axis()
    vel_maj, vel_min = adv.rotate_velocity(theta)
    adv.u, adv.v = vel_maj, vel_min

    # eps, noise, quality_flag = adv.dissipation(f_low=1.2, f_high=15, fs=32)
    # print(eps.values[:, 0])
    cov = adv.covariance(
        method="benilov",
        fs=32,
    )
    cov0 = cov["uw_wave"][:, 4].values
    cov = adv.covariance(method="phase", fs=32)
    cov1 = cov["uw_wave"][:, 4].values

    import matplotlib.pyplot as plt

    one = np.linspace(np.nanmin(cov1), np.nanmax(cov1), 100)
    plt.plot(cov1, cov0, "o")
    plt.plot(one, one, "-")
    plt.show()
    # print(cov.values)
    # t1 = time.time()
    # print(f"finished processing 20 files in {t1 - t0:.2f} seconds")
    # adv0 = ADV.from_saved_zarr("~/Desktop/adv_zarr_test")
    # test2_0 = adv0.u[9, 0, :].values

    # import matplotlib.pyplot as plt
    #
    # plt.plot(vel_maj[9, 0, :].values)
    # plt.plot(adv.u[9, 0, :].values)
    # plt.show()
