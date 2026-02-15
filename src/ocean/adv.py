import numpy as np
import scipy.signal as sig
from typing import Optional, Union, List, Dict, Any, Tuple
from sklearn.linear_model import LinearRegression
from utils.base_instrument import BaseInstrument
from utils.interp_utils import naninterp_pd
from utils.wave_utils import get_wavenumber, get_cg, jones_monismith_correction
from scipy.stats import median_abs_deviation

from utils.spectral_utils import psd, csd, get_frequency_range
from utils.constants import GRAVITATIONAL_ACCELERATION as g


class ADV(BaseInstrument):
    """
    Refactored ADV class with numpy-based processing and on-demand data loading.
    No longer stores raw data in xarray, instead uses efficient numpy processing.
    """

    @staticmethod
    def validate_inputs(
        files: Union[str, List],
        name_map: dict,
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[float, int, List[Union[float, int]]]] = None,
    ):

        # General validation
        BaseInstrument.validate_common_inputs(files, name_map, fs, z)

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
        return cls(files, name_map, fs, z)

    def set_preprocess_opts(self, opts: Dict[str, Any]):
        """Enable preprocessing for all subsequent burst loads using the options defined in the input dictionary.

        Parameters
        ----------
        opts : Dict[str, Any]
            Options for preprocessing. Currently supports the following keys/values
            {
                "despike": bool
                "despike_opts": {threshold: int, max_iter: int, robust_statistics: bool}
                "rotate": "align_principal", "align_current", or (horizontal_angle(s), vertical_angle(s))
        """

        self._preprocess_enabled = True

        self._despike = opts.get("despike", True)
        self._despike_opts = opts.get("despike_opts", {})
        self._rotate = opts.get("rotate", "align_principal")
        self._cached_idx = None
        self._cached_data = None

    def _apply_preprocessing(self, burst_data):
        if self._despike:
            burst_data = self._apply_despike(burst_data, **self._despike_opts)

        if isinstance(self._rotate, str):
            if self._rotate == "align_principal":
                theta_h, theta_v = ADV._align_with_principal_axis(burst_data)
            elif self._rotate == "align_current":
                theta_h, theta_v = ADV._align_with_current(burst_data)
            else:
                raise ValueError(f"Invalid rotation option '{self._rotate}'")
        else:
            theta_h, theta_v = self._rotate

        if np.sum(np.abs(theta_v)) != 0.0 or np.sum(np.abs(theta_h)) != 0.0:
            burst_data = self._rotate_velocity(burst_data, theta_h, theta_v)

        return burst_data

    def _apply_despike(self, data, threshold: int = 5, max_iter: int = 10, robust_statistics: bool = False):
        """
        Implements the Goring & Nikora (2002) phase-space de-spiking algorithm,
        returning data with modified data["u"], data["v"], and data["w"].

        Parameters
        ----------
        threshold : int
            Iterations will stop once there are threshold or fewer bad samples

        max_iter : int
            Maximum number of iterations

        robust_statistics : bool
            If True, ellipsoid centers will be based on the median and axis lengths will be based on median absolute
            deviation as suggested by Wahl (2003). If False, mean and standard deviation are used, consistent with the
            original Goring & Nikora implementation.

        Returns
        -------
        data : dict
            Original data dictionary with "u", "v", and "w" velocity arrays despiked

        """

        def flag_bad_indices(u: np.ndarray) -> np.ndarray:
            """Flag spikes in a 2D array (n_heights, n_samples) using phase-space method."""
            # Gradients along time axis
            du = np.gradient(u, axis=1) / 2
            du2 = np.gradient(du, axis=1) / 2

            # Per-row statistics → (n_heights,)
            if robust_statistics:
                sigma_u = median_abs_deviation(u, axis=1, nan_policy="omit")
                sigma_du = median_abs_deviation(du, axis=1, nan_policy="omit")
                sigma_du2 = median_abs_deviation(du2, axis=1, nan_policy="omit")
                u_bar = np.nanmedian(u, axis=1)
                du_bar = np.nanmedian(du, axis=1)
                du2_bar = np.nanmedian(du2, axis=1)
            else:
                sigma_u = np.nanstd(u, axis=1)
                sigma_du = np.nanstd(du, axis=1)
                sigma_du2 = np.nanstd(du2, axis=1)
                u_bar = np.nanmean(u, axis=1)
                du_bar = np.nanmean(du, axis=1)
                du2_bar = np.nanmean(du2, axis=1)

            # Expected absolute maximum
            n = u.shape[1]
            lam = np.sqrt(2 * np.log(n))

            # Rotation angle per row → (n_heights,)
            theta = np.arctan(np.nansum(u * du2, axis=1) / np.nansum(u**2, axis=1))

            # Ellipse axes (unrotated) → (n_heights,)
            a1 = lam * sigma_u
            b1 = lam * sigma_du
            a3 = lam * sigma_du
            b3 = lam * sigma_du2

            # Rotated ellipse axes via batched 2x2 solve
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            A = np.empty((u.shape[0], 2, 2))
            A[:, 0, 0] = cos_t**2
            A[:, 0, 1] = sin_t**2
            A[:, 1, 0] = sin_t**2
            A[:, 1, 1] = cos_t**2
            b_vec = np.stack([(lam * sigma_u) ** 2, (lam * sigma_du2) ** 2], axis=1)  # (n_heights, 2)
            x = np.linalg.solve(A, b_vec[:, :, None]).squeeze(-1)  # (n_heights, 2)
            a2 = np.sqrt(x[:, 0])
            b2 = np.sqrt(x[:, 1])

            # Broadcast all (n_heights,) stats to (n_heights, 1) for element-wise tests
            u_bar = u_bar[:, np.newaxis]
            du_bar = du_bar[:, np.newaxis]
            du2_bar = du2_bar[:, np.newaxis]
            a1 = a1[:, np.newaxis]
            b1 = b1[:, np.newaxis]
            a2 = a2[:, np.newaxis]
            b2 = b2[:, np.newaxis]
            a3 = a3[:, np.newaxis]
            b3 = b3[:, np.newaxis]
            cos_t = cos_t[:, np.newaxis]
            sin_t = sin_t[:, np.newaxis]

            # u vs du
            bad_u_du = (u - u_bar) ** 2 / a1**2 + (du - du_bar) ** 2 / b1**2 > 1

            # u vs du2 (rotated ellipse)
            bad_u_du2 = (
                (cos_t * (u - u_bar) + sin_t * (du2 - du2_bar)) ** 2 / a2**2
                + (sin_t * (u - u_bar) - cos_t * (du2 - du2_bar)) ** 2 / b2**2
            ) > 1

            # du vs du2
            bad_du_du2 = (du - du_bar) ** 2 / a3**2 + (du2 - du2_bar) ** 2 / b3**2 > 1

            return bad_u_du | bad_u_du2 | bad_du_du2

        def interp_rows(u: np.ndarray) -> np.ndarray:
            """Apply naninterp_pd independently to each row."""
            for i in range(u.shape[0]):
                u[i] = naninterp_pd(u[i])
            return u

        def despike_worker(u: np.ndarray) -> np.ndarray:
            u_out = u.copy()
            bad_index = flag_bad_indices(u_out)
            total_bad = np.sum(bad_index, axis=1)
            iterations = 0

            while np.any(total_bad > threshold) and iterations < max_iter:
                u_out[bad_index] = np.nan
                interp_rows(u_out)
                bad_index = flag_bad_indices(u_out)
                total_bad = np.sum(bad_index, axis=1)
                iterations += 1

            interp_rows(u_out)
            return u_out

        data["u"] = despike_worker(data["u"])
        data["v"] = despike_worker(data["v"])
        data["w"] = despike_worker(data["w"])
        return data

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

        h = 1e4 * np.nanmean(p) / (rho * g) + mab  # Average water depth

        # Getting sea surface elevation spectrum
        f, S_pp = psd(p, self.fs, **kwargs)
        df = np.max(np.diff(f))
        omega = 2 * np.pi * f
        k = get_wavenumber(omega, h)
        attenuation_correction = 1e4 * np.cosh(k * h) / (rho * g * np.cosh(k * mab))
        S_etaeta = S_pp * (attenuation_correction**2)

        # All the velocity components
        _, S_uu = psd(u, self.fs, **kwargs)
        _, S_up = csd(u, p, self.fs, **kwargs)
        S_ueta = S_up * attenuation_correction

        _, S_vv = psd(v, self.fs, **kwargs)
        _, S_vp = csd(v, p, self.fs, **kwargs)
        S_veta = S_vp * attenuation_correction

        _, S_ww = psd(w, self.fs, **kwargs)
        _, S_wp = csd(w, p, self.fs, **kwargs)
        S_weta = S_wp * attenuation_correction

        # Velocity cross spectra
        _, S_uw = csd(u, w, self.fs, **kwargs)
        _, S_vw = csd(v, w, self.fs, **kwargs)
        _, S_uv = csd(u, v, self.fs, **kwargs)

        # Defining frequency range
        start_index, end_index = get_frequency_range(f, f_low, f_high)

        # Calculating wave spectra
        S_uwave_uwave = S_ueta * np.conj(S_ueta) / S_etaeta
        S_vwave_vwave = S_veta * np.conj(S_veta) / S_etaeta
        S_wwave_wwave = S_weta * np.conj(S_weta) / S_etaeta
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
        out = {}
        out["uu_turb"] = np.nansum(np.real(S_ut_ut[start_index:end_index]) * df)
        out["uu_wave"] = np.nansum(np.real(S_uwave_uwave[start_index:end_index]) * df)
        out["vv_turb"] = np.nansum(np.real(S_vt_vt[start_index:end_index]) * df)
        out["vv_wave"] = np.nansum(np.real(S_vwave_vwave[start_index:end_index]) * df)
        out["ww_turb"] = np.nansum(np.real(S_wt_wt[start_index:end_index]) * df)
        out["ww_wave"] = np.nansum(np.real(S_wwave_wwave[start_index:end_index]) * df)
        out["uw_turb"] = np.nansum(np.real(S_ut_wt[start_index:end_index]) * df)
        out["uw_wave"] = np.nansum(np.real(S_uwave_wwave[start_index:end_index]) * df)
        out["vw_turb"] = np.nansum(np.real(S_vt_wt[start_index:end_index]) * df)
        out["vw_wave"] = np.nansum(np.real(S_vwave_wwave[start_index:end_index]) * df)
        out["uv_turb"] = np.nansum(np.real(S_ut_vt[start_index:end_index]) * df)
        out["uv_wave"] = np.nansum(np.real(S_uwave_vwave[start_index:end_index]) * df)

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

        out = {}

        u = sig.detrend(u)
        v = sig.detrend(v)
        w = sig.detrend(w)

        # All the velocity components
        _, S_uu = psd(u, fs=self.fs, **kwargs)
        _, S_vv = psd(v, fs=self.fs, **kwargs)
        _, S_ww = psd(w, fs=self.fs, **kwargs)

        # Velocity cross spectra
        _, S_uw = csd(u, w, fs=self.fs, **kwargs)
        _, S_vw = csd(v, w, fs=self.fs, **kwargs)
        f, S_uv = csd(u, v, fs=self.fs, **kwargs)

        phase_uw = np.arctan2(np.imag(S_uw), np.real(S_uw))
        phase_vw = np.arctan2(np.imag(S_vw), np.real(S_vw))
        phase_uv = np.arctan2(np.imag(S_uv), np.real(S_uv))
        df = np.nanmax(np.diff(f))

        # Searching for the wave peak within a reasonable range of frequencies --
        # adjust this for each data set
        if f_wave_low and f_wave_high:
            waverange = np.where(((f > f_wave_low) & (f < f_wave_high)))[0]
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
        out["uw_wave"] = np.nansum(Um_wave * wm_wave * np.cos(phase_uw[waverange]))
        out["uv_wave"] = np.nansum(Um_wave * Vm_wave * np.cos(phase_uv[waverange]))
        out["vw_wave"] = np.nansum(Vm_wave * wm_wave * np.cos(phase_vw[waverange]))

        out["uu_wave"] = np.nansum(Suu_wave * df)
        out["vv_wave"] = np.nansum(Svv_wave * df)
        out["ww_wave"] = np.nansum(Sww_wave * df)

        # Defining frequency range for full stress summation
        start_index, end_index = get_frequency_range(f, f_low, f_high)

        # Full reynolds stresses
        uu = np.nansum(np.real(S_uu[start_index:end_index]) * df)
        uv = np.nansum(np.real(S_uv[start_index:end_index]) * df)
        uw = np.nansum(np.real(S_uw[start_index:end_index]) * df)
        vv = np.nansum(np.real(S_vv[start_index:end_index]) * df)
        vw = np.nansum(np.real(S_vw[start_index:end_index]) * df)
        ww = np.nansum(np.real(S_ww[start_index:end_index]) * df)

        # Turbulent reynolds stresses
        out["uu_turb"] = uu - out["uu_wave"]
        out["vv_turb"] = vv - out["vv_wave"]
        out["ww_turb"] = ww - out["ww_wave"]
        out["uw_turb"] = uw - out["uw_wave"]
        out["uv_turb"] = uv - out["uv_wave"]
        out["vw_turb"] = vw - out["vw_wave"]

        return out

    def covariance(
        self,
        burst_data: Dict[str, np.ndarray],
        method: str = "cov",
        f_low: Optional[float] = None,
        f_high: Optional[float] = None,
        rho: Optional[float] = 1020,
        phase_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        """
        Calculate components of the covariance matrix.

        Parameters
        ----------
        burst_data : dict
            tbfi
        method : str
            Method to calculate covariances. Options are:
            - 'cov': Standard covariance calculation using the built-in np.cov
            - 'spectral_integral': Integrate the cross-spectrum over a specified frequency range
            - 'benilov': Benilov wave-turbulence decomposition
            - 'phase':  Bricker & Monismith phase-method wave-turbulence decomposition
        f_low : float, optional
            Lower frequency bound (Hz) for spectral integration, by default None
        f_high : float, optional
            Upper frequency bound (Hz) for spectral integration, by default None
        rho : float, optional
            Water density (kg/m^3), by default 1020
        phase_kwargs : dict, optional
            Additional arguments specific to phase decomposition method, by default None. If specified, should include
            keys 'f_wave_low' and 'f_wave_high' to define the frequency range of the wave band.
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
                u = burst_data["u"][height_idx, :]
                v = burst_data["v"][height_idx, :]
                w = burst_data["w"][height_idx, :]
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
                u = burst_data["u"][height_idx, :]
                v = burst_data["v"][height_idx, :]
                w = burst_data["w"][height_idx, :]

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
        else:
            raise IOError(f"Unrecognized method {method}")

        return out

    def dissipation(self, burst_data: dict, f_low: float, f_high: float, **kwargs) -> (float, float, int):
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

        def spectral_fit(u, v, w, f_low, f_high, **kwargs):
            """
            Carries out the spectral curve fit
            """
            if np.all(np.isnan(u)) or np.all(np.isnan(v)) or np.all(np.isnan(w)):
                return np.nan, np.nan, 0
            omega_range = [2 * np.pi * f_low, 2 * np.pi * f_high]
            alpha = 1.5

            w_prime = sig.detrend(w, type="linear")
            fw, Pw = psd(w_prime, self.fs, onesided=False, **kwargs)

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

        out = {}
        n_heights = self.n_heights
        out["eps"] = np.empty((n_heights,))
        out["noise"] = np.empty((n_heights,))
        out["quality_flag"] = np.empty((n_heights,), dtype=int)
        for height_idx in range(n_heights):
            u = burst_data["u"][height_idx, :]
            v = burst_data["v"][height_idx, :]
            w = burst_data["w"][height_idx, :]
            (eps, noise, quality_flag) = spectral_fit(u, v, w, f_low, f_high, **kwargs)
            out["eps"][height_idx] = eps
            out["noise"][height_idx] = noise
            out["quality_flag"][height_idx] = quality_flag

        return out

    def directional_wave_statistics(
        self,
        burst_data: dict,
        band_definitions: Optional[dict] = None,
        sea_correction: Optional[bool] = True,
        f_cutoff: Optional[float] = 1.0,
        rho: Optional[float] = 1020,
        **kwargs,
    ):
        if "p" not in burst_data.keys():
            raise ValueError("Pressure must be included in dataset to calculate directional wave statistics")

        n_heights = self.n_heights
        out = {}
        for height_idx in range(n_heights):
            u = burst_data["u"][height_idx, :]
            v = burst_data["v"][height_idx, :]
            p = burst_data["p"][height_idx, :]
            out[height_idx] = self.wave_worker(
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

        return out

    def wave_worker(
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
        """
        Calculate directional wave statistics from velocity and pressure measurements.

        This method computes comprehensive wave statistics including wave height, direction,
        spreading, period, and energy flux using the cross-spectral method. The analysis
        includes separate calculations for different wave bands (infragravity, swell, sea)
        and applies depth corrections for pressure measurements.

        Parameters
        ----------
        u : np.ndarray
            Horizontal velocity component in x-direction (m/s)
        v : np.ndarray
            Horizontal velocity component in y-direction (m/s)
        p : np.ndarray
            Pressure measurements (dbar)
        mab : float
            Height of pressure sensor above bed (m)
        rho : float
            Water density (kg/m³)
        band_definitions : dict, optional
            Dictionary defining frequency bands for spectral sums of the form
             {"infragravity": (f_low, f_high), "swell": (f_low, f_high), "sea": (f_low, f_high)}
             If None, uses default bands:
            - infragravity: 1/250 to 1/25 Hz
            - swell: 1/25 to 0.2 Hz
            - sea: 0.2 to 0.5 Hz
            Statistics for the full frequency range ("all") will be calculated as well.
        sea_correction : bool, optional
            Whether to apply Jones-Monismith correction for sea waves, by default True
        f_cutoff : float, optional
            Upper bound for spectral integration to avoid high frequency noise. Defaults to 1.0 Hz.
        **kwargs
            Additional arguments passed to spectral analysis functions

        Returns
        -------
        dict
            Dictionary containing wave statistics with keys including:
            - 'Hsig_all', 'Hsig_swell', 'Hsig_sea', 'Hsig_ig': Significant wave heights (m)
            - 'Hrms_all', 'Hrms_swell', 'Hrms_sea', 'Hrms_ig': RMS wave heights (m)
            - 'Tp_all', 'Tp_swell', 'Tp_sea', 'Tp_ig': Peak periods (s)
            - 'Tm_all', 'Tm_swell', 'Tm_sea', 'Tm_ig': Mean periods (s)
            - 'dir_all1', 'dir_swell1', 'dir_sea1', 'dir_ig1': Wave directions (degrees)
            - 'spread_all1', 'spread_swell1', 'spread_sea1', 'spread_ig1': Directional spread (degrees)
            - 'Us_bulk', 'Vs_bulk': Bulk Stokes drift velocities (m/s)
            - 'Us_spec', 'Vs_spec': Spectral Stokes drift velocities (m/s)
            - 'Sxx_swell', 'Syy_swell', 'Sxy_swell': Radiation stress components (N/m)
            - 'ub_var', 'ub_spec': Bottom orbital velocity statistics (m/s)

        Notes
        -----
        This method implements the cross-spectral directional wave analysis technique,
        which uses the coherence and phase relationships between pressure and velocity
        measurements to determine wave direction and spreading characteristics.

        The method applies depth corrections to convert pressure measurements to sea
        surface elevation using linear wave theory, accounting for the attenuation
        of pressure fluctuations with depth.
        """

        h = 1e4 * np.nanmean(p) / (rho * g) + mab  # Average water depth

        # Sanity check to make sure average depth is positive
        if h < 0:
            raise ValueError("Average water depth must be positive to calculate directional wave statistics.")

        u = sig.detrend(u, type="linear")
        v = sig.detrend(v, type="linear")
        p = sig.detrend(p, type="linear")

        # Calculating spectra
        f, S_uu = psd(u, fs=self.fs, **kwargs)
        f, S_vv = psd(v, fs=self.fs, **kwargs)
        f, S_pp = psd(p, fs=self.fs, **kwargs)
        f, S_uv = csd(u, v, fs=self.fs, **kwargs)
        f, S_pu = csd(p, u, fs=self.fs, **kwargs)
        f, S_pv = csd(p, v, fs=self.fs, **kwargs)
        df = np.max(np.diff(f))

        # Depth correction and spectral weighted averages
        if band_definitions is None:
            fbands = {
                "infragravity": ((f > 1 / 250) & (f <= 1 / 25) & (f <= f_cutoff)),
                "swell": ((f > 1 / 25) & (f <= 0.2) & (f <= f_cutoff)),
                "sea": ((f > 0.2) & (f <= 0.5) & (f <= f_cutoff)),
                "all": ((f > 0) & (f <= f_cutoff)),
            }
        else:
            fbands = {
                "infragravity": (
                    (f > band_definitions["infragravity"][0])
                    & (f <= band_definitions["infragravity"][1])
                    & (f <= f_cutoff)
                ),
                "swell": ((f > band_definitions["swell"][0]) & (f <= band_definitions["swell"][1]) & (f <= f_cutoff)),
                "sea": ((f > band_definitions["sea"][0]) & (f <= band_definitions["sea"][1]) & (f <= f_cutoff)),
                "all": ((f > 0) & (f <= f_cutoff)),
            }

        # Getting sea surface elevation spectrum
        omega = 2 * np.pi * f
        k = get_wavenumber(omega, h)
        attenuation_correction = 1e4 * np.cosh(k * h) / (rho * g * np.cosh(k * mab))
        S_etaeta = S_pp * (attenuation_correction**2)

        if sea_correction:
            S_etaeta = jones_monismith_correction(S_etaeta, S_pp, f)

        # Equivalent pressure
        UUpres = S_uu * (attenuation_correction**2)
        VVpres = S_vv * (attenuation_correction**2)
        UVpres = S_uv * (attenuation_correction**2)

        PUpres = S_pu * attenuation_correction
        PVpres = S_pv * attenuation_correction

        # Cospectrum and quadrature
        coPUpres = np.real(PUpres)
        coPVpres = np.real(PVpres)
        coUVpres = np.real(UVpres)

        # Directional moments -- e.g., Herbers et al., 1999.
        a1 = coPUpres / np.sqrt(S_pp * (UUpres + VVpres))
        b1 = coPVpres / np.sqrt(S_pp * (UUpres + VVpres))
        dir1 = np.degrees(np.arctan2(b1, a1))
        spread1 = np.degrees(np.sqrt(2 * (1 - (a1 * np.cos(np.radians(dir1)) + b1 * np.sin(np.radians(dir1))))))

        a2 = (UUpres - VVpres) / (UUpres + VVpres)
        b2 = 2 * coUVpres / (UUpres + VVpres)
        dir2 = np.degrees(np.arctan2(b2, a2) / 2)
        spread2 = np.degrees(
            np.sqrt(np.abs(0.5 - 0.5 * (a2 * np.cos(2 * np.radians(dir2)) + b2 * np.sin(2 * np.radians(dir2)))))
        )

        # Phase and group velocity
        Cp = omega / k
        Cg = get_cg(k, h)

        # Radiation stress
        Sxx = rho * g * ((1.5 + 0.5 * a2) * (Cg / Cp) - 0.5) * S_etaeta
        Syy = rho * g * ((1.5 - 0.5 * a2) * (Cg / Cp) - 0.5) * S_etaeta
        Sxy = rho * g * 0.5 * b2 * (Cg / Cp) * S_etaeta

        # Orbital velocity, basically following Wiberg & Sherwood (2008) but excluding
        # the factor of sqrt(2) (see Madsen 1994)
        # Time domain calculation
        u_prime = u - np.nanmean(u)
        v_prime = v - np.nanmean(v)
        u_orb_var = np.sqrt((np.nanvar(u_prime) + np.nanvar(v_prime)))

        # Spectral calculation
        u_orb_spec = np.sqrt(np.sum((S_uu + S_vv) * df))

        # Setting up output dictionary and storing the spectral output
        out = {}
        out["f"] = f
        out["df"] = df
        out["S_uu"] = S_uu
        out["S_vv"] = S_vv
        out["S_pp"] = S_pp
        out["S_uv"] = S_uv
        out["S_pu"] = S_pu
        out["S_pv"] = S_pv
        out["S_etaeta"] = S_etaeta
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
        out["Cp"] = Cp
        out["Cg"] = Cg
        out["u_orb_var"] = u_orb_var
        out["u_orb_spec"] = u_orb_spec

        # Looping over the frequency bands and adding bulk (integrated) parameters
        for band_name, band_indices in fbands.items():
            # Significant and rms wave height
            out[f"Hsig_{band_name}"] = 4 * np.sqrt(np.sum(S_etaeta[band_indices] * df))
            out[f"Hrms_{band_name}"] = np.sqrt(8 * np.sum(S_etaeta[band_indices] * df))

            # Mean frequency and period
            out[f"fm_{band_name}"] = np.sum(f[band_indices] * S_etaeta[band_indices]) / np.sum(S_etaeta[band_indices])
            out[f"Tm_{band_name}"] = 1 / out[f"fm_{band_name}"]

            # Peak frequency and period
            out[f"fp_{band_name}"] = f[band_indices][np.argmax(S_etaeta[band_indices])]
            out[f"Tp_{band_name}"] = 1 / out[f"fp_{band_name}"]

            # Directions
            out[f"a1_{band_name}"] = np.sum(a1[band_indices] * S_etaeta[band_indices]) / np.sum(S_etaeta[band_indices])
            out[f"b1_{band_name}"] = np.sum(b1[band_indices] * S_etaeta[band_indices]) / np.sum(S_etaeta[band_indices])
            out[f"a2_{band_name}"] = np.sum(a2[band_indices] * S_etaeta[band_indices]) / np.sum(S_etaeta[band_indices])
            out[f"b2_{band_name}"] = np.sum(b2[band_indices] * S_etaeta[band_indices]) / np.sum(S_etaeta[band_indices])

            out[f"dir1_{band_name}"] = np.degrees(np.arctan2(out[f"b1_{band_name}"], out[f"a1_{band_name}"]))
            out[f"dir2_{band_name}"] = np.degrees(np.arctan2(out[f"b2_{band_name}"], out[f"a2_{band_name}"]))
            out[f"spread1_{band_name}"] = np.degrees(
                np.sqrt(2 * (1 - np.sqrt(out[f"a1_{band_name}"] ** 2 + out[f"b1_{band_name}"] ** 2)))
            )
            out[f"spread2_{band_name}"] = np.degrees(
                np.sqrt(2 * (1 - np.sqrt(out[f"a2_{band_name}"] ** 2 + out[f"b2_{band_name}"] ** 2)))
            )

            # Radiation stress
            out[f"Sxx_{band_name}"] = np.sum(Sxx[band_indices] * df)
            out[f"Syy_{band_name}"] = np.sum(Syy[band_indices] * df)
            out[f"Sxy_{band_name}"] = np.sum(Sxy[band_indices] * df)

            # Bulk Stokes drift
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

            # Spectral Stokes drift (unfortunately different from the bulk estimate -- see Kumar et al. 2017)
            out[f"Us_spec_{band_name}"] = np.sum(
                S_etaeta[band_indices]
                * omega[band_indices]
                * k[band_indices]
                * (np.cosh(2 * k[band_indices] * mab) / (np.sinh(k[band_indices] * h) ** 2))
                * np.cos(np.radians(dir1[band_indices]))
                * df
            )
            out[f"Vs_spec_{band_name}"] = np.sum(
                S_etaeta[band_indices]
                * omega[band_indices]
                * k[band_indices]
                * (np.cosh(2 * k[band_indices] * mab) / (np.sinh(k[band_indices] * h) ** 2))
                * np.sin(np.radians(dir1[band_indices]))
                * df
            )
        return out

    @staticmethod
    def _align_with_principal_axis(data: dict) -> Tuple:
        """
        Calculates the direction of maximum variance from the u and v velocities (Thomson & Emery, 4.52b).

        Parameters
        ----------

        Returns
        -------
        theta : float
            direction of maximum variance in degrees, CCW positive from east
            assuming that u = eastward velocity, v = northward velocity
        """
        # (Co)variances
        u_bar = np.mean(data["u"], axis=1, keepdims=True)
        v_bar = np.mean(data["v"], axis=1, keepdims=True)
        u_prime = data["u"] - u_bar
        v_prime = data["v"] - v_bar
        u_var = np.mean(u_prime**2, axis=1)
        v_var = np.mean(v_prime**2, axis=1)
        cv = np.mean(u_prime * v_prime, axis=1)

        # Direction of maximum variance in xy-plane (heading)
        theta_h_radians = (0.5 * np.arctan2(2.0 * cv, (u_var - v_var)))

        # Pitch angle
        u_rot = data["u"] * np.cos(theta_h_radians) + data["v"] * np.sin(theta_h_radians)
        u_rot_bar = np.mean(u_rot, axis=1)
        w_bar = np.mean(data["w"], axis=1)
        theta_v_radians = np.arctan2(w_bar, u_rot_bar)

        out = (np.rad2deg(theta_h_radians), np.rad2deg(theta_v_radians))
        return out

    @staticmethod
    def _align_with_current(burst_data: dict) -> Tuple:
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

        u_bar = np.mean(burst_data["u"], axis=1)
        v_bar = np.mean(burst_data["v"], axis=1)
        w_bar = np.mean(burst_data["w"], axis=1)
        U = np.sqrt(u_bar ** 2 + v_bar ** 2)
        theta_h = np.arctan2(v_bar, u_bar)
        theta_v = np.arctan2(w_bar, U)
        out = (np.rad2deg(theta_h), np.rad2deg(theta_v))
        return out

    def _rotate_velocity(self, data: Dict[str, np.ndarray], theta_h, theta_v):
        """
        Rotates u, v, w velocities by directions defined by theta_h and theta_v.

        Parameters
        ----------
        data : dict
            Dictionary containing "u", "v", and "w" velocity arrays with shape (M, N)
        theta_h : float or np.ndarray
            Horizontal rotation angle(s) in degrees, scalar or shape (M,)
        theta_v : float or np.ndarray
            Vertical rotation angle(s) in degrees, scalar or shape (M,)

        Returns
        -------
        data : dict
            Original data dictionary with "u", "v", and "w" velocity arrays rotated
        """
        # (M,) or scalar → (M, 1) for broadcasting against (M, N)
        th = np.deg2rad(np.atleast_1d(theta_h))[:, np.newaxis]
        tv = np.deg2rad(np.atleast_1d(theta_v))[:, np.newaxis]

        cos_h, sin_h = np.cos(th), np.sin(th)
        cos_v, sin_v = np.cos(tv), np.sin(tv)

        u_rot = (
            data["u"] * cos_h * cos_v
            + data["v"] * sin_h * cos_v
            + data["w"] * sin_v
        )
        v_rot = -data["u"] * sin_h + data["v"] * cos_h
        w_rot = (
            -data["u"] * cos_h * sin_v
            - data["v"] * sin_h * sin_v
            + data["w"] * cos_v
        )
        data["u"] = u_rot
        data["v"] = v_rot
        data["w"] = w_rot
        return data

