import numpy as np
import scipy.signal as sig
from typing import Optional, Union
from utils.constants import GRAVITATIONAL_ACCELERATION as g, WATER_DENSITY as rho0
from utils.spectral_utils import psd, csd


def get_wavenumber(
    omega: Union[float, np.ndarray], h: Union[float, np.ndarray], max_iter: int = 10, tol: float = 1e-10
):
    """Calculate wavenumber from the surface gravity wave dispersion relation using Newton's method.

    Parameters
    ----------
    omega : float or np.ndarray
        Angular frequency (rad/s)
    h : float or np.ndarray
        Water depth (m)
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance

    Returns
    -------
    k : float or np.ndarray
        Wavenumber (rad/m)
    """
    omega = np.asarray(omega, dtype=float)
    h = np.broadcast_to(np.asarray(h, dtype=float), omega.shape)
    k = np.where(omega == 0, 0.0, omega / np.sqrt(g * h))
    mask = omega != 0
    for _ in range(max_iter):
        th = np.tanh(k * h)
        f = g * k * th - omega**2
        if np.max(np.abs(f[mask])) < tol:
            break
        dfdk = g * h * k / np.cosh(k * h) ** 2 + g * th
        k = np.where(mask, k - f / np.where(mask, dfdk, 1.0), 0.0)
    return k.item() if k.ndim == 0 else k


def get_cg(k: Union[float, np.ndarray], h: Union[float, np.ndarray]):
    """Returns the group velocity from the linear wave theory dispersion relation.

    Parameters
    ----------
    k : float or np.ndarray
        Wavenumber (rad/m)
    h : float or np.ndarray
        Water depth (m)

    Returns
    -------
    cg : float or np.ndarray
        Group velocity (m/s)
    """
    cp = np.sqrt((g / k) * np.tanh(k * h))
    cg = 0.5 * cp * (1 + (k * h) * (1 - (np.tanh(k * h)) ** 2) / np.tanh(k * h))

    return cg


def jones_monismith_correction(
    S_etaeta: np.ndarray,
    S_pp: np.ndarray,
    f: np.ndarray,
    f_cutoff: Optional[float] = 0.5,
):
    """Apply Jones & Monismith (2008) correction for high frequency noise
    introduced by the pressure attenuation.

    Parameters
    ----------
    S_etaeta : np.ndarray
        Sea surface elevation power spectral density (m^2/Hz)
    S_pp : np.ndarray
        Pressure power spectral density (dbar^2/Hz)
    f : np.ndarray
        Frequency array (Hz)
    f_cutoff : float, optional
        Maximum frequency to consider for peak detection (Hz). If None,
        uses the full frequency range.

    Returns
    -------
    S_etaeta_corrected : np.ndarray
        Corrected sea surface elevation power spectral density with
        f^-4 tail applied above 1.1 f_p

    Notes
    -----
    The correction procedure:
    1. Identifies the spectral peak in the pressure spectrum
    2. Finds a cutoff frequency where the spectrum approaches the noise floor
    3. Ensures cutoff is at least 1.1 times the peak frequency
    4. Replaces spectrum above cutoff with theoretical f^-4 tail

    References
    ----------
    Jones, N. L., & Monismith, S. G. (2008). The influence of whitecapping on wave height and period statistics. Journal
        of Physical Oceanography, 38(7), 1473-1490.
    """
    # Finding peak and cutoff frequency
    S_out = S_etaeta.copy()
    df = np.max(np.diff(f))
    noise_floor = np.mean(S_pp[f < 3 * df])

    global_cutoff = np.argmin(np.abs(f - f_cutoff)) if f_cutoff is not None else len(f)
    index_peak = np.argmax(S_pp[:global_cutoff])
    index_cutoff = np.argmin(np.abs(S_pp[index_peak:] - noise_floor * 12)) + index_peak

    # Increase this up to 1.1 f_p if necessary
    while f[index_cutoff] <= 1.1 * f[index_peak]:
        index_cutoff += 1

    # Replacing the rest of the spectrum with a 10^-4 fit
    m = S_etaeta[index_cutoff] * (f[index_cutoff] ** 4)
    S_out[index_cutoff:] = m * (f[index_cutoff:]) ** (-4)

    return S_out


def wave_stats(
    u: np.ndarray,
    v: np.ndarray,
    p: np.ndarray,
    fs: float,
    mab: float,
    rho: float = rho0,
    band_definitions: Optional[dict] = None,
    sea_correction: bool = True,
    f_cutoff: float = 1.0,
    **kwargs,
) -> dict:
    """
    Helper function for calculating all directional wave statistics

    Parameters
    ----------
    u : np.ndarray
        u1 velocity (m/s)
    v : np.ndarray
        u2 velocity (m/s)
    p : np.ndarray
        Pressure (dbar)
    fs : float
        Sampling frequency (Hz)
    mab : float
        pressure sensor meters above bed
    rho : float
        Water density (kg/m^3)
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

    Jones, N. L., & Monismith, S. G. (2007). Measuring short-period wind waves in a tidally forced environment with
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
    # Calculate water depth (m) prior to detrending
    h = 1e4 * np.nanmean(p) / (rho * g) + mab

    u = sig.detrend(u, type="linear")
    v = sig.detrend(v, type="linear")
    p = sig.detrend(p, type="linear")

    # Calculating spectra
    f, P_uu = psd(u, fs=fs, **kwargs)
    f, P_vv = psd(v, fs=fs, **kwargs)
    f, P_pp = psd(p, fs=fs, **kwargs)
    f, P_uv = csd(u, v, fs=fs, **kwargs)
    f, P_pu = csd(p, u, fs=fs, **kwargs)
    f, P_pv = csd(p, v, fs=fs, **kwargs)
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
        f_bands[band_name] = (f >= f_low) & (f < f_high) & (f < f_cutoff)
    f_bands["all"] = (f > 0) & (f < f_cutoff)

    # Getting sea surface elevation spectrum
    omega = 2 * np.pi * f
    k = get_wavenumber(omega, h)
    z = mab - h

    # cosh(k(z+h))/cosh(kh) = (e^{kz}+e^{-k(z+2h)}) / (1+e^{-2kh})
    cosh_term = (1 + np.exp(-2 * k * h)) / (np.exp(k * z) + np.exp(-k * (z + 2 * h)))
    attenuation_correction = (1e4 / (rho * g)) * cosh_term
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
    spread2 = np.degrees(np.sqrt(0.5 * (1 - (a2 * np.cos(2 * np.radians(dir2)) + b2 * np.sin(2 * np.radians(dir2))))))

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
