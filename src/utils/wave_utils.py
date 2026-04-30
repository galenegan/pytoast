import numpy as np
from typing import Optional, Union
from utils.constants import GRAVITATIONAL_ACCELERATION as g


def get_wavenumber(omega: Union[float, np.ndarray], h: Union[float, np.ndarray], max_iter: int = 10, tol: float = 1e-10):
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
        f = g * k * th - omega ** 2
        if np.max(np.abs(f[mask])) < tol:
            break
        dfdk = g * h * k / np.cosh(k * h) ** 2 + g * th
        k = np.where(mask, k - f / np.where(mask, dfdk, 1.0), 0.0)
    return k.item() if k.ndim == 0 else k


def get_cg(k, h):
    """Returns the group velocity from the linear wave theory dispersion
    relation.

    Parameters
    ----------
    k
    h

    Returns
    -------
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

    The f^-4 tail is consistent with theoretical expectations for
    equilibrium wave spectra at high frequencies and helps remove
    artifacts from sea wave contamination in swell band measurements.

    References
    ----------
    Jones, N. L., & Monismith, S. G. (2008). The influence of whitecapping
    on wave height and period statistics. Journal of Physical Oceanography,
    38(7), 1473-1490.
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
