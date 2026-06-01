import numpy as np
import scipy.signal as sig
from typing import Optional, Tuple


def get_window_len(N: int, num_windows: int) -> int:
    """
    Welch-method window length.

    Parameters
    ----------
    N : int
        Number of samples in the time series.
    num_windows : int
        Number of (50%-overlapping) windows desired.

    Returns
    -------
    int
        Window length in samples.
    """
    return int(2 * N / (num_windows + 1))


def get_frequency_range(
    f: np.ndarray, f_low: Optional[float] = None, f_high: Optional[float] = None
) -> Tuple[int, int]:
    """
    Index range into ``f`` covering [f_low, f_high].

    Parameters
    ----------
    f : np.ndarray
        Monotonically increasing frequency vector (Hz).
    f_low : float, optional
        Lower frequency bound (Hz). If None, start at index 0.
    f_high : float, optional
        Upper frequency bound (Hz). If None, end at ``len(f)``.

    Returns
    -------
    tuple of int
        ``(start_index, end_index)`` into ``f``.
    """
    if f_low is not None:
        start_index = int(np.argmin(np.abs(f - f_low)))
    else:
        start_index = 0

    if f_high is not None:
        end_index = int(np.argmin(np.abs(f - f_high)))
    else:
        end_index = len(f)

    return start_index, end_index


def psd(
    x: np.ndarray,
    fs: float,
    num_windows: int = 8,
    window_type: str = "hamming",
    window_len: Optional[int] = None,
    nfft: Optional[int] = None,
    detrend: bool = False,
    onesided: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Power spectral density via Welch's method.

    Parameters
    ----------
    x : np.ndarray
        Input signal. The longest axis is treated as time.
    fs : float
        Sampling frequency (Hz).
    num_windows : int, optional
        Number of (50%-overlapping) Welch windows when ``window_len`` is not given.
    window_type : str, optional
        Window passed to ``scipy.signal.welch`` (default ``'hamming'``).
    window_len : int, optional
        Window length in samples. If None, derived from ``num_windows`` and ``N``.
    nfft : int, optional
        FFT length. Defaults to ``window_len``.
    detrend : bool, optional
        If True, detrend each segment before transforming.
    onesided : bool, optional
        If True, return the one-sided spectrum.

    Returns
    -------
    f : np.ndarray
        Frequency vector (Hz).
    Pxx : np.ndarray
        Power spectral density (units of ``x``^2 / Hz).
    """
    N = max(x.shape)
    if window_len is None:
        window_len = get_window_len(N, num_windows)
    if nfft is None:
        nfft = window_len

    f, Pxx = sig.welch(
        x=x,
        fs=fs,
        window=window_type,
        nperseg=window_len,
        nfft=nfft,
        detrend=detrend,
        return_onesided=onesided,
    )

    return f, Pxx


def csd(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    num_windows: int = 8,
    window_type: str = "hamming",
    window_len: Optional[int] = None,
    nfft: Optional[int] = None,
    detrend: bool = False,
    onesided: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cross spectral density via Welch's method.

    Parameters
    ----------
    x, y : np.ndarray
        Two input signals of identical shape; the longest axis is time.
    fs : float
        Sampling frequency (Hz).
    num_windows : int, optional
        Number of (50%-overlapping) Welch windows when ``window_len`` is not given.
    window_type : str, optional
        Window passed to ``scipy.signal.csd``.
    window_len : int, optional
        Window length in samples. If None, derived from ``num_windows`` and ``N``.
    nfft : int, optional
        FFT length. Defaults to ``window_len``.
    detrend : bool, optional
        If True, detrend each segment before transforming.
    onesided : bool, optional
        If True, return the one-sided spectrum.

    Returns
    -------
    f : np.ndarray
        Frequency vector (Hz).
    Pxy : np.ndarray
        Complex cross spectral density (units of x*y / Hz).
    """
    N = max(x.shape)
    if window_len is None:
        window_len = get_window_len(N, num_windows)
    if nfft is None:
        nfft = window_len

    f, Pxy = sig.csd(
        x=x,
        y=y,
        fs=fs,
        window=window_type,
        nperseg=window_len,
        nfft=nfft,
        detrend=detrend,
        return_onesided=onesided,
    )

    return f, Pxy
