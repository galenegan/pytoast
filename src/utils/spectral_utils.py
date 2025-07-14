import numpy as np
import scipy.signal as sig
from typing import Optional


def get_window_len(N: int, num_windows: int) -> int:
    """
    Returns the welch method window length
    :param N:
    :return:
    """
    return int(2 * N / (num_windows + 1))


def get_frequency_range(f: np.ndarray, f_low: Optional[float] = None, f_high: Optional[float] = None) -> (int, int):

    if f_low:
        start_index = np.argmin(np.abs(f - f_low))
    else:
        start_index = 0

    if f_high:
        end_index = np.argmin(np.abs(f - f_high))
    else:
        end_index = len(f)

    return start_index, end_index


def psd(
    x: np.ndarray,
    fs: float,
    num_windows=8,
    window_type="hamming",
    window_len=None,
    nfft=None,
    detrend=False,
    onesided=True,
) -> (np.ndarray, np.ndarray):
    N = x.shape[-1]
    if window_len is None:
        window_len = get_window_len(N, num_windows)
    if nfft is None:
        nfft = window_len

    f, Pxx = sig.welch(
        x=x, fs=fs, window=window_type, nperseg=window_len, nfft=nfft, detrend=detrend, return_onesided=onesided
    )

    return f, Pxx


def csd(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    num_windows=8,
    window_type="hamming",
    window_len=None,
    nfft=None,
    detrend=False,
    onesided=True,
) -> (np.ndarray, np.ndarray):
    N = len(x)
    if window_len is None:
        window_len = get_window_len(N, num_windows)
    if nfft is None:
        nfft = window_len

    f, Pxy = sig.csd(
        x=x, y=y, fs=fs, window=window_type, nperseg=window_len, nfft=nfft, detrend=detrend, return_onesided=onesided
    )

    return f, Pxy
