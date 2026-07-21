import tempfile

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

RC_PARAMS = {
    "axes.labelsize": 16,
    "font.size": 13,
    "legend.fontsize": 12,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "text.usetex": False,
    "font.family": "sans-serif",
    "axes.grid": False,
}

for _key, _val in RC_PARAMS.items():
    plt.rcParams[_key] = _val

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


def get_frequency_range(f: np.ndarray, f_low: float | None = None, f_high: float | None = None) -> tuple[int, int]:
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
    window_len: int | None = None,
    nfft: int | None = None,
    detrend: bool = False,
    onesided: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
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
    window_len: int | None = None,
    nfft: int | None = None,
    detrend: bool = False,
    onesided: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
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


def plot_spectral_fit(
    x: np.ndarray,
    y: np.ndarray,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    eps: float,
    xlabel: str = r"Wavenumber $k$ (rad/m)",
    ylabel: str = r"Spectral density",
    title: str | None = None,
    out_file: str | None = None,
) -> str:
    """
    Save a log-log plot of a spectral curve fit to a PNG file.

    Draws the observed spectrum as scattered points together with the fitted -5/3 curve.
    The caller supplies the data points and the pre-computed fit line, so this function is
    agnostic to the specific model or independent variable (wavenumber or angular frequency).

    Parameters
    ----------
    x : np.ndarray
        Independent variable of the observed data points over the fit range (e.g. wavenumber
        in rad/m or angular frequency in rad/s).
    y : np.ndarray
        Observed spectral density at ``x``.
    x_fit : np.ndarray
        Independent variable of the fitted curve (e.g. a linspace over the fit range).
    y_fit : np.ndarray
        Fitted spectral density evaluated at ``x_fit``.
    eps : float
        Dissipation rate estimate (m^2/s^3), shown in the legend label.
    xlabel : str, optional
        Label for the x-axis. Defaults to a wavenumber label.
    ylabel : str, optional
        Label for the y-axis. Defaults to a generic spectral density label.
    title : str, optional
        Axis title. If None, no title is drawn.
    out_file : str, optional
        Destination PNG path. If None, a temporary file (suffix ".png") is created.

    Returns
    -------
    str
        Path to the saved PNG file.
    """
    if out_file is None:
        out_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, y, "o", label="Data", alpha=0.5)
    ax.plot(x_fit, y_fit, linewidth=2, label=f"Fit (eps={eps:.3e})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    if title is not None:
        ax.set_title(title)
    ax.legend()
    ax.xaxis.set_major_formatter("{x:.1f}")
    ax.xaxis.set_minor_formatter("{x:.1f}")
    fig.tight_layout(pad=0.5)
    fig.savefig(out_file, dpi=300)
    plt.close(fig)

    return out_file
