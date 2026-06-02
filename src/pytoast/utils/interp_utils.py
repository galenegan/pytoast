import numpy as np
import pandas as pd


def naninterp(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate over NaNs in a 1-D array.

    Edge NaNs are filled by forward / backward fill. The input is coerced to
    `float64`; non-floating dtypes are accepted.

    Parameters
    ----------
    x : np.ndarray
        1-D array.

    Returns
    -------
    np.ndarray
        1-D `float64` array of the same length as `x` with NaNs filled.

    Raises
    ------
    ValueError
        If `x` is not 1-D.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"naninterp expects a 1-D array, got ndim={x.ndim}")
    return np.asarray(pd.Series(x).interpolate(method="linear").ffill().bfill().to_numpy(copy=True))


def interp_rows(u: np.ndarray) -> np.ndarray:
    """Linearly interpolate over NaNs along the last axis of an N-D array.

    The input is coerced to `float64` and never mutated; a new array of the
    same shape is returned. 1-D input returns 1-D output.

    Parameters
    ----------
    u : np.ndarray
        N-D array with `ndim >= 1`. The last axis is treated as the time axis;
        all other axes are flattened, interpolated row-by-row, and restored.

    Returns
    -------
    np.ndarray
        `float64` array of the same shape as `u`.

    Raises
    ------
    ValueError
        If `u` is 0-D.
    """
    u = np.asarray(u, dtype=float)
    if u.ndim < 1:
        raise ValueError(f"interp_rows expects ndim >= 1, got ndim={u.ndim}")
    if u.ndim == 1:
        return naninterp(u)
    orig_shape = u.shape
    out = u.reshape(-1, orig_shape[-1]).copy()
    for i in range(out.shape[0]):
        out[i] = naninterp(out[i])
    return out.reshape(orig_shape)
