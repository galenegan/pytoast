import numpy as np
import pandas as pd


def naninterp(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).interpolate(method="linear").ffill().bfill().values


def interp_rows(u: np.ndarray) -> np.ndarray:
    """Apply naninterp_pd independently to each row."""
    for i in range(u.shape[0]):
        u[i] = naninterp(u[i])
    return u
