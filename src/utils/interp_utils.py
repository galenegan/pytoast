import pandas as pd
from string import ascii_lowercase
import xarray as xr


def naninterp_pd(x):
    return pd.Series(x).interpolate(method="linear").ffill().bfill().values


def naninterp_xr(x):
    dims = [letter for letter in ascii_lowercase[: len(x.shape)]]
    da = xr.DataArray(x, dims=dims)

    # Interpolate NaNs along axis=-1, and fill edges
    interp_da = da.interpolate_na(dim=dims[-1], method="linear", fill_value="extrapolate")

    return interp_da.values
