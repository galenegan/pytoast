import glob
import numpy as np
from typing import Optional, Union, List
from numpy.typing import ArrayLike
from scipy import io as sio
from pathlib import Path
import xarray as xr
from utils.parsing_utils import DatasetParser


class ADV:
    def __init__(self, dataset: xr.Dataset):
        self._ds = dataset

    @staticmethod
    def validate_inputs(files, name_map, fs, z):
        # TODO: validate all inputs to from_raw
        pass

    @classmethod
    def from_raw(
        cls,
        files: Union[str, List],
        name_map: dict,
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[float, List[float]]] = None,
        zarr_save_path: Optional[str] = None,
        overwrite: Optional[bool] = False,
    ):
        """
        Args:
            :param files: if str, must be a path to a netCDF file, .mat file, or zarr file store that contains the
            entire dataset you wish to load. If list, the elements of the list will be interpreted as files containing data
            from individual measurement burst periods. Supported burst file types are .npy (assuming it was saved as a
            dictionary), .mat (assuming it was saved as a Matlab Struct) and .csv (with variables in separate columns).
            If the variables associated with a particular name are two-dimensional, then the larger dimension is
            assumed to be time and the shorter dimension is assumed to be a vertical coordinate. In this case,
            a "z" list must be passed as an argument with a length matching the size of the shorter dimension

            :param name_map: a dictionary of the form:
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

            :param fs: sampling frequency (Hz), optional unless name_map["time"] is not specified

            :param z: mean height above the bed (m) for each instrument, optional unless the velocity variables have
            more than 1 dimension

        """
        ADV.validate_inputs(files, name_map, fs, z)
        parser = DatasetParser(files, name_map, fs, z)
        ds = parser.parse_input()
        if zarr_save_path and (overwrite or not Path(zarr_save_path).exists()):
            ds.to_zarr(zarr_save_path, consolidated=True)

        return cls(ds)

    @classmethod
    def from_saved_zarr(cls, zarr_path: str):
        ds = xr.open_zarr(zarr_path)
        return cls(ds)

    def __getattr__(self, name):
        if name in self._ds.variables:
            return self._ds[name]
        return getattr(self._ds, name)


if __name__ == "__main__":
    mean_depth = 13
    m_below_surface = np.linspace(1.8, 7.2, 6)
    mabs = [mean_depth - mbs for mbs in m_below_surface]

    # Testing this out
    files = glob.glob("/Users/ea-gegan/Documents/gitrepos/tke-budget/data/adv_fall/*.mat")[:10]

    # Name map:
    name_map = {"u": "E", "v": "N", "w": "w", "p": "P2", "time": "dn"}
    adv = ADV.from_raw(files, name_map)
