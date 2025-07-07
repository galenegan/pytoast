from abc import ABC, abstractmethod
import os
import xarray as xr
from typing import Union, List, Optional


class BaseInstrument(ABC):
    """Abstract base class for instruments, defining shared methods and attributes"""

    def __init__(self, dataset: xr.Dataset):
        super().__setattr__("ds", dataset)

    def __getattr__(self, name):
        if name in self.ds.variables:
            return self.ds[name]
        return getattr(self.ds, name)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setattr__(self, name, value):
        if "ds" in self.__dict__ and name in self.ds.variables:
            # Assign to dataset variable if it already exists
            self.ds[name] = value
        else:
            # Otherwise, assign normally (e.g., during __init__)
            super().__setattr__(name, value)

    @abstractmethod
    def _get_required_variables(self) -> List[str]:
        """Each instrument defines its required variables"""
        pass

    @staticmethod
    def validate_common_inputs(
        files: Union[str, List],
        name_map: dict,
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[float, int, List[Union[float, int]]]] = None,
        zarr_save_path: Optional[str] = None,
        overwrite: Optional[bool] = False,
    ):
        # Validate "files"
        if isinstance(files, str):
            if not files.lower().endswith((".nc", ".mat", ".zarr")):
                raise ValueError(f"If files is a string, it must be a .nc, .mat, or .zarr file. Got: {files}")
            if not os.path.exists(files):
                raise FileNotFoundError(f"The specified file does not exist: {files}")

        elif isinstance(files, list):
            valid_extensions = (".npy", ".mat", ".csv")
            for file in files:
                if not isinstance(file, str) or not file.lower().endswith(valid_extensions):
                    raise ValueError(
                        f"Each element in the files list must be a path ending in {valid_extensions}. Got: {file}"
                    )
                if not os.path.exists(file):
                    raise FileNotFoundError(f"The specified file does not exist: {file}")
        else:
            raise TypeError("`files` must be a string or a list of strings")

        if not isinstance(name_map, dict):
            raise TypeError("`name_map` must be a dictionary")

        if "time" not in name_map and fs is None:
            raise ValueError("You must specify either 'time' in name_map or provide 'fs'")

        # Validate "z"
        if z is not None:
            if not isinstance(z, (float, int, list)):
                raise TypeError("`z` must be either a float, int, or a list of floats/ints")
            if isinstance(z, list) and not all(isinstance(zi, (float, int)) for zi in z):
                raise TypeError("All elements of the `z` list must be floats or ints")

        # Validate "fs"
        if fs is not None and not isinstance(fs, (int, float)):
            raise TypeError("`fs` must be either an int or a float")

        # Validate "zarr_save_path"
        if zarr_save_path is not None:
            if not isinstance(zarr_save_path, str):
                raise TypeError("`zarr_save_path` must be a string")
            if os.path.exists(zarr_save_path) and not overwrite:
                raise FileExistsError(
                    f"The specified zarr_save_path already exists: {zarr_save_path}. Set overwrite=True to overwrite it."
                )

        # Validate "overwrite"
        if not isinstance(overwrite, bool):
            raise TypeError("`overwrite` must be a boolean")

    @classmethod
    def from_saved_zarr(cls, zarr_path: str):
        """
        Load an object from a zarr store that was previously saved to disk by pyToast.
        If you are want to initialize an object from your own zarr store for the first
        time, you should use {InstrumentClass}.from_raw(files="path/to/zarr", ...)

        Parameters
        ----------
        zarr_path : string
            Path to zarr store on local disk

        Returns
        -------
        BaseInstrument object
        """
        ds = xr.open_zarr(zarr_path)
        return cls(ds)
