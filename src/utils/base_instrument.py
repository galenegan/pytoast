from abc import ABC
import numpy as np
import os
import sys
from typing import Any, List, Union, Optional, Dict
import scipy.io as sio
import pandas as pd
import xarray as xr

from utils.io_utils import results_to_dataset


class BaseInstrument(ABC):
    """Abstract base class containing data loading and parsing methods that are
    used across instruments."""

    def __init__(
        self,
        files: Union[str, List[str]],
        name_map: dict,
        deployment_type: str = "fixed",
        fs: Optional[float] = None,
        z: Optional[Union[float, List[float], np.ndarray]] = None,
        data_keys: Optional[Union[str, List[str]]] = None,
        burst_dim: Optional[str] = None,
    ):
        """Base class initialization.

        Parameters
        ----------
        files : str or List[str]
            Path(s) to data file(s)
        name_map : dict
            Mapping of variable names
        deployment_type : str, optional
            One of {"fixed", "cast"} depending on how the instrument is deployed. Default is "fixed", in which case
            self.z will be converted to a constant numpy array of instrument deployment depths or measurement cell
            heights. If "cast", self.z will be set to None and vertical coordinates will be calculated as a data
            variable within individual measurement bursts.
        fs : float, optional
            Sampling frequency
        z : float, List[float], or np.ndarray, optional
            Height coordinates
        data_keys : str or List[str], optional
            One or more nested keys to traverse after loading a file (e.g. `"Data"` if
            variables in `name_map` live at `file["Data"]["variable_name"]`)
        burst_dim : str, optional
            Name of the burst dimension inside a monolithic NetCDF file. When given, `files`
            must be a single `.nc` path; the file is opened lazily with `xr.open_dataset` and
            each burst is exposed by slicing along this dimension. When None (default), each
            entry in `files` is treated as one burst.
        """
        files = files if isinstance(files, list) else [files]
        self.validate_common_inputs(files, name_map, deployment_type, fs, z)
        self.files = files
        self.name_map = name_map
        self.deployment_type = deployment_type
        self.data_keys = [data_keys] if isinstance(data_keys, str) else (list(data_keys) if data_keys else [])
        self.burst_dim = burst_dim
        self._monolithic_ds = None
        if burst_dim is not None:
            if len(files) != 1 or not files[0].lower().endswith(".nc"):
                raise ValueError("`burst_dim` requires `files` to be a single .nc path")
            ds = xr.open_dataset(files[0])
            if burst_dim not in ds.dims:
                raise ValueError(f"burst_dim {burst_dim!r} not found in dataset dims {tuple(ds.dims)}")
            self._monolithic_ds = ds
        self.fs, self.z, self.file_type, self.num_samples_per_burst = self._inspect_first_file(fs, z, deployment_type)
        self._cached_idx = None
        self._cached_data = None
        self._preprocess_enabled = False

    @staticmethod
    def validate_common_inputs(
        files: List[str],
        name_map: dict,
        deployment_type: str,
        fs: Optional[float] = None,
        z: Optional[Union[float, List[float], np.ndarray]] = None,
        data_keys: Optional[Union[str, List[str]]] = None,
    ):
        """Validate common input parameters shared across all instruments.

        Parameters
        ----------
        files : List[str]
            Input files
        name_map : dict
            Variable name mapping
        deployment_type : str
            "fixed" or "cast"
        fs : float, optional
            Sampling frequency
        z : float, List[float], or np.ndarray, optional
            Height coordinates
        data_keys : str or List[str], optional
            Additional keys to traverse after loading a file

        Raises
        ------
        ValueError
            If input parameters are invalid
        TypeError
            If parameter types are incorrect
        FileNotFoundError
            If specified files don't exist
        """
        # Validate "files"
        valid_extensions = (".npy", ".mat", ".csv", ".nc")
        if isinstance(files, list):
            for file in files:
                if not isinstance(file, str) or not file.lower().endswith(valid_extensions):
                    raise ValueError(
                        f"Each element in files list must be a path ending in one of {valid_extensions}. Got: {file}"
                    )
                if not os.path.exists(file):
                    raise FileNotFoundError(f"The specified file does not exist: {file}")
        else:
            raise TypeError("`files` must be a list")

        if not isinstance(name_map, dict):
            raise TypeError("`name_map` must be a dictionary")

        if "time" not in name_map and fs is None:
            raise ValueError("You must specify either 'time' in name_map or provide 'fs'")

        if not isinstance(deployment_type, str):
            raise TypeError("`deployment_type` must be a string")

        if deployment_type not in ["fixed", "cast"]:
            raise ValueError("`deployment_type` must be either 'fixed' or 'cast'")

        # Validate "z"
        if z is not None:
            if not isinstance(z, (float, int, list, np.ndarray)):
                raise TypeError("`z` must be either a float, int, list, or numpy array")
            if isinstance(z, list) and not all(isinstance(zi, (float, int)) for zi in z):
                raise TypeError("All elements of the `z` list must be floats or ints")

        # Validate "fs"
        if fs is not None and not isinstance(fs, (int, float)):
            raise TypeError("`fs` must be either an int or a float")

        # Validate "data_keys"
        if data_keys is not None:
            if not isinstance(data_keys, (str, list)):
                raise TypeError("`data_keys` must be either a string or a list")

    @staticmethod
    def _load_file(file_path, data_keys=None):
        suffix = file_path.split(".")[-1].lower()
        if suffix == "mat":
            data = strip_mat_nulls(sio.loadmat(file_path, simplify_cells=True))
            file_type = "mat"
        elif suffix == "npy":
            data = np.load(file_path, allow_pickle=True).item()
            file_type = "npy"
        elif suffix == "csv":
            data = pd.read_csv(file_path)
            file_type = "csv"
        elif suffix == "nc":
            data = xr.open_dataset(file_path)
            file_type = "nc"
        else:
            raise Exception(f"Unrecognized file type .{suffix} for filepath input")

        for key in data_keys or []:
            data = data[key]

        return data, file_type

    @staticmethod
    def _as_array(data: Any, key: str, file_type: str) -> np.ndarray:
        """Extract variable `key` from `data` as a numpy array.

        Centralizes extraction across dict (mat/npy), pandas DataFrame
        (csv), and xarray Dataset (nc). For xarray-backed data,
        accessing `.values` triggers a load of the sliced bytes.
        """
        value = data[key]
        if file_type == "nc":
            return np.asarray(value.values)
        if file_type == "csv":
            return np.asarray(value.values if hasattr(value, "values") else value)
        return np.asarray(value)

    def _inspect_first_file(self, fs, z, deployment_type):
        """Read the first file to infer fs and z (if not provided) and
        determine file_type and num_samples_per_burst.

        Parameters
        ----------
        fs : float or None
            Sampling frequency, or None to infer from time variable
        z : float, list, np.ndarray, or None
            Height coordinates, or None to infer from data dimensions
        deployment_type : str
            Either "fixed" or "cast". Determines whether z is a constant array of height coordinates or None.

        Returns
        -------
        fs : float
            Sampling frequency (provided or inferred)
        z : np.ndarray or None
            If `deployment_type == "fixed"`, height coordinates as a numpy array (provided or inferred). If
            `deployment` == "cast", None.
        file_type : str
            File format identifier (`"mat"`, `"npy"`, `"csv"`, `"nc"`)
        num_samples_per_burst : int
            Number of samples per burst
        """
        if not self.files:
            raise ValueError("No files provided")

        if self._monolithic_ds is not None:
            data = self._monolithic_ds.isel({self.burst_dim: 0})
            file_type = "nc"
        else:
            data, file_type = self._load_file(self.files[0], self.data_keys)

        # Normalize z to a numpy array, or infer from data dimensions
        self._physical_z = False
        if deployment_type == "cast":
            z = None
        else:
            if z is not None:
                self._physical_z = True
                if isinstance(z, (int, float)):
                    z = np.array([z])
                elif isinstance(z, list):
                    z = np.array(z)
            elif "z" in self.name_map:
                self._physical_z = True
                arr = self._as_array(data, self.name_map["z"], file_type)
                if arr.ndim == 0:
                    z = np.array([float(arr)])
                else:
                    z = np.asarray(arr)
            else:
                non_time_key = [key for key in self.name_map.keys() if key != "time"][0]
                if isinstance(non_time_key, str):
                    data_var = self._as_array(data, non_time_key, file_type)
                    if data_var.ndim > 1:
                        num_rows, num_cols = data_var.shape
                        if num_rows > num_cols:
                            data_var = data_var.T
                        z = np.arange(data_var.shape[0])
                    else:
                        z = np.array([0])
                else:
                    z = np.arange(len(non_time_key))

        # Determine num_samples and infer fs if needed
        if "time" not in self.name_map:
            first_out_key = list(self.name_map.keys())[0]
            data_var = self._as_array(data, first_out_key, file_type)
            if data_var.ndim > 1:
                num_rows, num_cols = data_var.shape
                num_samples = max(num_rows, num_cols)
            else:
                num_samples = len(data_var)
        else:
            time_array = self._as_array(data, self.name_map["time"], file_type)
            num_samples = len(time_array)
            if fs is None:
                datetime_array = self.process_time(time_array)
                fs = np.round(1 / ((datetime_array[1] - datetime_array[0]).astype(int) / 10**9), 2)

        return fs, z, file_type, num_samples

    def process_time(self, time_array: np.ndarray) -> np.ndarray:
        """Convert a time array to numpy datetime64 format.

        Parameters
        ----------
        time_array : np.ndarray
            Array of time values (datestrings, MATLAB datenums, or Unix epoch)

        Returns
        -------
        np.ndarray
            Array of datetime64 values with same shape as input
        """
        flattened_time = time_array.flatten()
        time_format = self.detect_time_format(flattened_time[0])
        if time_format == "datestring":
            datetime_array = pd.to_datetime(flattened_time).values
        elif time_format == "matlab":
            datetime_array = pd.to_datetime(flattened_time - 719529, unit="D").values
        elif time_format == "epoch":
            datetime_array = pd.to_datetime(flattened_time, unit="s").values

        return datetime_array.reshape(time_array.shape)

    @staticmethod
    def detect_time_format(time_input: Union[float, int, str]) -> str:
        """Detect if a time input represents Unix epoch time, MATLAB datenum,
        or a datestring.

        Args:
            time_input (float): The input float to test.

        Returns:
            str: `"datestring"`, `"epoch"`, `"matlab"`. Raises an exception if there is no match
        """
        # Rough numeric ranges as of 2020s:
        # Epoch: ~1.5e9 (1970-2020s)
        # MATLAB: ~7.3e5 (year ~2000), currently ~7.4e5 to ~7.5e5 in the 2020s

        if isinstance(time_input, str):
            return "datestring"
        elif 1e9 < time_input < 2e9:
            return "epoch"
        elif 7e5 < time_input < 8.5e5:
            return "matlab"
        else:
            raise IOError(f"Unrecognized time input {time_input} with type {type(time_input)}")

    def load_burst(self, burst_idx: int) -> Dict[str, np.ndarray]:
        """Load data for a single burst.

        Parameters
        ----------
        burst_idx : int
            Index of burst to load

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing burst data
        """
        if burst_idx >= self.n_bursts:
            raise IndexError(f"Burst index {burst_idx} out of range")

        if self._cached_idx == burst_idx:
            return self._cached_data

        if self._monolithic_ds is not None:
            data = self._monolithic_ds.isel({self.burst_dim: burst_idx})
            file_type = "nc"
        else:
            file_path = self.files[burst_idx]
            try:
                data, file_type = self._load_file(file_path, self.data_keys)
            except Exception as e:
                raise IOError(f"Failed to load {file_path}: {e}")

        # Extract and organize data
        burst_data = {}
        for out_key, in_key in self.name_map.items():
            if isinstance(in_key, list):
                # Multiple variables (e.g., from different instruments)
                var_data = np.array([self._as_array(data, k, file_type) for k in in_key])
            else:
                # Single variable
                var_data = self._as_array(data, in_key, file_type)
                if var_data.ndim > 1:
                    # Transpose if needed (time should be last dimension)
                    if var_data.shape[0] > var_data.shape[1]:
                        var_data = var_data.T
                else:
                    var_data = np.expand_dims(var_data, axis=0)  # 2D even if only 1D input

            # Enforcing byte order in case there is a mismatch
            var_data = var_data.astype(var_data.dtype.newbyteorder("="))

            burst_data[out_key] = var_data

        burst_data_out = self._apply_preprocessing(burst_data)

        self._cached_idx = burst_idx
        self._cached_data = burst_data_out

        return burst_data_out

    def _apply_preprocessing(self, burst_data):
        """Override in subclasses to add preprocessing steps."""
        return burst_data

    def subsample(self, start_idx: int, end_idx: int):
        """Subsample the instrument file list from start_idx:end_idx.

        Must be implemented in derived classes to account for unique
        initialization calls.
        """
        raise NotImplementedError("Subclasses must implement subsample()")

    @property
    def n_bursts(self):
        if self._monolithic_ds is not None:
            return int(self._monolithic_ds.sizes[self.burst_dim])
        return len(self.files)

    @property
    def n_heights(self):
        return len(self.z)

    def to_dataset(
        self,
        results: List[Dict[str, Any]],
        burst_times: np.ndarray,
        freq: Optional[np.ndarray] = None,
        attrs: Optional[dict] = None,
    ) -> xr.Dataset:
        """Concatenate per-burst result dictionaries into an xarray Dataset.

        Dimensions are inferred from result-value shapes against `self.z` and the
        optional `freq` coordinate; see `utils.io_utils.results_to_dataset` for the
        shape-to-dim mapping. Global attributes are augmented with the instrument
        class name and sampling rate.

        Parameters
        ----------
        results : list of dict
            Per-burst result dictionaries. Keys missing from a burst fill with NaN.
        burst_times : np.ndarray
            1D array of representative timestamps for each burst, length `len(results)`.
        freq : np.ndarray, optional
            Frequency coordinate for spectral outputs.
        attrs : dict, optional
            Additional global attributes. Merged over the auto-populated attrs.

        Returns
        -------
        xr.Dataset
        """
        merged_attrs = {
            "instrument": self.__class__.__name__,
            "fs": float(self.fs) if self.fs is not None else None,
        }
        if attrs:
            merged_attrs.update(attrs)
        return results_to_dataset(
            results=results,
            burst_times=burst_times,
            z=self.z,
            freq=freq,
            attrs=merged_attrs,
        )

    def to_netcdf(
        self,
        path: str,
        results: List[Dict[str, Any]],
        burst_times: np.ndarray,
        freq: Optional[np.ndarray] = None,
        attrs: Optional[dict] = None,
        **nc_kwargs,
    ) -> None:
        """Build a Dataset from per-burst results and write it to a NetCDF
        file.

        Parameters
        ----------
        path : str
            Output NetCDF path.
        results, burst_times, freq, attrs
            Forwarded to `to_dataset`.
        **nc_kwargs
            Forwarded to `xr.Dataset.to_netcdf`.
        """
        ds = self.to_dataset(results, burst_times, freq=freq, attrs=attrs)
        ds.to_netcdf(path, **nc_kwargs)


# Helper function
def strip_mat_nulls(obj):
    if isinstance(obj, dict):
        return {k.rstrip("\x00"): strip_mat_nulls(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(strip_mat_nulls(i) for i in obj)
    return obj
