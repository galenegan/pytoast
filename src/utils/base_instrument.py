from abc import ABC
import numpy as np
import os
from typing import List, Union, Optional, Dict
import scipy.io as sio
import pandas as pd
import xarray as xr


class BaseInstrument(ABC):
    """Abstract base class containing data loading and parsing methods that are used across instruments"""

    def __init__(
        self,
        files: Union[str, List[str]],
        name_map: dict,
        fs: Optional[float] = None,
        z: Optional[Union[float, List[float]]] = None,
    ):
        """
        Initialize data manager.

        Parameters
        ----------
        files : str or List[str]
            Path(s) to data files
        name_map : dict
            Mapping of variable names
        fs : float, optional
            Sampling frequency
        z : float or List[float], optional
            Height coordinates
        """
        files = files if isinstance(files, list) else [files]
        self.validate_common_inputs(files, name_map, fs, z)
        self.files = files
        self.name_map = name_map
        self.fs, self.z, self.file_type, self.num_samples_per_burst = self._inspect_first_file(fs, z)
        self._cached_idx = None
        self._cached_data = None
        self._preprocess_enabled = False

    @property
    def n_bursts(self):
        return len(self.files)

    @property
    def n_heights(self):
        return len(self.z)

    @staticmethod
    def validate_common_inputs(
        files: List[str],
        name_map: dict,
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[float, int, List[Union[float, int]]]] = None,
    ):
        """
        Validate common input parameters shared across all instruments.

        Parameters
        ----------
        files : List[str]
            Input files
        name_map : dict
            Variable name mapping
        fs : float, optional
            Sampling frequency
        z : float or List[float], optional
            Height coordinates

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

        # Validate "z"
        if z is not None:
            if not isinstance(z, (float, int, list, np.ndarray)):
                raise TypeError("`z` must be either a float, int, list, or numpy array")
            if isinstance(z, list) and not all(isinstance(zi, (float, int)) for zi in z):
                raise TypeError("All elements of the `z` list must be floats or ints")

        # Validate "fs"
        if fs is not None and not isinstance(fs, (int, float)):
            raise TypeError("`fs` must be either an int or a float")

    @staticmethod
    def _load_file(file_path):
        suffix = file_path.split(".")[-1].lower()
        if suffix == "mat":
            data = sio.loadmat(file_path, simplify_cells=True)
            file_type = "mat"
        elif suffix == "npy":
            data = np.load(file_path, allow_pickle=True).item()
            file_type = "npy"
        elif suffix == "csv":
            data = pd.read_csv(file_path)
            file_type = "csv"
        elif suffix == "nc":
            data = xr.load_dataarray(file_path)
            file_type = "nc"
        else:
            raise Exception(f"Unrecognized file type .{suffix} for filepath input")

        return data, file_type

    def _inspect_first_file(self, fs, z):
        """
        Read the first file to infer fs and z (if not provided) and determine
        file_type and num_samples_per_burst.

        Parameters
        ----------
        fs : float or None
            Sampling frequency, or None to infer from time variable
        z : float, list, np.ndarray, or None
            Height coordinates, or None to infer from data dimensions

        Returns
        -------
        fs : float
            Sampling frequency (provided or inferred)
        z : np.ndarray
            Height coordinates as a numpy array (provided or inferred)
        file_type : str
            File format identifier ("mat", "npy", "csv", "nc")
        num_samples_per_burst : int
            Number of samples per burst
        """
        if not self.files:
            raise ValueError("No files provided")

        data, file_type = self._load_file(self.files[0])

        # Normalize z to a numpy array, or infer from data dimensions
        # TODO: Test this to make sure it's generalizable to xarray DA, numpy array, and pandas df
        if z is not None:
            if isinstance(z, (int, float)):
                z = np.array([z])
            elif isinstance(z, list):
                z = np.array(z)
        elif "z" in self.name_map:
            key = self.name_map["z"]
            if isinstance(data[key], (int, float)):
                z = np.array([data[key]])
            elif isinstance(data[key], list):
                z = np.array(data[key])
            elif isinstance(data[key], np.ndarray):
                z = data[key]
        else:
            non_time_key = [key for key in self.name_map.keys() if key != "time"][0]
            if isinstance(non_time_key, str):
                data_var = data[non_time_key]
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
            data_var = data[list(self.name_map.keys())[0]]
            num_rows, num_cols = data_var.shape
            num_samples = max(num_rows, num_cols)
        else:
            num_samples = len(data[self.name_map["time"]])
            if fs is None:
                time_array = data[self.name_map["time"]]
                datetime_array = self.process_time(time_array)
                fs = np.round(1 / ((datetime_array[1] - datetime_array[0]).astype(int) / 10**9), 2)

        return fs, z, file_type, num_samples

    def process_time(self, time_array: np.ndarray) -> np.ndarray:
        """
        Convert a time array to numpy datetime64 format.

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
        """
        Detect if a time input represents Unix epoch time, MATLAB datenum, or a datestring

        Args:
            time_input (float): The input float to test.

        Returns:
            str: "datestring", "epoch", "matlab". Raises an exception if there is no match
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
        """
        Load data for a single burst.

        Parameters
        ----------
        burst_idx : int
            Index of burst to load

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing burst data
        """
        if burst_idx >= len(self.files):
            raise IndexError(f"Burst index {burst_idx} out of range")

        if self._cached_idx == burst_idx:
            return self._cached_data

        file_path = self.files[burst_idx]
        try:
            data, _ = self._load_file(file_path)
        except Exception as e:
            raise IOError(f"Failed to load {file_path}: {e}")

        # Extract and organize data
        burst_data = {}
        for out_key, in_key in self.name_map.items():
            if isinstance(in_key, list):
                # Multiple variables (e.g., from different instruments)
                var_data = np.array([data[k] for k in in_key])
            else:
                # Single variable
                var_data = data[in_key]
                if var_data.ndim > 1:
                    # Transpose if needed (time should be last dimension)
                    if var_data.shape[0] > var_data.shape[1]:
                        var_data = var_data.T
                else:
                    var_data = np.expand_dims(var_data, axis=0)  # 2D even if only one height

            burst_data[out_key] = var_data

        if self._preprocess_enabled:
            burst_data_out = self._apply_preprocessing(burst_data)
        else:
            burst_data_out = burst_data

        self._cached_idx = burst_idx
        self._cached_data = burst_data_out

        return burst_data_out

    def _apply_preprocessing(self, burst_data):
        """Override in subclasses to add preprocessing steps."""
        return burst_data

    def load_burst_range(self, start_idx: int, end_idx: int) -> Dict[str, np.ndarray]:
        """
        Load data for a range of bursts.

        Parameters
        ----------
        start_idx : int
            Starting burst index
        end_idx : int
            Ending burst index (exclusive)

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing burst data with shape (n_bursts, n_heights, n_samples)
        """
        burst_data_list = []
        for i in range(start_idx, end_idx):
            burst_data_list.append(self.load_burst(i))

        # Stack bursts
        stacked_data = {}
        for key in burst_data_list[0].keys():
            stacked_data[key] = np.stack([bd[key] for bd in burst_data_list], axis=0)

        return stacked_data

    def subsample(self, start_idx: int, end_idx: int):
        files = self.files[start_idx:end_idx]
        return self.__class__(files, self.name_map, self.fs, self.z)
