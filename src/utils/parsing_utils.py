import numpy as np
from typing import List, Union, Optional
import pandas as pd
import scipy.io as sio
import xarray as xr


class DatasetParser:
    def __init__(
        self,
        files: Union[List[str], str],
        name_map: dict,
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[float, List[float]]] = None,
    ):
        self.files = files
        self.name_map = name_map
        self.fs = fs
        self.z = z

    def parse_input(self):
        if isinstance(self.files, list):
            suffix = self.files[0].split(".")[-1].lower()
            if suffix == "mat":
                data_vars, coords, attrs = self.parse_mat_list()
            elif suffix == "npy":
                data_vars, coords, attrs = self.parse_npy_list()
            elif suffix == "csv":
                data_vars, coords, attrs = self.parse_csv_list()
            else:
                raise Exception(f"Unrecognized file type .{suffix} for filepath input")
        elif isinstance(self.files, str):
            if ".mat" in self.files:
                data_vars, coords, attrs = self.parse_mat()
            elif ".nc" in self.files:
                data_vars, coords, attrs = self.parse_netcdf()
            else:
                try:
                    data_vars, coords, attrs = self.parse_zarr()
                except Exception as e:  # TODO: Make this more specific
                    raise (f"{e}, not a zarr directory")
                except Exception as e:  # General case
                    raise (f"{e}, unrecognized file type in filepath")
        else:
            raise IOError(f"Unrecognized type {type(self.files)} for filepath input")

        return xr.Dataset(data_vars, coords, attrs)

    def parse_mat_list(self):
        variables = {key: [] for key in self.name_map.keys()}
        num_bad_files = 0
        for ii, file in enumerate(self.files):
            try:
                data = sio.loadmat(file, simplify_cells=True)
            except OSError as e:
                print(f"{e} for file {file}, skipping")
                num_bad_files += 1
                continue
            for out_key, in_key in self.name_map.items():
                # If each variable has multiple subvariables (e.g., from different instruments)
                if isinstance(in_key, list):
                    data_var = np.zeros((len(in_key), *data[in_key[0]].shape))
                    for sub_index, sub_key in enumerate(in_key):
                        data_var[sub_index, :] = data[in_key[sub_index]]
                    variables[out_key].append(data_var)

                # Single variable for each quantity, either 1d or 2d
                else:
                    data_var = data[in_key]
                    if data_var.ndim > 1:
                        num_rows, num_cols = data_var.shape
                        if num_rows > num_cols:
                            data_var = data_var.T

                    variables[out_key].append(data_var)

        # Define coordinates
        burst_coords = np.arange(len(self.files) - num_bad_files)
        if not self.z:
            non_time_key = [key for key in variables.keys() if key != "time"][0]
            num_heights = variables[non_time_key][0].shape[0]
            self.z = np.arange(num_heights)

        height_coords = self.z

        # Stack bursts into a 3D array (burst, height, time)
        data_vars = {
            key: (["burst", "height", "time"], np.stack(variables[key], axis=0))
            for key in variables.keys()
            if key != "time"
        }

        # Stack time arrays into a 2D array (burst, time)
        time_array = np.stack(variables["time"], axis=0)  # shape: (burst, time)
        datetime_array = self.process_time(time_array)

        coords = {"burst": burst_coords, "height": height_coords, "time": (["burst", "time"], datetime_array)}
        attrs = {"fs": self.fs}

        return data_vars, coords, attrs

    def parse_csv_list(self):
        pass

    def parse_npy_list(self):
        pass

    def parse_mat(self):
        pass

    def parse_zarr(self):
        pass

    def parse_netcdf(self):
        pass

    def process_time(self, time_array: np.ndarray) -> (np.ndarray, float):
        # Test on the first element
        flattened_time = time_array.flatten()
        format = self.detect_time_format(flattened_time[0])
        if format == "datestring":
            datetime_array = pd.to_datetime(flattened_time).values
        elif format == "matlab":
            datetime_array = pd.to_datetime(flattened_time - 719529, unit="D").values
        elif format == "epoch":
            datetime_array = pd.to_datetime(flattened_time, unit="s").values

        if not self.fs:
            # Assume it's an integer
            self.fs = np.round(1 / ((datetime_array[1] - datetime_array[0]).astype(int) / 10**9))

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
