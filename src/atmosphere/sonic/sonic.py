import numpy as np
from typing import Optional, Union, List
from numpy.typing import ArrayLike
from xarray import Dataset


class Sonic(Dataset):
    def __init__(self, filepath: Union[str, List[str]], name_map: dict, fs: Optional[Union[int, float]] = None):
        """
        Initializes a Sonic object, storing 3D velocity and optionally sonic temperature
        arrays in an xarray Dataset
        :param filepath: if str, must be a path to a netCDF file, .mat file, or zarr file store that contains the
        entire dataset you wish to load. If list, the elements of the list will be interpreted as files containing data
        from individual measurement burst periods. Supported burst file types are .npy (assuming it was saved as a
        dictionary), .mat (assuming it was saved as a Matlab Struct) and .csv (with variables in separate columns)

        :param name_map: a dictionary of the form:
        {
            "u": "x-velocity variable name",
            "v": "y-velocity variable name",
            "w": "z-velocity variable name",
            "ts": "sonic temperature variable name",
            "time": "time variable name",
        }
        Of these, "ts" and "time" are optional, but an error will be raised if "time" is not specified and a sampling
        frequency is also not specified.
        :param fs: sampling frequency (Hz), optional unless name_map["time"] is not specified
        """
        self.filepath = filepath
        self.name_map = name_map
        data_vars, coords, attrs = self.parse_input()
        super.__init__(data_vars, coords, attrs)

    def parse_input(self):
        if isinstance(self.filepath, List):
            suffix = self.filepath[0].split(".")[-1].lower()
            if suffix == "mat":
                data_vars, coords, attrs = self.parse_mat_list()
            elif suffix == "npy":
                data_vars, coords, attrs = self.parse_npy_list()
            elif suffix == "csv":
                data_vars, coords, attrs = self.parse_csv_list()
            else:
                raise Exception(f"Unrecognized file type .{suffix} for filepath input")
        elif isinstance(self.filepath, str):
            if ".mat" in self.filepath:
                data_vars, coords, attrs = self.parse_mat()
            elif ".nc" in self.filepath:
                data_vars, coords, attrs = self.parse_netcdf()
            else:
                try:
                    data_vars, coords, attrs = self.parse_zarr()
                except Exception as e:  # TODO: Make this more specific
                    raise (f"{e}, not a zarr directory")
                except Exception as e:  # General case
                    raise (f"{e}, unrecognized file type in filepath")
        else:
            raise IOError(f"Unrecognized type {type(self.filepath)} for filepath input")

        return data_vars, coords, attrs

    def parse_mat_list(self):
        pass

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

    # # Graveyard
    # """
    #         Initializes the Sonic object, storing 3D velocity and optionally sonic temperature arrays
    #         :param u: x-component of horizontal velocity (m/s)
    #         :param v: y-component of horizontal velocity (m/s)
    #         :param w: vertical velocity (m/s)
    #         :param Ts: sonic temperature (degrees C), optional
    #         :param time: time array, optional unless sampling frequency fs is not specified. The input will be coerced
    #         into a Pandas datetime series with a rough check to determine whether the input times are strings, epoch time,
    #         or Matlab time.
    #         :param fs: sampling frequency (Hz), optional unless time array is not specified.
    #         """
    # self.u = u
    # self.v = v
    # self.w = w
    # if Ts:
    #     self.Ts = Ts
    #
    #
    def detect_time_format(time_input: Union[float, int, str]) -> str:
        """
        Detect if a float represents Unix epoch time or MATLAB datenum.

        Args:
            time_input (float): The input float to test.

        Returns:
            str: 'epoch', 'matlab', or 'unknown'
        """

        # Rough numeric ranges as of 2020s:
        # Epoch: ~1.5e9 (1970-2020s)
        # MATLAB: ~7.3e5 (year ~2000), currently ~7.4e5 to ~7.5e5 in the 2020s

        if 1e9 < time_input < 2e9:
            return "epoch"
        elif 7e5 < time_input < 8.5e5:
            return "matlab"
        else:
            return "unknown"
