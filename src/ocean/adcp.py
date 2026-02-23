import numpy as np
from typing import Optional, Union, List, Dict, Any
from utils.base_instrument import BaseInstrument

from utils.rotate_utils import (
    align_with_principal_axis,
    align_with_flow,
    rotate_velocity_by_theta,
)


class ADCP(BaseInstrument):
    """
    ADCP class.
    """

    def __init__(
        self,
        files: Union[str, List],
        name_map: dict,
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[List[Union[float, int]], np.ndarray]] = None,
        coords: str = "beam",
        orientation: str = "up",
        manufacturer: str = "nortek",
    ):
        self.coords = coords
        self.orientation = orientation
        self.manufacturer = manufacturer
        super().__init__(files, name_map, fs, z)

    @staticmethod
    def validate_inputs(
        files: Union[str, List],
        name_map: dict,
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[List[Union[float, int]], np.ndarray]] = None,
        coords: str = "beam",
        orientation: str = "up",
        manufacturer: str = "nortek",
    ):

        # General validation
        BaseInstrument.validate_common_inputs(files, name_map, fs, z)

        # Instrument-specific requirements
        required_keys = ["u1", "u2", "u3"]

        for key in required_keys:
            if key not in name_map:
                raise ValueError(f"`name_map` must include a mapping for '{key}'")

        if coords not in ["beam", "xyz", "enu"]:
            raise ValueError("`coords` must be either 'beam', 'xyz', or 'enu'")

        if orientation not in ["up", "down"]:
            raise ValueError("`orientation` must be either 'up' or 'down'")

        if manufacturer not in ["nortek", "rdi"]:
            raise ValueError(
                "`manufacturer` must be either 'nortek' or 'rdi'. This is only used for "
                "beam/xyz/enu coordinate transformations, so there is no need to specify if your data are "
                "are already in the desired coordinates"
            )

    @classmethod
    def from_raw(
        cls,
        files: Union[str, List],
        name_map: dict,
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[float, int, List[Union[float, int]]]] = None,
        coords: str = "beam",
        orientation: str = "up",
        manufacturer: str = "nortek",
    ):
        """
        Initializes a new ADV object from data files.

        Parameters
        ----------
        files : str or List[str]
            If str, must be a path to a netCDF file, .mat file, or zarr file store that contains the
            entire dataset you wish to load. If list, the elements of the list will be interpreted as files containing data
            from individual measurement burst periods. Supported burst file types are .npy (assuming it was saved as a
            dictionary), .mat (assuming it was saved as a Matlab Struct) and .csv (with variables in separate columns).
            If the variables associated with a particular name are two-dimensional, then the larger dimension is
            assumed to be time and the shorter dimension is assumed to be a vertical coordinate. In this case,
            a "z" list must be passed as an argument with a length matching the size of the shorter dimension

        name_map : dict
            a dictionary of the form:
            {
                "u1": "variable name for first beam/direction velocity",
                "u2": "variable name for second beam/direction velocity",
                "u3": "variable name for third beam/direction velocity",
                "u4": "variable name for fourth beam/direction velocity",
                "u5": "variable name for fifth beam/direction velocity",
                "z": "height variable name",
                "p": "pressure variable name" ,
                "time": "time variable name",
            }
            Of these, "z", "p", "time", "u4" and "u5" are optional. However, an error will be raised if "time" is not
            specified and a sampling frequency is also not specified, and "z" will only be used if the class-level
            ADCP.z argument is not specified.

        z : List[float, int] or np.ndarray, optional
            vertical coordinate for each cell. This will be interpreted as either meters above the bed
            if orientation == "up", or meters below the surface if orientation == "down".
            If not specified, the height coordinate in the resulting ADCP object will be integer indices.

        fs : int or float, optional
            sampling frequency (Hz). If not specified, will be inferred (and rounded to an integer)
            from name_map["time"] values

        orientation : str, optional
            Instrument orientation. Can be "up" or "down"; affects the interpretation of the vertical coordinate


        Returns
        -------
        ADCP object

        """
        ADCP.validate_inputs(files, name_map, fs, z, orientation)
        return cls(files, name_map, fs, z, orientation)

    def set_preprocess_opts(self, opts: Dict[str, Any]):
        """Enable preprocessing for all subsequent burst loads using the options defined in the input dictionary.

        Parameters
        ----------
        opts : dict
            Preprocessing options. Supported keys:

            despike : dict, optional
                Options for simple threshold-based despiking. If not specified, no despiking is applied.
                Supported keys:

                threshold_min : float

                threshold_max : float

            rotate : dict, optional
                Options for rotations and coordinate transformations. If not specified, no rotations applied.
                Supported keys:

                coords_out : str, optional
                    Coordinates for ADCP.coords to be transformed to. One of ["beam", "xyz", "enu"].

                beam_angle : float, optional
                    Beam angle in degrees. Affects accuracy of coordinate transform for RDI instruments

                declination : float, optional
                    Magnetic declination in degrees. Added to heading for coordinate transformations.

                flow_rotation : str or Tuple[float], optional.
                    One of ["align_principal", "align_current", or (horizontal_angle, vertical_angle)].
                    If "align_principal" then the velocity will be rotated to align with the principal axis of the flow.
                    If "align_current" then the velocity will be rotated to align with the horizontal current magnitude
                    sqrt(u1^2 + u2^2). In both cases, the vertical velocity will be minimized. If float angles are
                    specified in a tuple, the flow will be rotated by those angles in the horizontal and vertical
                    planes. Specifying any option will throw an error if ADCP.coords == "beam", unless a
                    coordinate system change to "xyz" or "enu" is also requested.
        """

        self._preprocess_enabled = True

        self._despike = opts.get("despike", {})
        self._rotate = opts.get("rotate", {})
        self._cached_idx = None
        self._cached_data = None

    def _apply_preprocessing(self, burst_data):
        if self._despike:
            burst_data = self._apply_despike(burst_data, **self._despike_opts)

        if isinstance(self._rotate, str):
            if self._rotate == "align_principal":
                theta_h, theta_v = align_with_principal_axis(burst_data)
            elif self._rotate == "align_current":
                theta_h, theta_v = align_with_flow(burst_data)
            else:
                raise ValueError(f"Invalid rotation option '{self._rotate}'")
        else:
            theta_h, theta_v = self._rotate

        if np.sum(np.abs(theta_v)) != 0.0 or np.sum(np.abs(theta_h)) != 0.0:
            burst_data = rotate_velocity_by_theta(burst_data, theta_h, theta_v)

        return burst_data

    def shear(self, burst_data):
        pass

    def covariance(self, burst_data, method="variance"):
        pass

    def dissipation(self, burst_data, method="spectral"):
        pass

    def tke(self, burst_data):
        pass
