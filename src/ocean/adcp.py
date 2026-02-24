import numpy as np
from typing import Optional, Union, List, Dict, Any
from utils.base_instrument import BaseInstrument
from utils.interp_utils import interp_rows

from utils.rotate_utils import (
    align_with_principal_axis,
    align_with_flow,
    rotate_velocity_by_theta,
    coord_transform_3_beam_nortek,
    coord_transform_4_beam_nortek,
    coord_transform_4_beam_rdi
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
        source_coords: str = "beam",
        orientation: str = "up",
        manufacturer: str = "nortek",
    ):
        self.source_coords = source_coords
        self.orientation = orientation
        self.manufacturer = manufacturer
        super().__init__(files, name_map, fs, z)

    @staticmethod
    def validate_inputs(
        files: Union[str, List],
        name_map: dict,
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[List[Union[float, int]], np.ndarray]] = None,
        source_coords: str = "beam",
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

        if source_coords not in ["beam", "xyz", "enu"]:
            raise ValueError("`source_coords` must be either 'beam', 'xyz', or 'enu'")

        if orientation not in ["up", "down"]:
            raise ValueError("`orientation` must be either 'up' or 'down'")

        if manufacturer not in ["nortek", "rdi"]:
            raise ValueError(
                "`manufacturer` must be either 'nortek' or 'rdi'. This is only used for "
                "beam/xyz/enu coordinate transformations, so there is no need to specify if your data are "
                "are already in the desired coordinates"
            )

    @classmethod
    def from_files(
        cls,
        files: Union[str, List],
        name_map: dict,
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[float, int, List[Union[float, int]]]] = None,
        source_coords: str = "beam",
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
                "heading": "heading variable name",
                "pitch": "pitch variable name",
                "roll": "roll variable name",
                "z": "height variable name",
                "p": "pressure variable name" ,
                "time": "time variable name",
            }
            Of these, "heading", "pitch", "roll", "z", "p", "time", "u4" and "u5" are optional. However, an error will
            be raised if "time" is not specified and a sampling frequency is also not specified, and "z" will only be
            used if the class-level ADCP.z argument is not specified.

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

                transformation_matrix : np.ndarray, optional
                    Transformation matrix for the instrument. Must be specified for coordinate transformation if
                    manufacturer = 'nortek'. May be excluded if manufacturer = 'rdi' if beam angle is specified

                beam_angle : float, optional
                    Beam angle in degrees. If manufacturer = 'rdi', may be specified instead of transformation_matrix
                    for coordinate transformations.

                declination : float, optional
                    Magnetic declination in degrees. Added to heading for coordinate transformations.

                constant_hpr : Tuple[float], optional
                    Constant heading, pitch, and roll angles to apply.

                flow_rotation : str or Tuple[float], optional.
                    One of ["align_principal", "align_current", or (horizontal_angle, vertical_angle)].
                    If "align_principal" then the velocity will be rotated to align with the principal axes of the flow.
                    If "align_current" then the velocity will be rotated to align with the horizontal current magnitude
                    sqrt(u^2 + v^2). In both cases, the vertical velocity will be minimized. If float angles are
                    specified in a tuple, the flow will be rotated by those angles in the horizontal and vertical
                    planes. Specifying any option will throw an error if ADCP.coords == "beam", unless a
                    coordinate system change to "xyz" or "enu" is also requested.
        """

        self._preprocess_opts = opts
        self._preprocess_enabled = True

        self._despike = opts.get("despike", {})
        if self._despike:
            self._despike_opts = {key: val for key, val in self._despike.items() if key != "method"}

        self._rotate = opts.get("rotate", {})
        self._cached_idx = None
        self._cached_data = None

    def _apply_preprocessing(self, burst_data):
        # Setting coords regardless of preprocessing options
        coords_in = self.source_coords
        burst_data["coords"] = coords_in
        if not self._preprocess_enabled:
            return burst_data

        if self._despike:
            for beam_key in self.beam_keys:
                burst_data[beam_key] = self._apply_threshold_despike(burst_data[beam_key], **self._despike_opts)

        if self._rotate:
            n_heights = self.n_heights
            coords_out = self._rotate.get("coords_out")

            if coords_out:
                transformation_matrix = self._rotate.get("transformation_matrix")
                beam_angle = self._rotate.get("beam_angle")
                if transformation_matrix is None and self.manufacturer == "nortek":
                    raise ValueError("A transformation matrix must be provided for nortek coordinate transformations")

                if transformation_matrix is None and beam_angle is None and self.manufacturer == "rdi":
                    raise ValueError(
                        "Either a beam angle or transformation matrix must be provided for rdi coordinate "
                        "transformations"
                    )

                heading = burst_data.get("heading")
                pitch = burst_data.get("pitch")
                roll = burst_data.get("roll")

                if ((coords_in == "enu") or (coords_out == "enu")) and (
                    (heading is None) or (pitch is None) or (roll is None)
                ):
                    constant_hpr = self._rotate.get("constant_hpr")

                    if constant_hpr:
                        heading, pitch, roll = constant_hpr
                    else:
                        raise ValueError(
                            "Heading, pitch, and roll must be provided for any coordinate transformation to/from ENU"
                        )

                for height_idx in range(n_heights):

                    hi = heading[height_idx, :] if heading is not None else None
                    pi = pitch[height_idx, :] if pitch is not None else None
                    ri = roll[height_idx, :] if roll is not None else None

                    if self.manufacturer == "nortek" and self.num_beams == 3:
                        u1_new, u2_new, u3_new = coord_transform_3_beam_nortek(
                            u1=burst_data["u1"][height_idx, :],
                            u2=burst_data["u2"][height_idx, :],
                            u3=burst_data["u3"][height_idx, :],
                            heading=hi,
                            pitch=pi,
                            roll=ri,
                            transformation_matrix=transformation_matrix,
                            declination=self._rotate.get("declination", 0.0),
                            orientation=self.orientation,
                            coords_in=coords_in,
                            coords_out=coords_out,
                        )
                        burst_data["u1"][height_idx, :] = u1_new
                        burst_data["u2"][height_idx, :] = u2_new
                        burst_data["u3"][height_idx, :] = u3_new
                    elif self.manufacturer == "nortek" and self.num_beams > 3:
                        u1_new, u2_new, u3_new, u4_new = coord_transform_4_beam_nortek(
                            burst_data["u1"][height_idx, :],
                            burst_data["u2"][height_idx, :],
                            burst_data["u3"][height_idx, :],
                            burst_data["u4"][height_idx, :],
                            heading=hi,
                            pitch=pi,
                            roll=ri,
                            transformation_matrix=transformation_matrix,
                            declination=self._rotate.get("declination", 0.0),
                            orientation=self.orientation,
                            coords_in=coords_in,
                            coords_out=coords_out
                        )
                        burst_data["u1"][height_idx, :] = u1_new
                        burst_data["u2"][height_idx, :] = u2_new
                        burst_data["u3"][height_idx, :] = u3_new
                        burst_data["u4"][height_idx, :] = u4_new
                    elif self.manufacturer == "rdi" and self.num_beams > 3:
                        u1_new, u2_new, u3_new, u4_new = coord_transform_4_beam_rdi(
                            burst_data["u1"][height_idx, :],
                            burst_data["u2"][height_idx, :],
                            burst_data["u3"][height_idx, :],
                            burst_data["u4"][height_idx, :],
                            heading=hi,
                            pitch=pi,
                            roll=ri,
                            beam_angle=beam_angle,
                            transformation_matrix=transformation_matrix,
                            declination=self._rotate.get("declination", 0.0),
                            orientation=self.orientation,
                            coords_in=coords_in,
                            coords_out=coords_out,
                        )
                        burst_data["u1"][height_idx, :] = u1_new
                        burst_data["u2"][height_idx, :] = u2_new
                        burst_data["u3"][height_idx, :] = u3_new
                        burst_data["u4"][height_idx, :] = u4_new
                    else:
                        raise ValueError("Invalid combination of manufacturer and number of beams for coordinate transformation")
                burst_data["coords"] = coords_out

            flow_rotation = self._rotate.get("flow_rotation")

            if flow_rotation and burst_data["coords"] == "beam":
                raise ValueError(
                    "Cannot rotate flow velocity with ADCP.coords == 'beam'. Specify either 'xyz' or 'enu'"
                    " as 'coords_out' in the rotate options dictionary."
                )
            elif flow_rotation and burst_data["coords"] != "beam":
                if isinstance(flow_rotation, str):
                    if flow_rotation == "align_principal":
                        theta_h, theta_v = align_with_principal_axis(
                            burst_data["u1"], burst_data["u2"], burst_data["u3"]
                        )
                    elif flow_rotation == "align_current":
                        theta_h, theta_v = align_with_flow(burst_data["u1"], burst_data["u2"], burst_data["u3"])
                    else:
                        raise ValueError(f"Invalid rotation option '{flow_rotation}'")
                elif isinstance(flow_rotation, tuple):
                    theta_h, theta_v = flow_rotation

                u1_new, u2_new, u3_new = rotate_velocity_by_theta(
                    burst_data["u1"], burst_data["u2"], burst_data["u3"], theta_h, theta_v
                )
                burst_data["u1"] = u1_new
                burst_data["u2"] = u2_new
                burst_data["u3"] = u3_new
                burst_data["rotation"] = flow_rotation

        return burst_data

    def _apply_threshold_despike(
        self,
        u: np.ndarray,
        threshold_min: float = -3.0,
        threshold_max: float = 3.0,
    ):
        u_out = u.copy()
        bad_rows = (u_out < threshold_min) | (u_out > threshold_max)
        u_out[bad_rows] = np.nan
        interp_rows(u_out)
        return u_out

    def shear(self, burst_data):
        pass

    def covariance(self, burst_data, method="variance"):
        pass

    def dissipation(self, burst_data, method="spectral"):
        pass

    def tke(self, burst_data):
        pass

    @property
    def beam_keys(self):
        return [k for k in ["u1", "u2", "u3", "u4", "u5"] if k in self.name_map]

    @property
    def num_beams(self):
        return len(self.beam_keys)