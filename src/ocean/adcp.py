import copy
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import curve_fit
from scipy.stats import circmean, linregress
from typing import Optional, Union, List, Dict, Any
from utils.base_instrument import BaseInstrument
from utils.interp_utils import interp_rows
from utils.spectral_utils import psd

from utils.rotate_utils import (
    align_with_principal_axis,
    align_with_flow,
    rotate_velocity_by_theta,
    coord_transform_3_beam_nortek,
    coord_transform_4_beam_nortek,
    coord_transform_4_beam_rdi,
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
        data_keys: Optional[Union[str, List[str]]] = None,
        source_coords: str = "beam",
        orientation: str = "up",
        beam_angle: float = 25.0,
        manufacturer: str = "nortek",
    ):
        self.source_coords = source_coords
        self.orientation = orientation
        self.beam_angle = beam_angle
        self.manufacturer = manufacturer
        super().__init__(files, name_map, fs, z, data_keys)

    @staticmethod
    def validate_inputs(
        files: Union[str, List],
        name_map: dict,
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[List[Union[float, int]], np.ndarray]] = None,
        data_keys: Optional[Union[str, List[str]]] = None,
        source_coords: str = "beam",
        orientation: str = "up",
        beam_angle: float = 25.0,
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

        if not isinstance(beam_angle, (int, float)):
            raise ValueError("`beam_angle` must be a number")

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
        data_keys: Optional[Union[str, List[str]]] = None,
        source_coords: str = "beam",
        orientation: str = "up",
        beam_angle: float = 25.0,
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
        ADCP.validate_inputs(
            files=files,
            name_map=name_map,
            fs=fs,
            z=z,
            data_keys=data_keys,
            source_coords=source_coords,
            orientation=orientation,
            beam_angle=beam_angle,
            manufacturer=manufacturer
        )
        return cls(files, name_map, fs, z, data_keys, source_coords, orientation, beam_angle, manufacturer)

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
        burst_data["coords"] = self.source_coords
        if not self._preprocess_enabled:
            return burst_data

        if self._despike:
            for key in self.beam_keys:
                burst_data[key] = self._apply_threshold_despike(burst_data[key], **self._despike_opts)

        if self._rotate:
            coords_out = self._rotate.get("coords_out")
            if coords_out:
                burst_data = self._apply_coord_transform(burst_data, coords_out)

            flow_rotation = self._rotate.get("flow_rotation")
            if flow_rotation:
                if burst_data["coords"] == "beam":
                    raise ValueError(
                        "Cannot apply flow rotation in beam coordinates. Specify 'coords_out' "
                        "as 'xyz' or 'enu' in rotate options."
                    )
                burst_data = self._apply_flow_rotation(burst_data, flow_rotation)

        return burst_data

    def _apply_coord_transform(self, burst_data, coords_out):
        """
        Transform velocity components between coordinate systems.

        Uses configuration stored in self._rotate (transformation_matrix, declination,
        constant_hpr). Can be called from _apply_preprocessing during standard burst
        loading, or directly from analysis methods (e.g. covariance) when on-the-fly
        transformation is needed.

        Parameters
        ----------
        burst_data : dict
            Burst data dictionary. burst_data["coords"] must reflect the current
            coordinate system of u1/u2/u3.
        coords_out : str
            Target coordinate system. One of ["beam", "xyz", "enu"].

        Returns
        -------
        dict
            burst_data with velocity components transformed in-place and
            burst_data["coords"] updated to coords_out.
        """
        coords_in = burst_data["coords"]
        transformation_matrix = self._rotate.get("transformation_matrix")
        declination = self._rotate.get("declination", 0.0)

        if transformation_matrix is None and self.manufacturer == "nortek":
            raise ValueError("A transformation matrix must be provided for Nortek coordinate transformations")

        heading = burst_data.get("heading")
        pitch = burst_data.get("pitch")
        roll = burst_data.get("roll")

        if ((coords_in == "enu") or (coords_out == "enu")) and ((heading is None) or (pitch is None) or (roll is None)):
            constant_hpr = self._rotate.get("constant_hpr")
            if constant_hpr:
                heading, pitch, roll = constant_hpr
            else:
                raise ValueError(
                    "Heading, pitch, and roll must be provided for any coordinate transformation to/from ENU"
                )

        # HPR is instrument-level (one time series for the whole instrument), not
        # indexed per depth bin. Pass the same heading/pitch/roll to every bin.
        for height_idx in range(self.n_heights):
            u1 = burst_data["u1"][height_idx, :]
            u2 = burst_data["u2"][height_idx, :]
            u3 = burst_data["u3"][height_idx, :]

            if self.manufacturer == "nortek" and self.num_beams == 3:
                u1_new, u2_new, u3_new = coord_transform_3_beam_nortek(
                    u1=u1,
                    u2=u2,
                    u3=u3,
                    heading=heading,
                    pitch=pitch,
                    roll=roll,
                    transformation_matrix=transformation_matrix,
                    declination=declination,
                    orientation=self.orientation,
                    coords_in=coords_in,
                    coords_out=coords_out,
                )
                burst_data["u1"][height_idx, :] = u1_new
                burst_data["u2"][height_idx, :] = u2_new
                burst_data["u3"][height_idx, :] = u3_new
            elif self.manufacturer == "nortek" and self.num_beams > 3:
                u4 = burst_data["u4"][height_idx, :]
                u1_new, u2_new, u3_new, u4_new = coord_transform_4_beam_nortek(
                    u1=u1,
                    u2=u2,
                    u3=u3,
                    u4=u4,
                    heading=heading,
                    pitch=pitch,
                    roll=roll,
                    transformation_matrix=transformation_matrix,
                    declination=declination,
                    orientation=self.orientation,
                    coords_in=coords_in,
                    coords_out=coords_out,
                )
                burst_data["u1"][height_idx, :] = u1_new
                burst_data["u2"][height_idx, :] = u2_new
                burst_data["u3"][height_idx, :] = u3_new
                burst_data["u4"][height_idx, :] = u4_new
            elif self.manufacturer == "rdi":
                u4 = burst_data["u4"][height_idx, :]
                u1_new, u2_new, u3_new, u4_new = coord_transform_4_beam_rdi(
                    u1=u1,
                    u2=u2,
                    u3=u3,
                    u4=u4,
                    heading=heading,
                    pitch=pitch,
                    roll=roll,
                    beam_angle=self.beam_angle,
                    transformation_matrix=transformation_matrix,
                    declination=declination,
                    orientation=self.orientation,
                    coords_in=coords_in,
                    coords_out=coords_out,
                )
                burst_data["u1"][height_idx, :] = u1_new
                burst_data["u2"][height_idx, :] = u2_new
                burst_data["u3"][height_idx, :] = u3_new
                burst_data["u4"][height_idx, :] = u4_new
            else:
                raise ValueError(
                    f"Invalid combination of manufacturer='{self.manufacturer}' and "
                    f"num_beams={self.num_beams} for coordinate transformation"
                )

        burst_data["coords"] = coords_out
        return burst_data

    def _apply_flow_rotation(self, burst_data, flow_rotation):
        """
        Rotate u1/u2/u3 to align with the burst-mean flow direction.

        Parameters
        ----------
        burst_data : dict
            Burst data dictionary. Must be in non-beam coordinates.
        flow_rotation : str or tuple
            "align_principal", "align_current", or (theta_h_deg, theta_v_deg).

        Returns
        -------
        dict
            burst_data with u1/u2/u3 rotated and burst_data["rotation"] set.
        """
        if isinstance(flow_rotation, str):
            if flow_rotation == "align_principal":
                theta_h, theta_v = align_with_principal_axis(burst_data["u1"], burst_data["u2"], burst_data["u3"])
            elif flow_rotation == "align_current":
                theta_h, theta_v = align_with_flow(burst_data["u1"], burst_data["u2"], burst_data["u3"])
            else:
                raise ValueError(
                    f"Invalid flow_rotation '{flow_rotation}'. Must be 'align_principal' or 'align_current'."
                )
        elif isinstance(flow_rotation, tuple):
            theta_h, theta_v = flow_rotation
        else:
            raise TypeError(f"flow_rotation must be a str or tuple, got {type(flow_rotation)}")

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
        if burst_data["coords"] == "beam":
            raise ValueError(
                "Shear calculation is not supported for beam coordinates. "
                "Specify 'coords_out' as 'xyz' or 'enu' in preprocessing options."
            )
        z = self.z
        out = {}
        for vel_key in ["u1", "u2", "u3"]:
            u = burst_data[vel_key]
            u_bar = np.mean(u, axis=1)
            dudz = np.gradient(u_bar, z, axis=0, edge_order=2)
            out[f"d{vel_key}_dz"] = dudz

        return out

    def covariance(
        self,
        burst_data: dict,
        method: str = "variance",
        f_cutoff_ogive: float = 0.1,
        pitch: np.ndarray = np.array([0.0]),
        roll: np.ndarray = np.array([0.0]),
        **kwargs,
    ):

        if method not in ["variance", "ogive_fit", "5beam"]:
            raise ValueError(f"Invalid covariance method '{method}'. Must be 'variance', 'ogive_fit', or '5beam'.")

        if burst_data["coords"] != "beam":
            burst_data = copy.deepcopy(burst_data)
            burst_data = self._apply_coord_transform(burst_data, "beam")

        beam_angle_rad = np.deg2rad(self.beam_angle)
        out = {}
        if method == "variance" or method == "ogive_fit":
            if self.manufacturer == "nortek":
                stress_beam_map = {"uw": ("u1", "u3"), "vw": ("u2", "u4")}
            elif self.manufacturer == "rdi":
                stress_beam_map = {"uw": ("u1", "u2"), "vw": ("u3", "u4")}

            for stress_key, vel_pair in stress_beam_map.items():
                u1_key = vel_pair[0]
                u2_key = vel_pair[1]
                u1_bar = np.mean(burst_data[u1_key], axis=1, keepdims=True)
                u2_bar = np.mean(burst_data[u2_key], axis=1, keepdims=True)
                u1_prime = burst_data[u1_key] - u1_bar
                u2_prime = burst_data[u2_key] - u2_bar
                u1_var = np.mean(u1_prime**2, axis=1)
                u2_var = np.mean(u2_prime**2, axis=1)
                stress_estimate = (u1_var - u2_var) / (2 * np.sin(2 * beam_angle_rad))

                if method == "variance":
                    out[stress_key] = stress_estimate
                elif method == "ogive_fit":
                    # Ogive curve based on Kaimal spectrum
                    def model_ogive(k, uw, k0):
                        A = (7 / (3 * np.pi)) * np.sin(3 * np.pi / 7)
                        cospectrum = uw * A * (1 / k0) / (1 + (k / k0) ** (7 / 3))
                        ogive = cumulative_trapezoid(cospectrum, k, initial=0)
                        return ogive

                    out[stress_key] = np.empty((self.n_heights,))
                    for height_idx in range(self.n_heights):
                        u_bar = np.sqrt(u1_bar[height_idx] ** 2 + u2_bar[height_idx] ** 2).squeeze()
                        f, P_u1 = psd(u1_prime[height_idx, :], fs=self.fs, **kwargs)
                        f, P_u2 = psd(u2_prime[height_idx, :], fs=self.fs, **kwargs)
                        k_measured = 2 * np.pi * f / u_bar
                        Co_measured = (P_u1 - P_u2) / (2 * np.sin(2 * beam_angle_rad))
                        Co_measured_k = Co_measured * u_bar / (2 * np.pi)
                        ogive_measured = cumulative_trapezoid(Co_measured_k, k_measured, initial=0)

                        # Cutoff
                        k_cutoff = 2 * np.pi * f_cutoff_ogive / u_bar
                        fit_indices = k_measured < k_cutoff

                        # Initial guesses
                        uw_0 = stress_estimate[height_idx]
                        k0_0 = k_measured[np.argmax(k_measured * Co_measured_k)]

                        popt, pcov = curve_fit(
                            f=model_ogive,
                            xdata=k_measured[fit_indices],
                            ydata=ogive_measured[fit_indices],
                            p0=(uw_0, k0_0),
                        )
                        out[stress_key][height_idx] = popt[0]
        elif method == "5beam":
            if self.num_beams != 5:
                raise ValueError("5beam covariance requires 5 beams")

            # Implement guerra and thomson
            pitch = circmean(np.deg2rad(pitch))
            pitch = (pitch + np.pi) % (2 * np.pi) - np.pi
            roll = circmean(np.deg2rad(roll))
            roll = (roll + np.pi) % (2 * np.pi) - np.pi

            # Using their variable names to make life easier
            theta = beam_angle_rad
            u5 = burst_data["u5"]
            if self.manufacturer == "nortek":
                phi_2 = roll
                phi_3 = -pitch
                u1 = burst_data["u1"]
                u2 = burst_data["u3"]
                u3 = burst_data["u4"]
                u4 = burst_data["u2"]
            elif self.manufacturer == "rdi":
                phi_2 = pitch
                phi_3 = roll
                u1 = burst_data["u1"]
                u2 = burst_data["u2"]
                u3 = burst_data["u3"]
                u4 = burst_data["u4"]

            u1_bar = np.mean(u1, axis=1, keepdims=True)
            u2_bar = np.mean(u2, axis=1, keepdims=True)
            u3_bar = np.mean(u3, axis=1, keepdims=True)
            u4_bar = np.mean(u4, axis=1, keepdims=True)
            u5_bar = np.mean(u5, axis=1, keepdims=True)
            u1_prime = u1 - u1_bar
            u2_prime = u2 - u2_bar
            u3_prime = u3 - u3_bar
            u4_prime = u4 - u4_bar
            u5_prime = u5 - u5_bar
            u1_var = np.mean(u1_prime**2, axis=1)
            u2_var = np.mean(u2_prime**2, axis=1)
            u3_var = np.mean(u3_prime**2, axis=1)
            u4_var = np.mean(u4_prime**2, axis=1)
            u5_var = np.mean(u5_prime**2, axis=1)

            # Getting u-v covariance from xyz transformed data
            burst_xyz = self._apply_coord_transform(burst_data, "xyz")
            u = burst_xyz["u1"]
            v = burst_xyz["u2"]
            u_bar = np.mean(u, axis=1, keepdims=True)
            v_bar = np.mean(v, axis=1, keepdims=True)
            u_prime = u - u_bar
            v_prime = v - v_bar
            uv_cov = np.mean(u_prime * v_prime, axis=1)

            # Convenient definitions
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            denom = 4 * sin_theta**6 * cos_theta**2

            out["uu"] = (-1 / denom) * (
                -2 * sin_theta**4 * cos_theta**2 * (u2_var + u1_var - 2 * cos_theta**2 * u5_var)
                + 2 * sin_theta**5 * cos_theta * phi_3 * (u2_var - u1_var)
            )

            # Assuming u_1^3 in the paper is a typo
            out["vv"] = (-1 / denom) * (
                -2 * sin_theta**4 * cos_theta**2 * (u4_var + u1_var - 2 * cos_theta**2 * u5_var)
                - 2 * sin_theta**4 * cos_theta**2 * phi_3 * (u2_var - u1_var)
                + 2 * sin_theta**3 * cos_theta**3 * phi_3 * (u2_var - u1_var)
                - 2 * sin_theta**5 * cos_theta * phi_2 * (u4_var - u3_var)
            )

            out["ww"] = (-1 / denom) * (
                -2 * sin_theta**5 * cos_theta * phi_3 * (u2_var - u1_var)
                + 2 * sin_theta**5 * cos_theta * phi_2 * (u4_var - u3_var)
                - 4 * sin_theta**6 * cos_theta**2 * u5_var
            )

            out["uw"] = (-1 / denom) * (
                sin_theta**5 * cos_theta * (u2_var - u1_var)
                + 2 * sin_theta**4 * cos_theta**2 * phi_3 * (u2_var + u1_var)
                - 4 * sin_theta**4 * cos_theta**2 * phi_3 * u5_var
                - 4 * sin_theta**6 * cos_theta**2 * phi_2 * uv_cov
            )

            out["vw"] = (-1 / denom) * (
                sin_theta**5 * cos_theta * (u4_var - u3_var)
                - 2 * sin_theta**4 * cos_theta**2 * phi_2 * (u4_var + u3_var)
                + 4 * sin_theta**4 * cos_theta**2 * phi_2 * u5_var
                + 4 * sin_theta**6 * cos_theta**2 * phi_3 * uv_cov
            )

        return out

    def dissipation(self, burst_data, method="4beam_spectral", sf_kwargs=None, **kwargs):

        if burst_data["coords"] != "beam":
            u_bar = np.mean(np.sqrt(burst_data["u1"] ** 2 + burst_data["u2"] ** 2), axis=1)
            burst_data = copy.deepcopy(burst_data)
            burst_data = self._apply_coord_transform(burst_data, "beam")
        else:
            burst_data_xyz = self._apply_coord_transform(burst_data, "xyz")
            u_bar = np.mean(np.sqrt(burst_data_xyz["u1"] ** 2 + burst_data_xyz["u2"] ** 2), axis=1)

        if method not in ["4beam_spectral", "5th_beam_spectral", "structure_function"]:
            raise ValueError(
                f"Invalid dissipation method '{method}'. Must be '4beam_spectral', '5th_beam_spectral', or 'structure_function'."
            )

        # Kolmogorov constants
        C_u = 0.5
        C_w = 0.67

        beam_angle_rad = np.deg2rad(self.beam_angle)
        out = {}
        if method == "4beam_spectral":
            C = (
                2 * C_u * np.sin(beam_angle_rad) ** 2
                + 2 * C_w * np.sin(beam_angle_rad) ** 2
                + 4 * C_w * np.cos(beam_angle_rad) ** 2
            )

            # Beam velocities
            u1 = burst_data["u1"]
            u2 = burst_data["u2"]
            u3 = burst_data["u3"]
            u4 = burst_data["u4"]
            u1_bar = np.mean(u1, axis=1, keepdims=True)
            u2_bar = np.mean(u2, axis=1, keepdims=True)
            u3_bar = np.mean(u3, axis=1, keepdims=True)
            u4_bar = np.mean(u4, axis=1, keepdims=True)
            u1_prime = u1 - u1_bar
            u2_prime = u2 - u2_bar
            u3_prime = u3 - u3_bar
            u4_prime = u4 - u4_bar
            out["eps"] = np.empty((self.n_heights,))
            for height_idx in range(self.n_heights):
                f, P_11 = psd(u1_prime[height_idx, :], fs=self.fs, **kwargs)
                f, P_22 = psd(u2_prime[height_idx, :], fs=self.fs, **kwargs)
                f, P_33 = psd(u3_prime[height_idx, :], fs=self.fs, **kwargs)
                f, P_44 = psd(u4_prime[height_idx, :], fs=self.fs, **kwargs)
                P_T = P_11 + P_22 + P_33 + P_44
                P_T_k = P_T * u_bar[height_idx] / (2 * np.pi)
                k = 2 * np.pi * f / u_bar[height_idx]
                X = C * k ** (-5 / 3)
                y = P_T_k
                idx_fit = k > 0
                slope, *_ = linregress(X[idx_fit], y[idx_fit])
                out["eps"][height_idx] = slope ** (3 / 2)
        elif method == "5th_beam_spectral":
            u5 = burst_data["u5"]
            u5_bar = np.mean(u5, axis=1, keepdims=True)
            u5_prime = u5 - u5_bar
            out["eps"] = np.empty((self.n_heights,))
            for height_idx in range(self.n_heights):
                f, P_55 = psd(u5_prime[height_idx, :], fs=self.fs, **kwargs)
                P_55_k = P_55 * u_bar[height_idx] / (2 * np.pi)
                k = 2 * np.pi * f / u_bar[height_idx]
                X = C_w * k ** (-5 / 3)
                y = P_55_k
                idx_fit = k > 0
                slope, *_ = linregress(X[idx_fit], y[idx_fit])
                out["eps"][height_idx] = slope ** (3 / 2)

        elif method == "structure_function":
            z_start = sf_kwargs.get("z_start_idx", 0)
            z_end = sf_kwargs.get("z_end_idx", -1)
            min_points = sf_kwargs.get("min_points", 3)
            beams = sf_kwargs.get("beams", self.beam_keys)
            heights = self.z[z_start:z_end]

            # z x r x beam
            D_ll = np.zeros((len(heights), len(heights), len(beams))) * np.nan
            eps = np.empty((len(heights), len(beams)))
            for jj, vel_beam in enumerate(beams):
                u = burst_data[vel_beam]
                u_bar = np.mean(u, axis=1, keepdims=True)
                u_prime = u - u_bar
                for ii in range(len(heights) - min_points):
                    D_ll[ii, ii:z_end, jj] = np.mean((u_prime[ii:z_end, :] - u_prime[ii, :]) ** 2, axis=1)
                    r = heights[ii:z_end] - heights[ii]
                    X = 2.1 * r ** (2 / 3)
                    y = D_ll[ii, ii:z_end, jj]
                    good_indices = ~np.isnan(y)
                    if sum(good_indices) >= min_points:
                        slope, *_ = linregress(X[good_indices], y[good_indices])
                        eps[ii, jj] = slope ** (3 / 2)
                    else:
                        eps[ii, jj] = np.nan

            # Averaging over beams
            out["eps"] = np.nanmean(eps, axis=1)

        return out

    def tke(self, burst_data):
        pass

    @property
    def beam_keys(self):
        return [k for k in ["u1", "u2", "u3", "u4", "u5"] if k in self.name_map]

    @property
    def num_beams(self):
        return len(self.beam_keys)
