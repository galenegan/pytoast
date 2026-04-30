import copy
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import curve_fit
from scipy.stats import circmean, linregress
from typing import Optional, Union, List, Dict, Any
from utils.base_instrument import BaseInstrument
from utils.despike_utils import threshold, goring_nikora, recursive_gaussian
from utils.spectral_utils import psd

from utils.rotate_utils import (
    coord_transform_3_beam_nortek,
    coord_transform_4_beam_nortek,
    coord_transform_4_beam_rdi,
    min_angle,
    apply_flow_rotation,
)


class ADCP(BaseInstrument):
    """Class for processing data from Acoustic Doppler Current Profiler (ADCP)
    instruments.

    Contains methods for: - Loading data from source files
    - Preprocessing (despiking, coordinate transformations, flow-dependent rotations)
    - Calculating mean shear
    - Calculating turbulence statistics: TKE dissipation, Reynolds stress
    """

    def __init__(
        self,
        files: Union[str, List],
        name_map: dict,
        deployment_type: str = "fixed",
        fs: Optional[float] = None,
        z: Optional[Union[List[float], np.ndarray]] = None,
        data_keys: Optional[Union[str, List[str]]] = None,
        source_coords: str = "beam",
        orientation: str = "up",
        beam_angle: float = 25.0,
        manufacturer: str = "nortek",
    ):
        """Initialize an ADCP object.

        Parameters
        ----------
        files : str or List[str]
            Path(s) to data files. If a list, each element is treated as a file containing data from
            an individual burst period. Supported formats: .npy (saved as a dict), .mat (saved as a
            MATLAB struct), .csv (variables in columns). If variables are two-dimensional, the larger
            dimension is assumed to be time and the shorter dimension a vertical coordinate.
        name_map : dict
            Mapping of standard variable names to names in the data files, e.g.:
            {
                "u1": "first beam/direction velocity variable name",
                "u2": "second beam/direction velocity variable name",
                "u3": "third beam/direction velocity variable name",
                "u4": "fourth beam/direction velocity variable name",  # optional
                "u5": "fifth beam/direction velocity variable name",   # optional
                "heading": "heading variable name",   # optional
                "pitch": "pitch variable name",       # optional
                "roll": "roll variable name",         # optional
                "z": "height variable name",          # optional
                "p": "pressure variable name",        # optional
                "time": "time variable name",         # optional
            }
            An error is raised if `time` is absent and `fs` is also not provided. `z` in the name_map is only used if
            the `z` argument is not specified directly. `heading`, `pitch`, and `roll` are required for any coordinate
            transformation involving ENU coordinates. "u4" and "u5" can be optionally specified for instruments with
            4 or 5 beams.
        deployment_type : str, optional
            One of {"fixed", "cast"} depending on how the instrument is deployed. Default is "fixed", in which case
            self.z will be converted to a constant numpy array of instrument deployment depths or measurement cell
            heights. If "cast", self.z will be set to None and vertical coordinates will be calculated as a data
            variable within individual measurement bursts.
        fs : float, optional
            Sampling frequency (Hz). If not provided, it will be inferred (and rounded to 2 decimal places) from the
            `time` variable
        z : List[float] or np.ndarray, optional
            Vertical coordinate for each cell (interpreted as m above bed if `orientation="up"`, m below surface if
            `orientation="down"`). Defaults to integer indices if not specified.
        data_keys : str or List[str], optional
            One or more nested keys to traverse after loading the file (e.g. "Data" if the variables in name_map are
            stored at `burst["Data"]["variable_name"]`).
        source_coords : str, optional
            Velocity coordinate system in the source files. One of {`beam`, `xyz`, `enu`}.
            Defaults to `beam`.
        orientation : str, optional
            Instrument orientation. One of {`up`, `down`}. Affects interpretation of the vertical
            coordinate. Defaults to `up`.
        beam_angle : float, optional
            Beam angle from vertical (degrees). Used in beam-to-xyz coordinate transformations.
            Defaults to 25.0.
        manufacturer : str, optional
            Instrument manufacturer. One of {`nortek`, `rdi`}. Determines the coordinate transformation logic. Defaults
            to `nortek`.

        Returns
        -------
        ADCP object
        """
        self.source_coords = source_coords
        self.orientation = orientation
        self.beam_angle = beam_angle
        self.manufacturer = manufacturer
        files_list = files if isinstance(files, list) else [files]
        ADCP.validate_inputs(
            files_list,
            name_map,
            deployment_type,
            fs,
            z,
            data_keys,
            source_coords,
            orientation,
            beam_angle,
            manufacturer,
        )
        super().__init__(files, name_map, deployment_type=deployment_type, fs=fs, z=z, data_keys=data_keys)

    @staticmethod
    def validate_inputs(
        files: Union[str, List],
        name_map: dict,
        deployment_type: str = "fixed",
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[List[Union[float, int]], np.ndarray]] = None,
        data_keys: Optional[Union[str, List[str]]] = None,
        source_coords: str = "beam",
        orientation: str = "up",
        beam_angle: float = 25.0,
        manufacturer: str = "nortek",
    ):

        # General validation
        BaseInstrument.validate_common_inputs(files, name_map, deployment_type, fs, z, data_keys)

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

    def set_preprocess_opts(self, opts: Dict[str, Any]):
        """
        Enable preprocessing for all subsequent burst loads using the options defined in the input dictionary.

        Parameters
        ----------
        opts : dict
            Preprocessing options. Supported keys:

            despike : dict, optional
                Options for despiking. If not specified, no despiking is applied. Supported keys:

                method : {'threshold', 'goring_nikora', 'recursive_gaussian'}
                    If `threshold`, data is despiked by replacing any samples with a magnitude outside a specified
                    range. If `goring_nikora`, data is despiked using the Goring & Nikora (2002) algorithm. If
                    `recursive_gaussian`, data is despiked using a recursive Gaussian filter.

                If ``{'method': 'goring_nikora', ...}``, additional keys can be (see `goring_nikora` docstring):
                    remaining_spikes : int
                    max_iter : int
                    robust_statistics : bool

                If ``{'method': 'threshold', ...}``, additional keys can be:
                    threshold_min : float
                    threshold_max : float

                If ``{'method': 'recursive_gaussian', ...}``, additional keys can be:
                    alpha : float
                    max_iter : int

            rotate : dict, optional
                Options for rotations and coordinate transformations. If not specified, no rotations applied.
                Supported keys:

                    coords_out : str, optional
                        Coordinates for burst["coords"] to be transformed to. One of {`beam`, `xyz`, `enu`}.
                    transformation_matrix : np.ndarray, optional
                        Transformation matrix for the instrument. Must be specified for coordinate transformation if
                        manufacturer = 'nortek'. May be excluded if manufacturer = 'rdi' in which case ADCP.beam_angle
                        is used to compute the transformation matrix.
                    declination : float, optional
                        Magnetic declination in degrees. Added to heading for coordinate transformations.
                    constant_hpr : Tuple[float], optional
                        Constant heading, pitch, and roll angles to apply.
                    flow_rotation : str or Tuple[float], optional.
                        One of {`align_principal`, `align_streamwise`, or (horizontal_angle, vertical_angle)}. If
                        `align_principal` then the velocity will be rotated to align with the principal axes of the
                        flow. If `align_streamwise` then the velocity will be rotated to align with the horizontal current
                        magnitude sqrt(u^2 + v^2). In both cases, the vertical velocity will be minimized. If float
                        angles are specified in a tuple, the flow will be rotated by those angles in the horizontal and
                        vertical planes. Specifying any option will throw an error if `burst["coords"]` == `"beam"`,
                        unless a coordinate system change to `xyz` or `enu` is also requested.
        """

        self._preprocess_opts = opts
        self._preprocess_enabled = True

        self._despike = opts.get("despike", {})
        if self._despike:
            self._despike_method = self._despike.get("method")
            self._despike_opts = {key: val for key, val in self._despike.items() if key != "method"}

        self._rotate = opts.get("rotate", {})
        self._cached_idx = None
        self._cached_data = None

    def _apply_preprocessing(self, burst_data):
        burst_data["coords"] = self.source_coords
        if not self._preprocess_enabled:
            return burst_data

        if self._despike:
            despike_fn = {
                "goring_nikora": goring_nikora,
                "threshold": threshold,
                "recursive_guassian": recursive_gaussian,
            }.get(self._despike_method)
            if despike_fn is None:
                raise ValueError(f"Invalid despiking method '{self._despike_method}'")
            for key in ["u1", "u2", "u3"]:
                burst_data[key] = despike_fn(burst_data[key], **self._despike_opts)

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
                burst_data = apply_flow_rotation(burst_data, flow_rotation)

        return burst_data

    def _apply_coord_transform(self, burst_data, coords_out):
        """Transform velocity components between coordinate systems.

        Uses configuration stored in self._rotate. Can be called from _apply_preprocessing during standard burst
        loading, or directly from analysis methods (e.g., covariance).

        Parameters
        ----------
        burst_data : dict
            Burst data dictionary, with `burst_data["coords"]` reflecting the current velocity coordinate system
        coords_out : str
            Target coordinate system. One of {`beam`, `xyz`, `enu`}.

        Returns
        -------
        dict
            burst_data with velocity components transformed in-place and
            `burst_data["coords"]` updated to `coords_out`.
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

        # Unlike with an ADV stack, HPR is instrument-level and not indexed per depth bin. Therefore, pass the same
        # heading/pitch/roll to every bin.
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

    def shear(self, burst_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculates the mean vertical shear of the 3 cartesian velocity
        components.

        Uses numpy's gradient function with second-order accuracy at the boundaries.

        Parameters
        ----------
        burst_data : dict
            Burst data dictionary. Must be in non-beam coordinates.

        Returns
        -------
        out : dict
            Dictionary containing vertical shear profiles for each velocity component.
        """
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
        ogive_r2_min: float = 0.9,
        sigma_wave_ratio_max: Optional[float] = None,
        pitch: np.ndarray = np.array([0.0]),
        roll: np.ndarray = np.array([0.0]),
        **kwargs,
    ):
        """Calculate Reynolds stress components for a given burst.

        Parameters
        ----------
        burst_data : dict
            Burst data dictionary (any coordinates allowed)
        method : str
            One of {`variance`, `ogive_fit`, `5beam`}, corresponding to the methods of Stacey et al. (1999),
            Kirincich et al. (2010), and Guerra and Thomson (2017), respectively. All methods assume that the
            ADCP beam axes (e.g., 1-3 and 2-4 for Nortek instruments) are aligned with the principal axes of the
            flow. If this is the case, then the `uw` component can be interpreted as the Reynolds stress along the
            major axis and `vw` as the Reynolds stress along the minor axis.
        f_cutoff_ogive : float
            Upper frequency bound (Hz) for the `ogive_fit` method, which should correspond to the frequency at which
            waves begin to significantly contaminate the velocity signal. Defaults to 0.1 Hz.
        ogive_r2_min : float
            Minimum coefficient of determination (r^2) for the `ogive_fit` method to ensure consistency with the
            theoretical Kaimal spectrum. Defaults to 0.9.
        sigma_wave_ratio_max : float
            Maximum ratio of the wave velocity standard deviation to mean velocity for the `ogive_fit` method. If not
            specified then no maximum is applied.
        pitch : np.ndarray
            Instrument pitch angle (degrees) for the burst period. Used in the `5beam` method, defaults to 0.0
        roll : np.ndarray
            Instrument roll angle (degrees) for the burst period. Used in the `5beam` method, defaults to 0.0
        kwargs : dict
            Additional arguments passed to spectral_utils.csd

        Returns
        -------
        out : dict
            Dictionary containing vertical profiles for the various Reynolds stress components. `variance` and
            `ogive_fit` methods only return `uw` and `vw`, while `5beam` additionally returns `uu`, `vv`, and `ww`.

        References
        ----------
        Stacey, M. T., Monismith, S. G., & Burau, J. R. (1999). Measurements of Reynolds stress profiles in unstratified
            tidal flow. Journal of Geophysical Research: Oceans, 104(C5), 10933-10949.
        Kirincich, A. R., Lentz, S. J., & Gerbi, G. P. (2010). Calculating Reynolds stresses from ADCP measurements in
            the presence of surface gravity waves using the cospectra-fit method. Journal of Atmospheric and Oceanic
            Technology, 27(5), 889-907.
        Guerra, M., & Thomson, J. (2017). Turbulence measurements from five-beam acoustic Doppler current profilers.
            Journal of Atmospheric and Oceanic Technology, 34(6), 1267-1284.
        """
        if method not in ["variance", "ogive_fit", "5beam"]:
            raise ValueError(f"Invalid covariance method '{method}'. Must be 'variance', 'ogive_fit', or '5beam'.")

        if burst_data["coords"] != "beam":
            u_bar = np.mean(np.sqrt(burst_data["u1"] ** 2 + burst_data["u2"] ** 2), axis=1)
            burst_data = copy.deepcopy(burst_data)
            burst_data = self._apply_coord_transform(burst_data, "beam")
        else:
            burst_data_temp = copy.deepcopy(burst_data)
            burst_data_xyz = self._apply_coord_transform(burst_data_temp, "xyz")
            u_bar = np.mean(np.sqrt(burst_data_xyz["u1"] ** 2 + burst_data_xyz["u2"] ** 2), axis=1)

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

                    def model_ogive(k, uw, k0):
                        A = (7 / (3 * np.pi)) * np.sin(3 * np.pi / 7)
                        cospectrum = uw * A * (1 / k0) / (1 + (k / k0) ** (7 / 3))
                        ogive_curve = cumulative_trapezoid(cospectrum, k, initial=0)
                        # In the standard formulation (e.g., their Figure 4, panel 3) the Ogive curve is an increasing
                        # function of k/k0 that plateaus at the stress u'w' at high wavenumbers. Here, we subtract that
                        # curve from the stress that we want so that the plateau is at low wavenumbers where we carry
                        # out the fit.
                        flipped_ogive = uw - ogive_curve
                        return flipped_ogive

                    out[stress_key] = np.full((self.n_heights,), np.nan)
                    for height_idx in range(self.n_heights):
                        u_bar_z = u_bar[height_idx]
                        f, P_u1 = psd(u1_prime[height_idx, :], fs=self.fs, **kwargs)
                        f, P_u2 = psd(u2_prime[height_idx, :], fs=self.fs, **kwargs)
                        k_measured = 2 * np.pi * f / u_bar_z
                        Co_measured = (P_u1 - P_u2) / (2 * np.sin(2 * beam_angle_rad))
                        Co_measured_k = Co_measured * u_bar_z / (2 * np.pi)

                        # Same flipping around of the measured Ogive curve as we did with the model
                        ogive_cumulative = cumulative_trapezoid(Co_measured_k, k_measured, initial=0)
                        ogive_measured = ogive_cumulative[-1] - ogive_cumulative

                        k_cutoff = 2 * np.pi * f_cutoff_ogive / u_bar_z
                        fit_indices = (k_measured > 0) & (k_measured < k_cutoff)

                        # sigma_wave_ratio_max check
                        # wave variance is estimated from beam PSD above the cutoff frequency.
                        if sigma_wave_ratio_max is not None:
                            wave_indices = f > f_cutoff_ogive
                            if wave_indices.any():
                                sigma_wave_sq = np.trapezoid(
                                    (P_u1[wave_indices] + P_u2[wave_indices]) / 2,
                                    f[wave_indices],
                                )
                                sigma_wave = np.sqrt(max(sigma_wave_sq, 0.0))
                                if sigma_wave / u_bar_z > sigma_wave_ratio_max:
                                    continue

                        # Initial guesses: k0 from the sub-wave band only, to avoid
                        # the wave peak biasing the spectral-peak estimate.
                        uw_0 = stress_estimate[height_idx]
                        fit_k = k_measured[fit_indices]
                        fit_Co_k = Co_measured_k[fit_indices]
                        if fit_k.size > 0 and np.any(fit_Co_k != 0):
                            k0_0 = fit_k[np.argmax(np.abs(fit_k * fit_Co_k))]
                        else:
                            k0_0 = k_cutoff / 2

                        # Wrap in a try/except in case it doesn't converge
                        try:
                            popt, _ = curve_fit(
                                f=model_ogive,
                                xdata=k_measured[fit_indices],
                                ydata=ogive_measured[fit_indices],
                                p0=(uw_0, k0_0),
                                bounds=([-np.inf, 0], [np.inf, np.inf]),
                                maxfev=10000,
                            )
                        except RuntimeError:
                            continue

                        uw_fit, k0_fit = popt

                        # Make sure that k0 is positive (should be enforced by bounds, but guard against edge cases).
                        if k0_fit <= 0:
                            continue

                        # r^2 between model and measured ogive
                        ogive_model = model_ogive(k_measured[fit_indices], uw_fit, k0_fit)
                        ss_res = np.sum((ogive_measured[fit_indices] - ogive_model) ** 2)
                        ss_tot = np.sum((ogive_measured[fit_indices] - np.mean(ogive_measured[fit_indices])) ** 2)
                        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                        if r2 < ogive_r2_min:
                            continue

                        out[stress_key][height_idx] = uw_fit

        elif method == "5beam":
            if self.num_beams != 5:
                raise ValueError("5beam covariance requires 5 beams")

            # Implement guerra and thomson
            pitch = circmean(np.deg2rad(min_angle(pitch)))
            roll = circmean(np.deg2rad(min_angle(roll)))

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
                -2 * sin_theta**4 * cos_theta**2 * (u4_var + u3_var - 2 * cos_theta**2 * u5_var)
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

    def dissipation(
        self,
        burst_data: Dict[str, np.ndarray],
        method: str = "4beam_spectral",
        f_min: float = None,
        f_max: float = None,
        sf_kwargs: dict = None,
        **kwargs,
    ) -> np.ndarray:
        """Estimate the dissipation rate of TKE for a given burst.

        Parameters
        ----------
        burst_data : dict
            Burst data dictionary (any coordinates allowed)
        method : str
            One of {`4beam_spectral`, `5th_beam_spectral`, `structure_function`}. The spectral methods follow
            McMillan et al. (2016) and the structure function method follows McMillan and Hay (2017).
        f_min: : float
            Lower bound of inertial subrange for the spectral fits
        f_max : float
            Upper bound of inertial subrange for the spectral fits.
        sf_kwargs : dict
            Additional keyword arguments to pass to the structure function method. Keys allowed:
                z_start_idx : int
                    Lower bound index of self.z to include in the structure function calculation. Defaults to 0.
                z_end_idx : int
                    Upper bound index of self.z to include in the structure function calculation. Defaults to self.n_heights.
                r_min : float
                    Minimum separation to include in the regression for epsilon. Default None
                r_max : float
                    Maximum separation to include in the regression for epsilon. Default None
                min_points : float
                    Minimum number of data points to include in the regression. Defaults to 3
                beams : List[str]
                    Beam names (e.g., ["u1", "u2"]) to average over. Defaults to self.beam_keys

        kwargs : dict
            Additional keyword arguments to pass to the spectral utils.

        Returns
        -------
        eps : np.ndarray
            Vertical profile of dissipation for the burst period

        References
        ----------
        McMillan, J. M., Hay, A. E., Lueck, R. G., & Wolk, F. (2016). Rates of dissipation of turbulent kinetic energy
            in a high Reynolds number tidal channel. Journal of Atmospheric and Oceanic Technology, 33(4), 817-837.
        McMillan, J. M., & Hay, A. E. (2017). Spectral and structure function estimates of turbulence dissipation rates
            in a high-flow tidal channel using broadband ADCPs. Journal of Atmospheric and Oceanic Technology, 34(1), 5-20.
        """
        if burst_data["coords"] != "beam":
            u_bar = np.mean(np.sqrt(burst_data["u1"] ** 2 + burst_data["u2"] ** 2), axis=1)
            burst_data = copy.deepcopy(burst_data)
            burst_data = self._apply_coord_transform(burst_data, "beam")
        else:
            burst_data_temp = copy.deepcopy(burst_data)
            burst_data_xyz = self._apply_coord_transform(burst_data_temp, "xyz")
            u_bar = np.mean(np.sqrt(burst_data_xyz["u1"] ** 2 + burst_data_xyz["u2"] ** 2), axis=1)

        if method not in ["4beam_spectral", "5th_beam_spectral", "structure_function"]:
            raise ValueError(
                f"Invalid dissipation method '{method}'. Must be '4beam_spectral', '5th_beam_spectral', or 'structure_function'."
            )

        # Kolmogorov constants. Some people prefer 0.52 and 0.69, but it only 4% matters.
        C_u = 0.5
        C_w = 0.67

        beam_angle_rad = np.deg2rad(self.beam_angle)
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
            eps_out = np.empty((self.n_heights,))
            for height_idx in range(self.n_heights):
                f, P_11 = psd(u1_prime[height_idx, :], fs=self.fs, **kwargs)
                f, P_22 = psd(u2_prime[height_idx, :], fs=self.fs, **kwargs)
                f, P_33 = psd(u3_prime[height_idx, :], fs=self.fs, **kwargs)
                f, P_44 = psd(u4_prime[height_idx, :], fs=self.fs, **kwargs)
                P_T = P_11 + P_22 + P_33 + P_44
                P_T_k = P_T * u_bar[height_idx] / (2 * np.pi)
                k = 2 * np.pi * f / u_bar[height_idx]
                idx_fit = k > 0
                if f_min:
                    idx_fit &= k >= 2 * np.pi * f_min / u_bar[height_idx]
                if f_max:
                    idx_fit &= k <= 2 * np.pi * f_max / u_bar[height_idx]
                X = C * k ** (-5 / 3)
                y = P_T_k
                slope, *_ = linregress(X[idx_fit], y[idx_fit])
                eps_out[height_idx] = slope ** (3 / 2)
        elif method == "5th_beam_spectral":
            u5 = burst_data["u5"]
            u5_bar = np.mean(u5, axis=1, keepdims=True)
            u5_prime = u5 - u5_bar
            eps_out = np.empty((self.n_heights,))
            for height_idx in range(self.n_heights):
                f, P_55 = psd(u5_prime[height_idx, :], fs=self.fs, **kwargs)
                P_55_k = P_55 * u_bar[height_idx] / (2 * np.pi)
                k = 2 * np.pi * f / u_bar[height_idx]
                X = C_w * k ** (-5 / 3)
                y = P_55_k
                idx_fit = k > 0
                if f_min:
                    idx_fit &= k >= 2 * np.pi * f_min / u_bar[height_idx]
                if f_max:
                    idx_fit &= k <= 2 * np.pi * f_max / u_bar[height_idx]
                slope, *_ = linregress(X[idx_fit], y[idx_fit])
                eps_out[height_idx] = slope ** (3 / 2)

        elif method == "structure_function":
            sf_kwargs = sf_kwargs or {}
            z_start = sf_kwargs.get("z_start_idx", 0)
            z_end = sf_kwargs.get("z_end_idx", self.n_heights)
            r_min = sf_kwargs.get("r_min", None)
            r_max = sf_kwargs.get("r_max", None)
            beams = sf_kwargs.get("beams", self.beam_keys)
            min_points = sf_kwargs.get("min_points", 3)

            heights = self.z[z_start:z_end]
            # z x r x beam
            D_ll = np.full((len(heights), len(heights), len(beams)), np.nan)
            eps = np.full((len(heights), len(beams)), np.nan)
            n_heights_sf = len(heights)
            for jj, vel_beam in enumerate(beams):
                u = burst_data[vel_beam]
                u_bar = np.mean(u, axis=1, keepdims=True)
                u_prime = u - u_bar
                # Subset to the requested depth range for SF computation
                u_prime_sf = u_prime[z_start:z_end, :]
                for ii in range(n_heights_sf - min_points):
                    dW = u_prime_sf[ii:, :] - u_prime_sf[ii, :]
                    dW2 = dW**2
                    # 5-sigma outlier rejection on velocity difference pairs
                    sigma = np.nanstd(dW2)
                    dW2[np.abs(dW2) > 5 * sigma] = np.nan
                    D_ll[ii, ii:, jj] = np.nanmean(dW2, axis=1)
                    r = heights[ii:] - heights[ii]
                    X = 2.1 * r ** (2 / 3)
                    y = D_ll[ii, ii:, jj]

                    # Restrict fit to inertial subrange; always exclude r=0
                    fit_mask = r > 0
                    if r_min is not None:
                        fit_mask &= r >= r_min
                    if r_max is not None:
                        fit_mask &= r <= r_max
                    good_indices = fit_mask & ~np.isnan(y)

                    if good_indices.sum() >= min_points:
                        slope, *_ = linregress(X[good_indices], y[good_indices])
                        eps[ii, jj] = slope ** (3 / 2)
                    else:
                        eps[ii, jj] = np.nan

            # Averaging over beams
            eps_out = np.nanmean(eps, axis=1)

        return eps_out

    @property
    def beam_keys(self):
        return [k for k in ["u1", "u2", "u3", "u4", "u5"] if k in self.name_map]

    @property
    def num_beams(self):
        return len(self.beam_keys)

    def subsample(self, start_idx, end_idx):
        """Subsample the ADCP object between files[start_idx] and
        files[end_idx].

        Parameters
        ----------
        start_idx : int
            First file to include in subsampling
        end_idx : int
            Upper bound (exclusive) on file index in subsampling

        Returns
        -------
        new_adcp : ADCP
            Subsampled ADCP object
        """
        new_adcp = self.__class__(
            files=self.files[start_idx:end_idx],
            name_map=self.name_map,
            deployment_type=self.deployment_type,
            fs=self.fs,
            z=self.z,
            data_keys=self.data_keys,
            source_coords=self.source_coords,
            orientation=self.orientation,
            beam_angle=self.beam_angle,
            manufacturer=self.manufacturer,
        )
        if self._preprocess_enabled:
            new_adcp.set_preprocess_opts(self._preprocess_opts)
        return new_adcp
