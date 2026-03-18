import numpy as np
from typing import Optional, Union, List, Dict, Any, TypeAlias
from src.utils.despike_utils import threshold, goring_nikora, recursive_gaussian
from src.utils.base_instrument import BaseInstrument
import src.utils.sea_thermo as sea_thermo

Numeric: TypeAlias = float | int | np.ndarray


class CTD(BaseInstrument):
    """
    Class for processing CTD (conductivity/temperature/depth) data. Contains
    methods for loading data from source files, preprocessing, and calculating
    thermodynamic quantities from CTD observations.

    The core functionality is a *very limited* port of the Gibbs SeaWater (GSW)
    Oceanographic Toolbox (TEOS-10, https://www.teos-10.org). Only the
    equation of state and directly derived quantities are implemented. Variable
    names generally follow the GSW conventions for consistency with the source
    code.

    Burst dictionary conventions
    ----------------------------
    Variables in a burst dict are assumed to be 2-D arrays of shape
    (n_heights, n_samples), where the first axis corresponds to instrument
    depths (length self.n_heights) and the second axis is time. The individual
    thermodynamic methods accept any Numeric type and broadcast over these
    arrays without modification.

    Standard burst dict keys recognized by `CTD.derive`:

    Input keys
        sp  : practical salinity (PSS-78)                         [unitless]
        t   : in-situ temperature                                    [deg C]
        p   : sea pressure (absolute pressure − 10.1325 dbar)         [dbar]
        lat : latitude (scalar)                         [deg N] -- optional

    Output keys added by `CTD.derive`:
        sa          : Absolute Salinity                               [g/kg]
        ct          : Conservative Temperature                       [deg C]
        rho         : in-situ density                               [kg/m³]
        sigma0      : potential density anomaly ref 0 dbar           [kg/m³]
        alpha       : thermal expansion coefficient                    [1/K]
        beta        : haline contraction coefficient                  [kg/g]
        sound_speed : speed of sound                                   [m/s]
        t_freezing  : in-situ freezing temperature                   [deg C]
        cp          : isobaric heat capacity                       [J/(kg K)]
        nu          : kinematic viscosity                            [m²/s]
        N2          : buoyancy frequency squared (n_heights > 1)     [1/s²]
        z           : depth (positive downward)                          [m]

    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
        seawater - 2010. Intergovernmental Oceanographic Commission,
        Manuals and Guides No. 56, UNESCO.
    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
        polynomial expressions for the density and specific volume of seawater
        using the TEOS-10 standard. Ocean Modelling, 90, 29-43.
    McDougall, T.J. and P.M. Barker, 2011: Getting started with TEOS-10 and
        the Gibbs Seawater (GSW) Oceanographic Toolbox. SCOR/IAPSO WG127.
    """

    def __init__(
        self,
        files: Union[str, List],
        name_map: dict,
        fs: Optional[float] = None,
        z: Optional[Union[float, List[float]]] = None,
        data_keys: Optional[Union[str, List[str]]] = None,
    ):
        """
        Initialize a CTD object.

        Parameters
        ----------
        files : str or List[str]
            Path(s) to data files. If a list, each element is treated as a file
            containing data from an individual burst period. Supported formats:
            .npy (saved as a dict), .mat (saved as a MATLAB struct), .csv
            (variables in columns). If variables are two-dimensional, the larger
            dimension is assumed to be time and the shorter dimension a vertical
            coordinate.
        name_map : dict
            Mapping of standard variable names to names in the data files, e.g.:
            {
                "sp": "salinity variable name",
                "t":  "temperature variable name",
                "p":  "pressure variable name",
                "time": "time variable name",
            }
            Lists are used when data from multiple instruments are stored in
            separate variables rather than a 2-D array.
        fs : float, optional
            Sampling frequency (Hz). If not provided, it will be inferred (and
            rounded to 2 decimal places) from the ``time`` variable.
        z : float or List[float], optional
            Instrument depth(s) below the surface (m). Defaults to integer
            indices if not specified.
        data_keys : str or List[str], optional
            One or more nested keys to traverse after loading the file (e.g.
            ``"Data"`` if the variables in name_map are stored at
            ``burst["Data"]["variable_name"]``).

        Returns
        -------
        CTD
        """
        files_list = files if isinstance(files, list) else [files]
        CTD.validate_inputs(files_list, name_map, fs, z, data_keys)
        super().__init__(files, name_map, fs, z, data_keys)

    @staticmethod
    def validate_inputs(
        files: Union[str, List],
        name_map: dict,
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[float, int, List[Union[float, int]]]] = None,
        data_keys: Optional[Union[str, List[str]]] = None,
    ):
        BaseInstrument.validate_common_inputs(files, name_map, fs, z, data_keys)

    def set_preprocess_opts(self, opts: Dict[str, Any]):
        """
        Enable preprocessing for all subsequent burst loads.

        Parameters
        ----------
        opts : dict
            Preprocessing options. Supported keys:

            despike : dict, optional
                Options for despiking. Supported keys:

                method : {'threshold', 'goring_nikora', 'recursive_gaussian'}
                    Despiking algorithm to apply.

                If ``{'method': 'goring_nikora', ...}``, additional keys:
                    remaining_spikes : int
                    max_iter : int
                    robust_statistics : bool

                If ``{'method': 'threshold', ...}``, additional keys:
                    threshold_min : float
                    threshold_max : float

                If ``{'method': 'recursive_gaussian', ...}``, additional keys:
                    alpha : float
                    max_iter : int
        """
        self._preprocess_opts = opts
        self._preprocess_enabled = True

        self._despike = opts.get("despike", {})
        if self._despike:
            self._despike_method = self._despike.get("method")
            self._despike_opts = {key: val for key, val in self._despike.items() if key != "method"}

        self._cached_idx = None
        self._cached_data = None

    def _apply_preprocessing(self, burst_data):
        if not self._preprocess_enabled:
            return burst_data

        if self._despike:
            despike_fn = {
                "goring_nikora": goring_nikora,
                "threshold": threshold,
                "recursive_gaussian": recursive_gaussian,
            }.get(self._despike_method)
            if despike_fn is None:
                raise ValueError(f"Invalid despiking method '{self._despike_method}'")
            var_keys = [k for k in self.name_map if k != "time"]
            for key in var_keys:
                burst_data[key] = despike_fn(burst_data[key], **self._despike_opts)

        return burst_data

    def sa_from_sp(self, sp: Numeric) -> Numeric:
        """
        Absolute Salinity from Practical Salinity using the constant-ratio
        approximation (gsw_sa_from_sp.m, simplified).

        Uses sa = sp × (35.16504 / 35), which skips the geographic Absolute
        Salinity Anomaly (SAAR) correction. Typical error is  ~0.01 g/kg in
        the open ocean. Errors can reach ~0.1 g/kg in marginal seas (Baltic,
        Red Sea, Arctic shelf) where SAAR is significant.

        Parameters
        ----------
        sp : Numeric
            Practical Salinity (PSS-78) [unitless]

        Returns
        -------
        Numeric
            Absolute Salinity [g/kg]
        """
        return sea_thermo.sa_from_sp(sp)

    def ct_from_t(self, sa: Numeric, t: Numeric, p: Numeric) -> Numeric:
        """
        Conservative Temperature from in-situ temperature (gsw_ct_from_t.m).

        Computes potential temperature at p_ref = 0 dbar via two iterations of
        Newton's method using Gibbs-entropy polynomials, then converts to
        Conservative Temperature via the potential-enthalpy polynomial.

        Parameters
        ----------
        sa : Numeric
            Absolute Salinity [g/kg]
        t : Numeric
            In-situ temperature (ITS-90) [deg C]
        p : Numeric
            Sea pressure [dbar]

        Returns
        -------
        Numeric
            Conservative Temperature (ITS-90) [deg C]
        """
        return sea_thermo.ct_from_t(sa, t, p)

    # -------------------------------------------------------------------------
    # 75-term equation of state (Roquet et al., 2015)
    # Coefficients and polynomial structure match gsw_specvol.m / gsw_rho.m /
    # gsw_alpha.m / gsw_beta.m / gsw_sound_speed.m / gsw_sigma0.m exactly.
    # -------------------------------------------------------------------------

    def specific_volume(self, sa: Numeric, ct: Numeric, p: Numeric) -> Numeric:
        """
        Specific volume from the 75-term polynomial EOS (gsw_specvol.m).

        Parameters
        ----------
        sa : Numeric
            Absolute Salinity [g/kg]
        ct : Numeric
            Conservative Temperature [deg C]
        p : Numeric
            Sea pressure [dbar]

        Returns
        -------
        Numeric
            Specific volume [m³/kg]
        """
        return sea_thermo.specific_volume(sa, ct, p)

    def density(self, sa: Numeric, ct: Numeric, p: Numeric) -> Numeric:
        """
        In-situ density from the 75-term polynomial EOS (gsw_rho.m).

        Parameters
        ----------
        sa : Numeric
            Absolute Salinity [g/kg]
        ct : Numeric
            Conservative Temperature [deg C]
        p : Numeric
            Sea pressure [dbar]

        Returns
        -------
        Numeric
            In-situ density [kg/m³]
        """
        return sea_thermo.density(sa, ct, p)

    def alpha(self, sa: Numeric, ct: Numeric, p: Numeric) -> Numeric:
        """
        Thermal expansion coefficient with respect to Conservative Temperature
        from the 75-term polynomial EOS (gsw_alpha.m).

        Parameters
        ----------
        sa : Numeric
            Absolute Salinity [g/kg]
        ct : Numeric
            Conservative Temperature [deg C]
        p : Numeric
            Sea pressure [dbar]

        Returns
        -------
        Numeric
            Thermal expansion coefficient [1/K]
        """
        return sea_thermo.alpha(sa, ct, p)

    def beta(self, sa: Numeric, ct: Numeric, p: Numeric) -> Numeric:
        """
        Haline contraction coefficient at constant Conservative Temperature
        from the 75-term polynomial EOS (gsw_beta.m).

        Parameters
        ----------
        sa : Numeric
            Absolute Salinity [g/kg]
        ct : Numeric
            Conservative Temperature [deg C]
        p : Numeric
            Sea pressure [dbar]

        Returns
        -------
        Numeric
            Haline contraction coefficient [kg/g]
        """
        return sea_thermo.beta(sa, ct, p)

    def sound_speed(self, sa: Numeric, ct: Numeric, p: Numeric) -> Numeric:
        """
        Speed of sound in seawater from the 75-term polynomial EOS (gsw_sound_speed.m).

        Parameters
        ----------
        sa : Numeric
            Absolute Salinity [g/kg]
        ct : Numeric
            Conservative Temperature [deg C]
        p : Numeric
            Sea pressure [dbar]

        Returns
        -------
        Numeric
            Speed of sound [m/s]
        """
        return sea_thermo.sound_speed(sa, ct, p)

    def sigma0(self, sa: Numeric, ct: Numeric) -> Numeric:
        """
        Potential density anomaly referenced to 0 dbar from the 75-term EOS
        (gsw_sigma0.m). Equal to potential density minus 1000 kg/m³.

        Parameters
        ----------
        sa : Numeric
            Absolute Salinity [g/kg]
        ct : Numeric
            Conservative Temperature [deg C]

        Returns
        -------
        Numeric
            Potential density anomaly [kg/m³]
        """
        return sea_thermo.sigma0(sa, ct)

    def freezing_temperature(self, sa: Numeric, p: Numeric) -> Numeric:
        """
        In-situ freezing temperature from a direct polynomial fit (gsw_t_freezing_poly.m).

        Uses the 23-coefficient polynomial given in the comments of gsw_t_freezing_poly.m,
        which avoids calling CT_freezing and t_from_CT. Error is between -8e-4 K and
        +3e-4 K compared with the exact Newton-Raphson method.

        Parameters
        ----------
        sa : Numeric
            Absolute Salinity [g/kg]
        p : Numeric
            Sea pressure [dbar]

        Returns
        -------
        Numeric
            Freezing temperature [deg C]
        """
        return sea_thermo.freezing_temperature(sa, p)

    def heat_capacity(self, sa: Numeric, t: Numeric, p: Numeric) -> Numeric:
        """
        Isobaric specific heat capacity of seawater (Fofonoff, 1985, Table 7).

        C_p(S, t, p) = A + B*S + C*S^(3/2)
                     + (D + E*S + F*S^(3/2)) * p
                     + (G + H*S + I*S^(3/2)) * p^2
                     + (J + K*S + M*S^(3/2)) * p^3

        where each letter coefficient is a polynomial in temperature t, and S is
        Practical Salinity (PSS-78), p is in bars.

        Parameters
        ----------
        sa : Numeric
            Absolute Salinity [g/kg]
        t : Numeric
            In-situ temperature [deg C]
        p : Numeric
            Sea pressure [dbar]

        Returns
        -------
        Numeric
            Isobaric heat capacity [J/(kg·K)]

        References
        ----------
        Fofonoff, N.P., 1985: Physical properties of seawater: A new salinity
            scale and equation of state for seawater. J. Geophys. Res., 90,
            3332-3342.
        """
        return sea_thermo.heat_capacity(sa, t, p)

    def dynamic_viscosity(self, t: Numeric, sa: Numeric) -> Numeric:
        """
        Dynamic viscosity of seawater (Sharqawy et al., 2010).

        Parameters
        ----------
        t : Numeric
            In-situ temperature [deg C]
        sa : Numeric
            Absolute Salinity [g/kg]

        Returns
        -------
        Numeric
            Dynamic viscosity [Pa s]
        """
        return sea_thermo.dynamic_viscosity(t, sa)

    def kinematic_viscosity(self, t: Numeric, sa: Numeric) -> Numeric:
        """
        Kinematic viscosity of seawater.

        Parameters
        ----------
        t : Numeric
            In-situ temperature [deg C]
        sa : Numeric
            Absolute Salinity [g/kg]

        Returns
        -------
        Numeric
            Kinematic viscosity [m²/s]
        """
        return sea_thermo.kinematic_viscosity(t, sa)

    def thermal_conductivity(self, sa: Numeric, t: Numeric, p: Numeric) -> Numeric:
        """
        Thermal conductivity of seawater (Sharqawy et al., 2010, Eq. 14).

        Parameters
        ----------
        sa : Numeric
            Absolute Salinity [g/kg]
        t : Numeric
            In-situ temperature [deg C]
        p : Numeric
            Sea pressure [dbar]


        Returns
        -------
        Numeric
            Thermal conductivity [W/(m K)]
        """
        return sea_thermo.thermal_conductivity(sa, t, p)

    def buoyancy_frequency(
        self,
        sa: np.ndarray,
        ct: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """
        Squared buoyancy (Brunt-Väisälä) frequency from a vertical profile.

        Computed via finite differences of Conservative Temperature and Absolute
        Salinity using the thermal expansion and haline contraction coefficients
        from the 75-term EOS:

            N^2 = g × (alpha × dct/dz − beta × dsa/dz)

        where z is positive upward (taken from self.z). N^2 is evaluated at
        mid-points between adjacent instrument depths, so the output has length
        n_heights − 1 along axis 0.

        Requires n_heights > 1 (i.e., the mooring must have sensors at more than
        one depth).

        Parameters
        ----------
        sa : np.ndarray
            Absolute Salinity, shape (n_heights, n_samples) [g/kg]
        ct : np.ndarray
            Conservative Temperature, shape (n_heights, n_samples) [deg C]
        p : np.ndarray
            Sea pressure, shape (n_heights, n_samples) [dbar]

        Returns
        -------
        np.ndarray
            N² at mid-depth levels, shape (n_heights − 1, n_samples) [1/s²]
        """
        return sea_thermo.buoyancy_frequency(sa, ct, p, self.z)

    def depth_from_pressure(self, p: Numeric, lat: Optional[Numeric] = None) -> Numeric:
        """
        Depth from sea pressure using the UNESCO (1983) formula with optional
        latitude-dependent gravity (Saunders & Fofonoff, 1976).

        Note: depth is returned as a positive quantity (distance below surface). If this function is used to
        populate self.z (positive upward), then the depths returned by this function should be multiplied by -1.

        Parameters
        ----------
        p : Numeric
            Sea pressure [dbar]
        lat : Numeric, optional
            Latitude [degrees north]. If not provided, g = 9.81 m/s² is used.

        Returns
        -------
        Numeric
            Depth (positive downward) [m]
        """
        return sea_thermo.depth_from_pressure(p, lat)

    def pressure_from_depth(self, z: Numeric, lat: Optional[Numeric] = None) -> Numeric:
        """
        Sea pressure from depth (positive downward) using a one-step Newton
        refinement of a hydrostatic initial guess.

        Parameters
        ----------
        z : Numeric
            Depth (positive downward) [m]
        lat : Numeric, optional
            Latitude [degrees north]. If not provided, g = 9.81 m/s² is used.

        Returns
        -------
        Numeric
            Sea pressure [dbar]
        """
        return sea_thermo.pressure_from_depth(z, lat)

    def derive(self, burst_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute all thermodynamic quantities derivable from the variables present
        in a burst dictionary, and return the burst dictionary augmented with
        those results.

        Each quantity is computed only when all of its required inputs are
        available as keys in ``burst_data``. The method never raises for missing
        inputs — it simply skips any quantities it cannot compute.

        Input keys recognized
        ----------------------
        sp  : Practical Salinity (PSS-78)             [unitless]
        t   : in-situ temperature                       [deg C]
        p   : sea pressure                               [dbar]
        lat : latitude (scalar)     [deg N] -- optional, used for depth

        Output keys added to burst_data
        --------------------------------
        sa          : Absolute Salinity [g/kg]             -- requires sp
        ct          : Conservative Temperature [deg C]     -- requires sa, t, p
        rho         : in-situ density [kg/m³]              -- requires sa, ct, p
        sigma0      : potential density anomaly [kg/m³]    -- requires sa, ct
        alpha       : thermal expansion [1/K]              -- requires sa, ct, p
        beta        : haline contraction [kg/g]            -- requires sa, ct, p
        sound_speed : speed of sound [m/s]                 -- requires sa, ct, p
        t_freezing  : freezing temperature [deg C]         -- requires sa, p
        cp          : isobaric heat capacity [J/(kg K)]    -- requires sa, t, p
        nu          : kinematic viscosity [m²/s]           -- requires t, sa
        N2          : buoyancy frequency² [1/s²]           -- requires sa, ct, p
                      (only computed when n_heights > 1)
        z           : depth (positive downward) [m]        -- requires p

        Parameters
        ----------
        burst_data : dict
            Burst dictionary. Arrays are expected to have shape
            (n_heights, n_samples). Modified in-place and also returned.

        Returns
        -------
        dict
            The input ``burst_data`` dictionary with derived quantities added.
        """
        sp = burst_data.get("sp")
        t = burst_data.get("t")
        p = burst_data.get("p")
        lat = burst_data.get("lat")

        has_sp = sp is not None
        has_t = t is not None
        has_p = p is not None

        sa = None
        if has_sp:
            sa = self.sa_from_sp(sp)
            burst_data["sa"] = sa

        ct = None
        if sa is not None and has_t and has_p:
            ct = self.ct_from_t(sa, t, p)
            burst_data["ct"] = ct

        if sa is not None and ct is not None and has_p:
            burst_data["rho"] = self.density(sa, ct, p)
            burst_data["alpha"] = self.alpha(sa, ct, p)
            burst_data["beta"] = self.beta(sa, ct, p)
            burst_data["sound_speed"] = self.sound_speed(sa, ct, p)

        if sa is not None and ct is not None:
            burst_data["sigma0"] = self.sigma0(sa, ct)

        if sa is not None and has_p:
            burst_data["t_freezing"] = self.freezing_temperature(sa, p)

        if sa is not None and has_t and has_p:
            burst_data["cp"] = self.heat_capacity(sa, t, p)

        if sa is not None and has_t:
            burst_data["nu"] = self.kinematic_viscosity(t, sa)

        if sa is not None and ct is not None and has_p and self.n_heights > 1:
            burst_data["N2"] = self.buoyancy_frequency(sa, ct, p)

        if has_p:
            burst_data["z"] = self.depth_from_pressure(p, lat)

        return burst_data

    @property
    def data_keys(self):
        return [k for k in self.name_map if k != "time"]
