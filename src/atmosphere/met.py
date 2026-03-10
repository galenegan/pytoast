import numpy as np
from typing import Optional, Union, List, Dict, Any, TypeAlias
from src.utils.despike_utils import threshold, goring_nikora, recursive_gaussian
from src.utils.base_instrument import BaseInstrument
from src.utils.constants import (
    GRAVITATIONAL_ACCELERATION as g,
    GAS_CONSTANT_UNIVERSAL as R,
    GAS_CONSTANT_DRY_AIR as R_a,
    GAS_CONSTANT_WATER_VAPOR as R_v,
    MOL_MASS_DRY_AIR as m_a,
    MOL_MASS_WATER_VAPOR as m_v,
)

Numeric: TypeAlias = float | int | np.ndarray


class Met(BaseInstrument):
    """
    Class for processing bulk meteorological data. Contains methods for:
    - Loading data from source files
    - Preprocessing
    - Calculating and converting from/to various useful thermodynamic quantities

    Many of the methods are implemented based on their descriptions in Bradley & Fairall (2007). If a particular
    function/equation lacks a citation, it can likely be found in Appendix A therein.

    Burst dictionary conventions
    ----------------------------
    Variables in a burst dict are assumed to be 2-D arrays of shape (n_heights, n_samples), where the
    first axis corresponds to instrument heights (length self.n_heights) and the second axis is time.
    The individual thermodynamic methods accept any Numeric type and broadcast over these arrays without
    modification. Height information is always taken from self.z (shape (n_heights,)) rather than
    from burst dict keys, so that self.z remains the single source of truth.

    Standard burst dict keys recognised by :meth:`derive`:

    Input keys
        t         : air temperature (deg C)
        p         : atmospheric pressure (mbar)
        rh        : relative humidity (%)
        salinity  : seawater salinity (PSU) -- optional, corrects vapor-pressure quantities

    Output keys added by :meth:`derive`
        e_s         : saturation vapor pressure (mbar)
        e           : water vapor pressure (mbar)
        rho_v       : water vapor density (kg/m^3)
        w          : mixing ratio (kg/kg)
        q           : specific humidity (kg/kg)
        t_v         : virtual temperature (deg C)
        rho_air     : moist air density (kg/m^3)
        rho_air_dry : dry air density (kg/m^3)
        cp          : specific heat capacity (J/(kg K))
        L_v         : latent heat of vaporization (J/kg)
        nu          : kinematic viscosity (m^2/s)
        theta       : potential temperature (deg C)

    References
    ----------
    Bradley, E. F., & Fairall, C. W. (2007). A guide to making climate quality meteorological and flux measurements at
        sea.
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
        Initialize a Met object.

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
                "t": "temperature variable name" or ["var 1", "var 2", ...],
                "p": "pressure variable name" or ["var 1", "var 2", ...],
                "rh": "relative humidity name" or ["var 1", "var 2", ...],
                "time": "time variable name" or ["var 1", "var 2", ...],
            }
            Lists are used when data from multiple instruments are stored in
            separate variables rather than a 2-D array.
        fs : float, optional
            Sampling frequency (Hz). If not provided, it will be inferred (and rounded to 2 decimal places) from the
            `time` variable
        z : float or List[float], optional
            Mean height above the surface (m) for each instrument. Defaults to integer indices if not
            specified.
        data_keys : str or List[str], optional
            One or more nested keys to traverse after loading the file (e.g. "Data" if the variables
            in name_map are stored at `burst["Data"]["variable_name"]`).

        Returns
        -------
        Met
        """
        files_list = files if isinstance(files, list) else [files]
        Met.validate_inputs(files_list, name_map, fs, z, data_keys)
        super().__init__(files, name_map, fs, z, data_keys)

    @staticmethod
    def validate_inputs(
        files: Union[str, List],
        name_map: dict,
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[float, int, List[Union[float, int]]]] = None,
        data_keys: Optional[Union[str, List[str]]] = None,
    ):

        # General validation
        BaseInstrument.validate_common_inputs(files, name_map, fs, z, data_keys)

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
                "recursive_guassian": recursive_gaussian,
            }.get(self._despike_method)
            if despike_fn is None:
                raise ValueError(f"Invalid despiking method '{self._despike_method}'")
            for key in self.data_keys:
                burst_data[key] = despike_fn(burst_data[key], **self._despike_opts)

        return burst_data

    def t_kelvin(self, t: Numeric) -> Numeric:
        """Convert temperature from Celsius to Kelvin."""
        return t + 273.15

    def p_pascal(self, p: Numeric) -> Numeric:
        """Convert pressure from millibar to Pascal."""
        return p * 100

    def saturation_vapor_pressure(self, t: Numeric, p: Numeric, salinity: Optional[Numeric] = None) -> Numeric:
        """
        Saturation vapor pressure given pressure, temperature, and (optionally) seawater salinity

        Parameters
        ----------
        t : Numeric
            Air temperature in Celcius
        p : Numeric
            Atmospheric pressure in millibar
        salinity : Numeric, optional
            If specified, the saturation vapor pressure is corrected to its "above seawater"
            value using salinity in PSU

        Returns
        -------
        Numeric
            Saturation vapor pressure in millibar

        """
        e_s = 6.1121 * (1.0007 + 3.46e-6 * p) * np.exp(17.502 * t / (240.97 + t))
        if salinity is not None:
            return e_s * (1 - 5.37e-04 * salinity)
        else:
            return e_s

    def water_vapor_pressure(self, t: Numeric, p: Numeric, rh: Numeric, salinity: Optional[Numeric] = None) -> Numeric:
        """
        Water vapor pressure given temperature, pressure, relative humidity, and (optionally) seawater salinity

        Parameters
        ----------
        t : Numeric
            Air temperature in Celcius
        p : Numeric
            Atmospheric pressure in millibar
        rh : Numeric
            Relative humidity in %
        salinity : Numeric, optional
            If specified, the saturation vapor pressure is corrected to its "above seawater"
            value using salinity in PSU

        Returns
        -------
        Numeric
            Water vapor pressure in millibar
        """
        e_s = self.saturation_vapor_pressure(t, p, salinity)
        return (rh / 100) * e_s

    def water_vapor_density(self, t: Numeric, p: Numeric, rh: Numeric, salinity: Optional[Numeric] = None) -> Numeric:
        """
        Water vapor density given temperature, pressure, relative humidity, and (optionally) seawater salinity

        Parameters
        ----------
        t : Numeric
            Air temperature in Celcius
        p : Numeric
            Atmospheric pressure in millibar
        rh : Numeric
            Relative humidity in %
        salinity : Numeric, optional
            If specified, the saturation vapor pressure is corrected to its "above seawater"
            value using salinity in PSU

        Returns
        -------
        Numeric
            Water vapor density in kg/m^3
        """
        e = self.water_vapor_pressure(t, p, rh, salinity)
        return 100 * e / (R_v * self.t_kelvin(t))

    def mixing_ratio(self, t: Numeric, p: Numeric, rh: Numeric, salinity: Optional[Numeric] = None) -> Numeric:
        """
        Water vapor mixing ratio given temperature, pressure, relative humidity, and (optionally) seawater salinity

        Parameters
        ----------
        t : Numeric
            Air temperature in Celcius
        p : Numeric
            Atmospheric pressure in millibar
        rh : Numeric
            Relative humidity in %
        salinity : Numeric, optional
            If specified, the saturation vapor pressure is corrected to its "above seawater"
            value using salinity in PSU

        Returns
        -------
        Numeric
            Mixing ratio in kg/kg
        """
        e = self.water_vapor_pressure(t, p, rh, salinity)
        return 0.622 * e / (p - e)

    def specific_humidity(self, t: Numeric, p: Numeric, rh: Numeric, salinity: Optional[Numeric] = None) -> Numeric:
        """
        Specific humidity given temperature, pressure, relative humidity, and (optionally) seawater salinity

        Parameters
        ----------
        t : Numeric
            Air temperature in Celcius
        p : Numeric
            Atmospheric pressure in millibar
        rh : Numeric
            Relative humidity in %
        salinity : Numeric, optional
            If specified, the saturation vapor pressure is corrected to its "above seawater"
            value using salinity in PSU

        Returns
        -------
        Numeric
            Specific humidity in kg/kg
        """
        e = self.water_vapor_pressure(t, p, rh, salinity)
        q = 0.622 * e / (p - 0.378 * e)
        return q

    def virtual_temperature(self, t: Numeric, p: Numeric, rh: Numeric, salinity: Optional[Numeric] = None) -> Numeric:
        """
        Virtual temperature given temperature, pressure, relative humidity, and (optionally) seawater salinity

        Parameters
        ----------
        t : Numeric
            Air temperature in Celcius
        p : Numeric
            Atmospheric pressure in millibar
        rh : Numeric
            Relative humidity in %
        salinity : Numeric, optional
            If specified, the saturation vapor pressure is corrected to its "above seawater"
            value using salinity in PSU

        Returns
        -------
        Numeric
            Virtual temperature in Celcius
        """
        q = self.specific_humidity(t, p, rh, salinity)
        t_v = t * (1 + 0.61 * q)
        return t_v

    def air_density(self, t: Numeric, p: Numeric, rh: Numeric) -> Numeric:
        """
        Moist air density given temperature, pressure, and relative humidity

        Parameters
        ----------
        t : Numeric
            Air temperature in Celcius
        p : Numeric
            Atmospheric pressure in millibar
        rh : Numeric
            Relative humidity in %

        Returns
        -------
        Numeric
            Moist air density in kg/m^3
        """
        e = self.water_vapor_pressure(t, p, rh)
        p_dry = p - e
        rho_air = (self.p_pascal(p_dry) * m_a + self.p_pascal(e) * m_v) / (R * self.t_kelvin(t))
        return rho_air

    def dry_air_density(self, t: Numeric, p: Numeric) -> Numeric:
        """
        Dry air density given temperature and pressure

        Parameters
        ----------
        t : Numeric
            Air temperature in Celcius
        p : Numeric
            Atmospheric pressure in millibar

        Returns
        -------
        Numeric
            Dry air density in kg/m^3
        """
        rho_air_dry = self.p_pascal(p) / (R_a * self.t_kelvin(t))
        return rho_air_dry

    def specific_heat(self, t: Numeric) -> Numeric:
        """
        Specific heat capacity of air at constant pressure

        Parameters
        ----------
        t : Numeric
            Air temperature in Celcius

        Returns
        -------
        Numeric
            Specific heat capacity in J/(kg K)
        """
        return 1005.6 + 0.0172 * t + 0.000392 * t**2

    def latent_heat_of_vaporization(self, t: Numeric) -> Numeric:
        """
        Latent heat of vaporization

        Parameters
        ----------
        t : Numeric
            Air temperature in Celcius

        Returns
        -------
        Numeric
            Latent heat of vaporization in J/kg
        """
        return (2.501 - 0.00237 * t) * 1e6

    def kinematic_viscosity(self, t: Numeric) -> Numeric:
        """
        Kinematic viscosity of air

        Parameters
        ----------
        t : Numeric
            Air temperature in Celcius

        Returns
        -------
        Numeric
            Kinematic viscosity in m^2/s
        """
        return 1.326e-5 * (1 + 6.542e-3 * t + 8.301e-6 * t**2 - 4.840e-9 * t**3)

    def potential_temperature(self, t: Numeric, z: Numeric) -> Numeric:
        """
        Potential temperature, i.e. the temperature an air parcel would have if brought adiabatically
        to a reference level at the surface

        Parameters
        ----------
        t : Numeric
            Air temperature in Celcius
        z : Numeric
            Height above the surface in meters. When called via :meth:`derive`, this is taken from
            self.z and broadcast over the time dimension automatically.

        Returns
        -------
        Numeric
            Potential temperature in Celcius
        """
        cp = self.specific_heat(t)
        theta = t + (g / cp) * z
        return theta

    def derive(self, burst_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute all thermodynamic quantities derivable from the variables present in a burst
        dictionary, and return the burst dictionary augmented with those results.

        Each quantity is computed only when all of its required inputs are available as keys in
        ``burst_data``. The method never raises for missing inputs -- it simply skips any
        quantities it cannot compute. Salinity is treated as optional throughout; when the
        ``"salinity"`` key is present its value is forwarded to the vapor-pressure calculations.

        Height is always taken from ``self.z`` (shape (n_heights,)) and is not read from
        ``burst_data``. When computing potential temperature, ``self.z`` is reshaped to
        (n_heights, 1) so that it broadcasts correctly against (n_heights, n_samples) arrays.

        Input keys recognized
        ----------------------
        t         : air temperature (deg C),         shape (n_heights, n_samples)
        p         : atmospheric pressure (mbar),     shape (n_heights, n_samples)
        rh        : relative humidity (%),           shape (n_heights, n_samples)
        salinity  : seawater salinity (PSU),         shape (n_heights, n_samples) -- optional

        Output keys added to burst_data (all shape (n_heights, n_samples))
        -------------------------------------------------------------------
        e_s         : saturation vapor pressure (mbar)     -- requires t, p
        rho_air_dry : dry air density (kg/m^3)              -- requires t, p
        e           : water vapor pressure (mbar)          -- requires t, p, rh
        rho_v       : water vapor density (kg/m^3)         -- requires t, p, rh
        w          : mixing ratio (kg/kg)                  -- requires t, p, rh
        q           : specific humidity (kg/kg)             -- requires t, p, rh
        t_v         : virtual temperature (deg C)           -- requires t, p, rh
        rho_air     : moist air density (kg/m^3)            -- requires t, p, rh
        cp          : specific heat capacity (J/(kg K))     -- requires t
        L_v         : latent heat of vaporization (J/kg)    -- requires t
        nu          : kinematic viscosity (m^2/s)           -- requires t
        theta       : potential temperature (deg C)         -- requires t (uses self.z)

        Parameters
        ----------
        burst_data : dict
            Burst dictionary whose keys are standard variable names (see above). Arrays are expected
            to have shape (n_heights, n_samples). The dictionary is modified in-place and also
            returned.

        Returns
        -------
        dict
            The input ``burst_data`` dictionary with derived quantities added as new keys.

        """
        t = burst_data.get("t")
        p = burst_data.get("p")
        rh = burst_data.get("rh")
        salinity = burst_data.get("salinity")

        has_t = t is not None
        has_p = p is not None
        has_rh = rh is not None

        # Temperature-only quantities
        if has_t:
            burst_data["cp"] = self.specific_heat(t)
            burst_data["L_v"] = self.latent_heat_of_vaporization(t)
            burst_data["nu"] = self.kinematic_viscosity(t)
            # self.z has shape (n_heights,); reshape to (n_heights, 1) to broadcast over time axis
            burst_data["theta"] = self.potential_temperature(t, self.z.reshape(-1, 1))

        # Temperature + pressure quantities
        if has_t and has_p:
            burst_data["e_s"] = self.saturation_vapor_pressure(t, p, salinity)
            burst_data["rho_air_dry"] = self.dry_air_density(t, p)

        # Temperature + pressure + relative humidity quantities
        if has_t and has_p and has_rh:
            burst_data["e"] = self.water_vapor_pressure(t, p, rh, salinity)
            burst_data["rho_v"] = self.water_vapor_density(t, p, rh, salinity)
            burst_data["w"] = self.mixing_ratio(t, p, rh, salinity)
            burst_data["q"] = self.specific_humidity(t, p, rh, salinity)
            burst_data["t_v"] = self.virtual_temperature(t, p, rh, salinity)
            burst_data["rho_air"] = self.air_density(t, p, rh)

        return burst_data

    @property
    def data_keys(self):
        return [k for k in self.name_map if k != "time"]
