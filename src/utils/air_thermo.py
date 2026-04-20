import numpy as np
from typing import Optional, TypeAlias

from utils.constants import (
    GRAVITATIONAL_ACCELERATION as g,
    GAS_CONSTANT_UNIVERSAL as R,
    GAS_CONSTANT_DRY_AIR as R_a,
    GAS_CONSTANT_WATER_VAPOR as R_v,
    MOL_MASS_DRY_AIR as m_a,
    MOL_MASS_WATER_VAPOR as m_v,
)

Numeric: TypeAlias = float | int | np.ndarray


def t_c2kelvin(t: Numeric) -> Numeric:
    """Convert temperature from Celsius to Kelvin."""
    return t + 273.15


def p_mbar2pa(p: Numeric) -> Numeric:
    """Convert pressure from millibar to Pascal."""
    return p * 100


def saturation_vapor_pressure(
    t: Numeric, p: Numeric, sp: Optional[Numeric] = None, t_freeze: Optional[Numeric] = None
) -> Numeric:
    """
    Saturation vapor pressure given pressure, temperature, and (optionally) seawater salinity

    Parameters
    ----------
    t : Numeric
        Air temperature in Celcius
    p : Numeric
        Atmospheric pressure in millibar
    sp : Numeric, optional
        If specified, the saturation vapor pressure is corrected to its "above seawater"
        value using salinity in PSU
    t_freeze: Numeric, optional
        If specified, the saturation vapor pressure is corrected to account for ice conditions

    Returns
    -------
    Numeric
        Saturation vapor pressure in millibar

    """
    e_s = 6.1121 * (1.0007 + 3.46e-6 * p) * np.exp(17.502 * t / (240.97 + t))

    if t_freeze is not None:
        t = np.asarray(t, dtype=float)
        p = np.asarray(p, dtype=float)
        e_s = np.asarray(e_s, dtype=float)
        t_freeze = np.asarray(t_freeze, dtype=float)
        ice_idx = t < t_freeze
        if np.any(ice_idx):
            e_s[ice_idx] = (
                6.1115 * np.exp(22.452 * t[ice_idx] / (t[ice_idx] + 272.55)) * (1.0003 + 4.18e-6 * p[ice_idx])
            )

    if sp is not None:
        return e_s * (1 - 5.37e-04 * sp)
    else:
        return e_s


def water_vapor_pressure(t: Numeric, p: Numeric, rh: Numeric, sp: Optional[Numeric] = None) -> Numeric:
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
    sp : Numeric, optional
        If specified, the saturation vapor pressure is corrected to its "above seawater"
        value using salinity in PSU

    Returns
    -------
    Numeric
        Water vapor pressure in millibar
    """
    e_s = saturation_vapor_pressure(t, p, sp)
    return (rh / 100) * e_s


def water_vapor_density(t: Numeric, p: Numeric, rh: Numeric, sp: Optional[Numeric] = None) -> Numeric:
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
    sp : Numeric, optional
        If specified, the saturation vapor pressure is corrected to its "above seawater"
        value using salinity in PSU

    Returns
    -------
    Numeric
        Water vapor density in kg/m^3
    """
    e = water_vapor_pressure(t, p, rh, sp)
    return 100 * e / (R_v * t_c2kelvin(t))


def mixing_ratio(t: Numeric, p: Numeric, rh: Numeric, sp: Optional[Numeric] = None) -> Numeric:
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
    sp : Numeric, optional
        If specified, the saturation vapor pressure is corrected to its "above seawater"
        value using salinity in PSU

    Returns
    -------
    Numeric
        Mixing ratio in kg/kg
    """
    e = water_vapor_pressure(t, p, rh, sp)
    return 0.622 * e / (p - e)


def specific_humidity(t: Numeric, p: Numeric, rh: Numeric, sp: Optional[Numeric] = None) -> Numeric:
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
    sp : Numeric, optional
        If specified, the saturation vapor pressure is corrected to its "above seawater"
        value using salinity in PSU

    Returns
    -------
    Numeric
        Specific humidity in kg/kg
    """
    e = water_vapor_pressure(t, p, rh, sp)
    q = 0.622 * e / (p - 0.378 * e)
    return q


def saturation_specific_humidity(
    t: Numeric, p: Numeric, sp: Optional[Numeric] = None, t_freeze: Optional[Numeric] = None
) -> Numeric:
    """
    Specific humidity given temperature, pressure, relative humidity, and (optionally) seawater salinity

    Parameters
    ----------
    t : Numeric
        Air temperature in Celcius
    p : Numeric
        Atmospheric pressure in millibar
    sp : Numeric, optional
        If specified, the saturation vapor pressure is corrected to its "above seawater"
        value using salinity in PSU
    t_freeze: Numeric, optional
        If specific, the saturation vapor pressure is corrected to account for ice conditions

    Returns
    -------
    Numeric
        Specific humidity in kg/kg
    """
    e_s = saturation_vapor_pressure(t, p, sp, t_freeze)
    q_s = 0.622 * e_s / (p - 0.378 * e_s)
    return q_s


def relative_humidity_from_specific_humidity(
    t: Numeric, p: Numeric, q: Numeric, t_freeze: Optional[Numeric] = None
) -> Numeric:
    """

    Parameters
    ----------
    t
    p
    q
    t_freeze

    Returns
    -------

    """
    e_s = saturation_vapor_pressure(t, p, t_freeze=t_freeze)
    vapor_pressure = q * p / (0.622 + 0.378 * q)
    return 100 * vapor_pressure / e_s


def virtual_temperature(t: Numeric, p: Numeric, rh: Numeric, sp: Optional[Numeric] = None) -> Numeric:
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
    sp : Numeric, optional
        If specified, the saturation vapor pressure is corrected to its "above seawater"
        value using salinity in PSU

    Returns
    -------
    Numeric
        Virtual temperature in Celcius
    """
    q = specific_humidity(t, p, rh, sp)
    t_v = t * (1 + 0.61 * q)
    return t_v


def air_density(t: Numeric, p: Numeric, rh: Numeric) -> Numeric:
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
    e = water_vapor_pressure(t, p, rh)
    p_dry = p - e
    rho_air = (p_mbar2pa(p_dry) * m_a + p_mbar2pa(e) * m_v) / (R * t_c2kelvin(t))
    return rho_air


def dry_air_density(t: Numeric, p: Numeric) -> Numeric:
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
    rho_air_dry = p_mbar2pa(p) / (R_a * t_c2kelvin(t))
    return rho_air_dry


def specific_heat(t: Numeric) -> Numeric:
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


def dry_adiabatic_lapse_rate(t: Numeric, g_lat: Numeric = g) -> Numeric:
    """

    Parameters
    ----------
    t

    Returns
    -------

    """
    cp = specific_heat(t)
    return g_lat / cp


def latent_heat_of_vaporization(t: Numeric) -> Numeric:
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


def kinematic_viscosity(t: Numeric) -> Numeric:
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


def potential_temperature(t: Numeric, z: Numeric) -> Numeric:
    """
    Potential temperature, i.e. the temperature an air parcel would have if brought adiabatically
    to a reference level at the surface

    Parameters
    ----------
    t : Numeric
        Air temperature in Celcius
    z : Numeric
        Height above the surface in meters.

    Returns
    -------
    Numeric
        Potential temperature in Celcius
    """
    cp = specific_heat(t)
    theta = t + (g / cp) * z
    return theta
