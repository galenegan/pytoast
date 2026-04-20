import pytest
import numpy as np
import numpy.testing as npt

from utils.air_thermo import *
from utils.constants import GRAVITATIONAL_ACCELERATION as g

P_ATM = 1013.25
T_AIR = 20.0

# Unit conversions


def test_t_kelvin():
    assert t_c2kelvin(0.0) == 273.15


def test_p_pascal():
    assert p_mbar2pa(10) == 1000


# Standard values


def test_kinematic_viscosity_standard():
    # At 20 C, kinematic viscosity ~1.504e-5 m^2/s per the polynomial formula
    nu = kinematic_viscosity(T_AIR)
    expected = 1.326e-5 * (1 + 6.542e-3 * 20 + 8.301e-6 * 400 - 4.840e-9 * 8000)
    npt.assert_allclose(nu, expected, rtol=1e-10)


def test_dry_air_density_standard():
    # Dry air at 0 C, 1013.25 mbar: ~1.292 kg/m^3
    rho = dry_air_density(0.0, P_ATM)
    npt.assert_allclose(rho, 1.292, rtol=1e-3)

    # Dry air at 15 C, 1013.25 mbar: ~1.225 kg/m^3
    rho = dry_air_density(15, P_ATM)
    npt.assert_allclose(rho, 1.225, rtol=1e-3)


def test_specific_heat_at_0C():
    # Cp at 0 C is the leading coefficient (no temperature-dependent terms)
    assert specific_heat(0.0) == 1005.6


def test_latent_heat_at_0C():
    # Latent heat of vaporization at 0 C is the standard value 2.501e6 J/kg
    assert latent_heat_of_vaporization(0.0) == 2.501e6


def test_latent_heat_decreases_with_temperature():
    # Lv decreases with increasing temperature (negative coefficient)
    assert latent_heat_of_vaporization(20.0) < latent_heat_of_vaporization(0.0)


def test_kinematic_viscosity_increases_with_temperature():
    assert kinematic_viscosity(30.0) > kinematic_viscosity(10.0)


# Round trips and consistency


def test_vapor_pressure_and_sat_vapor_pressure():
    w1 = water_vapor_pressure(T_AIR, P_ATM, 100)
    w2 = saturation_vapor_pressure(T_AIR, P_ATM)
    npt.assert_allclose(w1, w2, rtol=1e-10)


def test_specific_humidity_round_trip():
    q = specific_humidity(T_AIR, P_ATM, 75)
    rh = relative_humidity_from_specific_humidity(T_AIR, P_ATM, q)
    npt.assert_allclose(rh, 75.0, rtol=1e-10)


def test_saturation_specific_humidity_equals_specific_humidity_at_100pct_rh():
    q = specific_humidity(T_AIR, P_ATM, 100.0)
    q_s = saturation_specific_humidity(T_AIR, P_ATM)
    npt.assert_allclose(q, q_s, rtol=1e-10)


def test_mixing_ratio_consistency():
    # Verify w = 0.622 * e / (p - e) using intermediate water_vapor_pressure
    rh = 60.0
    w = mixing_ratio(T_AIR, P_ATM, rh)
    e = water_vapor_pressure(T_AIR, P_ATM, rh)
    npt.assert_allclose(w, 0.622 * e / (P_ATM - e), rtol=1e-10)


def test_water_vapor_density_dry_air():
    # At 0% RH there is no water vapor
    assert water_vapor_density(T_AIR, P_ATM, 0.0) == 0.0


def test_water_vapor_density_increases_with_rh():
    wvd_50 = water_vapor_density(T_AIR, P_ATM, 50.0)
    wvd_100 = water_vapor_density(T_AIR, P_ATM, 100.0)
    assert wvd_100 > wvd_50


def test_virtual_temperature_dry_air():
    # At 0% RH specific humidity is ~0, so virtual temperature ~= air temperature
    t_v = virtual_temperature(T_AIR, P_ATM, 0.0)
    npt.assert_allclose(t_v, T_AIR, rtol=1e-10)


def test_virtual_temperature_moist_exceeds_dry():
    # Moist air has higher virtual temperature than dry air at same T
    t_v_moist = virtual_temperature(T_AIR, P_ATM, 80.0)
    t_v_dry = virtual_temperature(T_AIR, P_ATM, 0.0)
    assert t_v_moist > t_v_dry


def test_air_density_vs_dry_density_at_zero_rh():
    # At 0% RH, moist and dry density formulas agree to within rounding of gas constants
    rho_moist = air_density(T_AIR, P_ATM, 0.0)
    rho_dry = dry_air_density(T_AIR, P_ATM)
    npt.assert_allclose(rho_moist, rho_dry, rtol=1e-3)


def test_potential_temperature_at_surface():
    # At z=0 the adiabatic correction vanishes
    assert potential_temperature(T_AIR, 0.0) == T_AIR


def test_potential_temperature_increases_with_height():
    # Potential temperature increases with altitude for an adiabatic atmosphere
    theta_low = potential_temperature(T_AIR, 100.0)
    theta_high = potential_temperature(T_AIR, 1000.0)
    assert theta_high > theta_low


def test_dry_adiabatic_lapse_rate_equals_g_over_cp():
    dalr = dry_adiabatic_lapse_rate(0.0)
    npt.assert_allclose(dalr, g / specific_heat(0.0), rtol=1e-10)


###########################
# Saturation vapor pressure
###########################
def test_saturation_vapor_pressure_sp_zero_matches_plain():
    # sp=0 salinity correction factor is 1, result should equal no-salinity call
    e_s = saturation_vapor_pressure(T_AIR, P_ATM)
    e_s_sp0 = saturation_vapor_pressure(T_AIR, P_ATM, sp=0.0)
    npt.assert_allclose(e_s_sp0, e_s, rtol=1e-10)


def test_saturation_vapor_pressure_decreases_with_salinity():
    # Raoult's law: dissolved salt lowers saturation vapor pressure
    e_s_fresh = saturation_vapor_pressure(T_AIR, P_ATM, sp=0.0)
    e_s_salt = saturation_vapor_pressure(T_AIR, P_ATM, sp=35.0)
    assert e_s_salt < e_s_fresh


def test_saturation_vapor_pressure_ice_below_freezing():
    # Ice formula gives lower e_s than liquid formula below 0 C
    t = np.array([-10.0, -5.0])
    p = np.full(2, P_ATM)
    e_s_ice = saturation_vapor_pressure(t, p, t_freeze=np.zeros(2))
    e_s_liquid = saturation_vapor_pressure(t, p)
    npt.assert_array_less(e_s_ice, e_s_liquid)


def test_saturation_vapor_pressure_ice_above_freezing_unchanged():
    # Ice correction must not alter values above the freeze threshold
    t = np.array([5.0, 10.0])
    p = np.full(2, P_ATM)
    e_s_ice = saturation_vapor_pressure(t, p, t_freeze=np.zeros(2))
    e_s_liquid = saturation_vapor_pressure(t, p)
    npt.assert_allclose(e_s_ice, e_s_liquid, rtol=1e-10)
