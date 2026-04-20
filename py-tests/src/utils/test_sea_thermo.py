import pytest
import numpy as np
import numpy.testing as npt
import scipy.io as sio
from pathlib import Path

from utils.sea_thermo import *
from utils.constants import GRAVITATIONAL_ACCELERATION
####################
# Reference data
####################

TEST_DATA_PATH = f"{Path(__file__).parent}/testdata/gsw_cv.mat"
KEYS_TO_CHECK = [
    "SA_from_SP",
    "CT_from_t",
    "specvol",
    "rho",
    "alpha",
    "beta",
    "sound_speed",
    "sigma0",
    "t_freezing",
    "cp_t_exact",
    "n2",
]


############
# Fixtures
############


@pytest.fixture(scope="module")
def ref():
    full_output = sio.loadmat(TEST_DATA_PATH, simplify_cells=True)["gsw_cv"]
    out = {key: full_output[key] for key in KEYS_TO_CHECK}
    # z_from_p is elevation (+up, negative in ocean); flip sign to match our
    # depth_from_pressure convention (+down, positive in ocean)
    out["depth_from_pressure"] = -full_output["z_from_p"]
    out["pressure_from_depth"] = full_output["p_from_z"]
    return out


@pytest.fixture(scope="module")
def results():
    inp = sio.loadmat(TEST_DATA_PATH, simplify_cells=True)["gsw_cv"]

    # Source casts
    sp = inp["SP_chck_cast"]
    t = inp["t_chck_cast"]
    p = inp["p_chck_cast"]
    lat = inp["lat_chck_cast"]  # shape (3,), broadcasts against (45, 3)
    z_elev = inp["z_from_p"]  # elevation (+up, negative in ocean), (45, 3)

    # Derived quantities
    sa = sa_from_sp(sp)
    ct = ct_from_t(sa, t, p)

    # n2 must be computed per cast because each cast has a different latitude
    n2_cols = [buoyancy_frequency(sa[:, i : i + 1], ct[:, i : i + 1], p[:, i : i + 1], lat[i]) for i in range(3)]

    out = {
        "SA_from_SP": sa,
        "CT_from_t": ct,
        "specvol": specific_volume(sa, ct, p),
        "rho": density(sa, ct, p),
        "alpha": alpha(sa, ct, p),
        "beta": beta(sa, ct, p),
        "sound_speed": sound_speed(sa, ct, p),
        "sigma0": sigma0(sa, ct),
        "t_freezing": freezing_temperature(sa, p),
        "cp_t_exact": heat_capacity(sa, t, p),
        "n2": np.hstack(n2_cols),
        # -z_elev converts GSW elevation to positive-downward depth
        "depth_from_pressure": depth_from_pressure(p, lat),
        "pressure_from_depth": pressure_from_depth(-z_elev, lat),
    }
    return out


##########################
# Tests - GSW reference data
##########################


@pytest.mark.parametrize(
    "key,rtol",
    [
        ("SA_from_SP", 2e-2),
        ("CT_from_t", 1e-3),
        ("specvol", 1e-3),
        ("rho", 1e-3),
        ("alpha", 2e-2),  # near-zero-alpha outliers in marginal seas inflate max
        ("beta", 1e-3),
        ("sound_speed", 1e-3),
        ("sigma0", 2e-2),  # simplified SA propagates into sigma0 at high latitude
        ("t_freezing", 2e-2),
        ("cp_t_exact", 5e-3),  # Fofonoff (1985) polynomial vs TEOS-10 exact Gibbs function: ~0.4% inherent bias
        ("n2", 5e-2),  # simplified SA (no SAAR) accumulates error in alpha/beta at depth
        ("depth_from_pressure", 1e-4),
        ("pressure_from_depth", 1e-4),
    ],
)
def test_key(results, ref, key, rtol):
    npt.assert_allclose(results[key], ref[key], rtol=rtol)


######################################################
# Tests - functions without GSW reference data (Sharqawy et al. 2010)
######################################################


def test_dynamic_viscosity_pure_water():
    # Pure water at 25 C: ~0.890 mPa s (Sharqawy et al. 2010)
    npt.assert_allclose(dynamic_viscosity(25.0, 0.0), 8.903e-4, rtol=1e-3)


def test_dynamic_viscosity_seawater():
    # Seawater at 20 C, 35 g/kg: ~1.073 mPa s (Sharqawy et al. 2010)
    npt.assert_allclose(dynamic_viscosity(20.0, 35.0), 1.073e-3, rtol=1e-2)


def test_kinematic_viscosity_equals_dynamic_over_density():
    # nu = mu / rho; function evaluates density at p=0 internally
    t = np.array([5.0, 15.0, 25.0])
    sa = np.array([0.0, 20.0, 35.0])
    ct = ct_from_t(sa, t, np.zeros(3))
    npt.assert_allclose(
        kinematic_viscosity(t, sa),
        dynamic_viscosity(t, sa) / density(sa, ct, np.zeros(3)),
        rtol=1e-10,
    )


def test_thermal_conductivity_value():
    npt.assert_allclose(thermal_conductivity(35.0, 20.0, 0.0), 0.598, rtol=1e-2)


def test_depth_pressure_round_trip():
    p = np.array([1000.0, 2000.0, 3000.0])
    z = depth_from_pressure(p, np.zeros(3))
    npt.assert_allclose(pressure_from_depth(z, np.zeros(3)), p, rtol=1e-5)
