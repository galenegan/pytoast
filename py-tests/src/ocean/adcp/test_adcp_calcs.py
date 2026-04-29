import numpy as np
import numpy.testing as npt
import pytest

from ocean.adcp import ADCP
from testhelpers.stub_utils import make_adcp
from testhelpers.synth_utils import KAPPA, generate_profile_burst


def _xyz_burst(u, v, w):
    """
    Build a burst dict in xyz coords with 4 channels [u, v, z1, z2] (z1 = z2 = w),
    plus u5 = w. ADCP.covariance/dissipation deepcopy this when coords != "beam"
    before applying inv(T) to get beam coordinates, so the method's standard-projection
    geometry is preserved.
    """
    return {"u1": u, "u2": v, "u3": w, "u4": w.copy(), "u5": w.copy(), "coords": "xyz"}


########
# Shear
########


def test_shear_recovers_log_profile():
    z = np.geomspace(0.5, 5.0, 16)
    u_star = 0.05
    u, v, w, truth = generate_profile_burst(fs=4, duration_s=1800, z=z, u_star=u_star, seed=0)
    adcp = make_adcp(fs=4, z=z, num_beams=5)
    burst = {"u1": u, "u2": v, "u3": w, "coords": "xyz"}
    out = adcp.shear(burst)

    expected = u_star / (KAPPA * z)
    # Tolerance is set by two effects: (1) np.gradient(edge_order=2) truncation error,
    # which scales as dz^2 * |d^3u/dz^3|, and (2) sample-mean noise on u_bar(z) from
    # the finite-duration turbulence realization. Boundary bins are skipped for (1).
    npt.assert_allclose(out["du1_dz"][2:-2], expected[2:-2], rtol=0.15)


def test_shear_in_beam_raises():
    adcp = make_adcp(fs=4, z=np.array([1.0, 2.0]), num_beams=5)
    burst = {"u1": np.zeros((2, 100)), "u2": np.zeros((2, 100)), "u3": np.zeros((2, 100)), "coords": "beam"}
    with pytest.raises(ValueError, match="beam coordinates"):
        adcp.shear(burst)


#############
# Covariance
#############


def test_variance_method_recovers_uw():
    z = np.linspace(0.5, 5.0, 8)
    u, v, w, truth = generate_profile_burst(fs=4, duration_s=1800, z=z, u_star=0.05, seed=0)
    adcp = make_adcp(fs=4, z=z, num_beams=4)
    out = ADCP.covariance(adcp, _xyz_burst(u, v, w), method="variance")

    npt.assert_allclose(out["uw"], truth["uw"], rtol=0.1)
    npt.assert_allclose(out["vw"], np.zeros_like(truth["uw"]), atol=2e-4)


def test_variance_method_zero_for_isotropic():
    z = np.linspace(0.5, 2.5, 4)
    u, v, w, _ = generate_profile_burst(
        fs=8,
        duration_s=1200,
        z=z,
        profile_type="isotropic",
        U=0.5,
        epsilon=1e-5,
        seed=0,
    )
    adcp = make_adcp(fs=8, z=z, num_beams=4)
    out = ADCP.covariance(adcp, _xyz_burst(u, v, w), method="variance")
    # Isotropic synth has no Reynolds shear stress; recovered uw, vw should be small
    # compared with the velocity variance (~5e-3 m^2/s^2 for these parameters).
    npt.assert_allclose(out["uw"], 0.0, atol=5e-3)
    npt.assert_allclose(out["vw"], 0.0, atol=5e-3)


def test_5beam_method_recovers_stresses():
    z = np.linspace(0.5, 5.0, 8)
    u, v, w, truth = generate_profile_burst(fs=4, duration_s=1800, z=z, u_star=0.05, seed=0)
    adcp = make_adcp(fs=4, z=z, num_beams=5)
    out = ADCP.covariance(
        adcp,
        _xyz_burst(u, v, w),
        method="5beam",
        pitch=np.array([0.0]),
        roll=np.array([0.0]),
    )

    npt.assert_allclose(out["uu"], truth["uu"], rtol=0.05)
    npt.assert_allclose(out["vv"], truth["vv"], rtol=0.05)
    npt.assert_allclose(out["ww"], truth["ww"], rtol=0.05)
    npt.assert_allclose(out["uw"], truth["uw"], rtol=0.1)

def test_ogive_method_recovers_stresses():
    z = np.linspace(0.5, 5.0, 8)
    u, v, w, truth = generate_profile_burst(fs=4, duration_s=1800, z=z, u_star=0.05, seed=0)
    adcp = make_adcp(fs=4, z=z, num_beams=5)
    out = ADCP.covariance(
        adcp,
        _xyz_burst(u, v, w),
        method="ogive_fit"
    )
    npt.assert_allclose(out["uw"], truth["uw"], rtol=0.1)


def test_covariance_invalid_method_raises():
    adcp = make_adcp(fs=4, z=np.array([1.0]), num_beams=4)
    burst = _xyz_burst(np.zeros((1, 64)), np.zeros((1, 64)), np.zeros((1, 64)))
    with pytest.raises(ValueError, match="Invalid covariance method"):
        ADCP.covariance(adcp, burst, method="bogus")


def test_covariance_5beam_requires_5_beams():
    z = np.array([1.0])
    u, v, w, _ = generate_profile_burst(fs=4, duration_s=300, z=z, u_star=0.05, seed=0)
    adcp = make_adcp(fs=4, z=z, num_beams=4)
    with pytest.raises(ValueError, match="5beam covariance requires 5 beams"):
        ADCP.covariance(adcp, _xyz_burst(u, v, w), method="5beam")


##############
# Dissipation
##############


def test_4beam_spectral_recovers_eps():
    z = np.linspace(0.5, 2.5, 4)
    eps_truth = 1.0e-5
    u, v, w, truth = generate_profile_burst(
        fs=16,
        duration_s=1800,
        z=z,
        profile_type="isotropic",
        U=0.5,
        epsilon=eps_truth,
        seed=0,
    )
    adcp = make_adcp(fs=16, z=z, num_beams=4)
    eps_out = ADCP.dissipation(adcp, _xyz_burst(u, v, w), method="4beam_spectral", f_min=1.0, f_max=4.0)
    npt.assert_allclose(eps_out, truth["epsilon"], rtol=0.1)


def test_5th_beam_spectral_recovers_eps():
    z = np.linspace(0.5, 2.5, 4)
    eps_truth = 1.0e-5
    u, v, w, truth = generate_profile_burst(
        fs=16,
        duration_s=1800,
        z=z,
        profile_type="isotropic",
        U=0.5,
        epsilon=eps_truth,
        seed=0,
    )
    adcp = make_adcp(fs=16, z=z, num_beams=5)
    eps_out = ADCP.dissipation(adcp, _xyz_burst(u, v, w), method="5th_beam_spectral", f_min=1.0, f_max=4.0)
    npt.assert_allclose(eps_out, truth["epsilon"], rtol=0.1)


def test_dissipation_invalid_method_raises():
    adcp = make_adcp(fs=4, z=np.array([1.0]), num_beams=4)
    burst = _xyz_burst(np.zeros((1, 64)), np.zeros((1, 64)), np.zeros((1, 64)))
    with pytest.raises(ValueError, match="Invalid dissipation method"):
        ADCP.dissipation(adcp, burst, method="bogus")
