import types
import numpy as np
import numpy.testing as npt
from scipy.special import gamma

from ocean.adv import ADV
from testhelpers.synth_utils import generate_wave_turb_burst


def _make_adv(fs, n_heights=1):
    """Return a minimal namespace that satisfies ADV.dissipation attribute
    requirements."""
    return types.SimpleNamespace(fs=fs, n_heights=n_heights)



def test_pure_wave():
    sigma0 = 0.1
    omega = 2 * np.pi / 8
    J11, J22, J33 = ADV._calcJii(sigma0, sigma0, sigma0, 0, 0)
    Mii_integral = (J11 + J22 + J33) * 2 * (2 * np.pi) ** (3 / 2) / (omega ** (5 / 3))
    Mii_analytical = gamma(5 / 6) * 2 ** (17 / 6) * np.pi * sigma0 ** (2 / 3) / (omega ** (5 / 3))
    npt.assert_allclose(Mii_analytical, Mii_integral, rtol=2e-3)  # Some error expected from the trapezoidal integration


def test_dissipation_recovers_prescribed_eps():
    """Validate ADV.dissipation against an analytically known isotropic
    Kolmogorov turbulence field in the Taylor (frozen-turbulence) limit.

    The synthetic burst has mean flow U_d along +x and -5/3 spectra
    matching Gerbi et al. (2009) Eqs. 13-14 with prescribed dissipation
    `eps_true`. The fitted dissipation should recover `eps_true` to
    within sample-estimation noise.
    """
    eps_true = 1.0e-5
    U_d = 1.0
    u, v, w, _, truth = generate_wave_turb_burst(
        fs=16,
        duration_s=1800,
        a=0.0,
        u_d_mean=U_d,
        epsilon=eps_true,
        seed=0,
    )
    adv = _make_adv(fs=16)
    adv._calcJii = ADV._calcJii
    burst = {
        "u1": u.reshape(1, -1),
        "u2": v.reshape(1, -1),
        "u3": w.reshape(1, -1),
        "coords": "xyz",
    }
    out = ADV.dissipation(adv, burst, f_low=0.3, f_high=3.0)
    npt.assert_allclose(out["eps"][0], eps_true, rtol=0.1)
    assert out["quality_flag"][0] == 1


