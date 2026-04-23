import types
import numpy as np
import numpy.testing as npt
from pathlib import Path

from ocean.adv import ADV
from .synth_utils import generate_wave_turb_burst


# def _generate_synthetic_data():
#     fs = 32
#     rng = np.random.default_rng(42)
#     u = rng.random(fs * 60).reshape(1, -1)
#     v = (0.1 * u + rng.random(fs * 60)).reshape(1, -1)
#     w = (-0.1 * u + rng.random(fs * 60)).reshape(1, -1)
#     return u, v, w, fs


def _make_adv(fs, n_heights=1):
    """Return a minimal namespace that satisfies ADV.covariance attribute requirements."""
    return types.SimpleNamespace(fs=fs, n_heights=n_heights)


class TestMethods:
    def test_spectral_matches_direct(self):
        """description."""
        fs = 32
        u, v, w, _, truth = generate_wave_turb_burst(
            fs=fs,
            duration_s=600,
            a=0.0,
            u_d_mean=0.5,
            epsilon=1e-4,
            seed=0,
        )
        u = u - np.mean(u)
        v = v - np.mean(v)
        w = w - np.mean(w)
        adv = _make_adv(fs)
        burst_data = {"u1": u.reshape(1, -1), "u2": v.reshape(1, -1), "u3": w.reshape(1, -1), "coords": "xyz"}
        result_direct = ADV.covariance(adv, burst_data, method="cov")
        result_spectral = ADV.covariance(
            adv, burst_data, method="spectral_integral", window_len=len(u), detrend=False, window_type="boxcar"
        )

        for key in result_direct.keys():
            npt.assert_allclose(result_direct[key], result_spectral[key], rtol=1e-10)

    def test_spectral_close_to_phase(self):
        """description."""
        fs = 32
        u, v, w, _, truth = generate_wave_turb_burst(
            fs=fs,
            duration_s=600,
            a=0.0,
            u_d_mean=0.5,
            epsilon=1e-4,
            seed=0,
        )
        u = u - np.mean(u)
        v = v - np.mean(v)
        w = w - np.mean(w)
        adv = _make_adv(fs)
        burst_data = {"u1": u.reshape(1, -1), "u2": v.reshape(1, -1), "u3": w.reshape(1, -1), "coords": "xyz"}
        result_spectral = ADV.covariance(adv, burst_data, method="spectral_integral")
        result_phase = ADV.phase_decomposition(adv, u, v, w, f_wave_low=0.1, f_wave_high=0.11)
        for key in result_spectral.keys():
            # Results should be close, though the wave decomposition introduces noise (especially in the off-diagonal
            # components)
            npt.assert_allclose(result_spectral[key], result_phase[f"{key}_turb"], atol=5e-4)

    def test_benilov_recovers_prescribed_stresses(self):
        """
        Validate benilov_decomposition against an analytically known burst: a linear
        monochromatic wave along x superposed on wave-uncorrelated Gaussian turbulence
        with a prescribed Reynolds stress tensor.
        """
        u, v, w, p, truth = generate_wave_turb_burst(fs=8, duration_s=1800, seed=0)
        adv = _make_adv(fs=8)
        adv._physical_z = True
        result = ADV.benilov_decomposition(adv, u, v, w, p, mab=1.0, rho=1020.0, num_windows=32)
        for key, expected in truth.items():
            npt.assert_allclose(result[key], expected, rtol=0.2, atol=5e-4, err_msg=key)

    def test_phase_recovers_prescribed_stresses(self):
        """
        Validate phase_decomposition against an analytically known burst (see
        test_benilov_recovers_prescribed_stresses for construction). Pressure data
        are not used by this method.
        """
        u, v, w, _, truth = generate_wave_turb_burst(fs=8, duration_s=1800, seed=0)
        adv = _make_adv(fs=8)
        result = ADV.phase_decomposition(adv, u, v, w, num_windows=32)
        for key, expected in truth.items():
            npt.assert_allclose(result[key], expected, rtol=0.2, atol=5e-4, err_msg=key)
