import numpy.testing as npt
import numpy as np
import types
from testhelpers.synth_utils import generate_wave_turb_burst
from atmosphere.sonic import Sonic

def _make_sonic(fs, n_heights=1):
    """Return a minimal namespace that satisfies ADV.covariance attribute requirements."""
    return types.SimpleNamespace(fs=fs, n_heights=n_heights)



class TestCovariance:
    def test_cov_matches_spectral(self):
        fs = 32
        u, v, w, _, truth = generate_wave_turb_burst(
            fs=fs,
            duration_s=600,
            a=0.0,
            u_d_mean=0.5,
            epsilon=1e-4,
            seed=0,
        )
        sonic = _make_sonic(fs)
        burst_data = {"u1": u.reshape(1, -1), "u2": v.reshape(1, -1), "u3": w.reshape(1, -1), "coords": "xyz"}
        result_direct = Sonic.covariance(sonic, burst_data, method="cov")
        result_spectral = Sonic.covariance(
            sonic, burst_data, method="spectral_integral", window_len=len(u), detrend="constant", window_type="boxcar"
        )

        for key in result_direct.keys():
            npt.assert_allclose(result_direct[key], result_spectral[key], rtol=1e-10)

    def test_cov_recovers_prescribed_stresses(self):
        """
        fill in
        """
        u, v, w, _, truth = generate_wave_turb_burst(fs=16, a=0, duration_s=1800, seed=0)
        sonic = _make_sonic(fs=16)
        burst_data = {"u1": u.reshape(1, -1), "u2": v.reshape(1, -1), "u3": w.reshape(1, -1), "coords": "xyz"}
        result = Sonic.covariance(sonic, burst_data, method="cov")

        for key, expected in truth.items():
            if "turb" in key:
                npt.assert_allclose(result[key.replace("_turb", "")], expected, rtol=0.2, atol=5e-4, err_msg=key)

    def test_tke(self):
        """
        fill in
        Returns
        -------

        """
        fs = 32
        u, v, w, _, truth = generate_wave_turb_burst(
            fs=fs,
            duration_s=600,
            a=0.0,
            u_d_mean=10,
            seed=0,
        )
        sonic = _make_sonic(fs)
        burst_data = {"u1": u.reshape(1, -1), "u2": v.reshape(1, -1), "u3": w.reshape(1, -1), "coords": "xyz"}
        tke_truth = 0.5 * (truth["uu_turb"] + truth["vv_turb"] + truth["ww_turb"])
        tke_calc = Sonic.tke(sonic, burst_data)
        npt.assert_allclose(tke_truth, tke_calc, rtol=1e-3)

class TestDissipation:
    def test_dissipation_close_to_theory(self):
        fs = 32
        eps_truth = 1e-3
        u, v, w, _, truth = generate_wave_turb_burst(
            fs=fs,
            duration_s=600,
            a=0.0,
            u_d_mean=10,
            epsilon=eps_truth,
            seed=0,
        )

        sonic = _make_sonic(fs)
        burst_data = {"u1": u.reshape(1, -1), "u2": v.reshape(1, -1), "u3": w.reshape(1, -1), "coords": "xyz"}
        eps_calc = Sonic.dissipation(sonic, burst_data, f_low=2, f_high=10, henjes_correction=False)

        # Accounting for different Kolmogorov constants
        eps_calc *= (0.53 / (1.5 * 18/55))
        npt.assert_allclose(eps_truth, eps_calc, rtol=1e-1)