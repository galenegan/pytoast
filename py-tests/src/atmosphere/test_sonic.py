import glob
import numpy as np
import numpy.testing as npt
from pathlib import Path
from testhelpers.synth_utils import generate_wave_turb_burst
from testhelpers.stub_utils import eq_except, make_sonic
from atmosphere.sonic import Sonic


class TestCovariance:
    def test_cov_matches_spectral(self):
        fs = 32
        u, v, w, _, truth = generate_wave_turb_burst(
            fs=fs,
            duration_s=600,
            a=0.0,
            U=0.5,
            epsilon=1e-4,
            seed=0,
        )
        sonic = make_sonic(fs)
        burst_data = {"u1": u.reshape(1, -1), "u2": v.reshape(1, -1), "u3": w.reshape(1, -1), "coords": "xyz"}
        result_direct = Sonic.covariance(sonic, burst_data, method="cov")
        result_spectral = Sonic.covariance(
            sonic, burst_data, method="spectral_integral", window_len=len(u), detrend="constant", window_type="boxcar"
        )

        for key in result_direct.keys():
            npt.assert_allclose(result_direct[key], result_spectral[key], rtol=1e-10)

    def test_cov_recovers_prescribed_stresses(self):
        u, v, w, _, truth = generate_wave_turb_burst(fs=16, a=0, duration_s=1800, seed=0)
        sonic = make_sonic(fs=16)
        burst_data = {"u1": u.reshape(1, -1), "u2": v.reshape(1, -1), "u3": w.reshape(1, -1), "coords": "xyz"}
        result = Sonic.covariance(sonic, burst_data, method="cov")

        for key, expected in truth.items():
            if "turb" in key:
                npt.assert_allclose(result[key.replace("_turb", "")], expected, rtol=0.2, atol=5e-4, err_msg=key)

    def test_tke(self):
        fs = 32
        u, v, w, _, truth = generate_wave_turb_burst(
            fs=fs,
            duration_s=600,
            a=0.0,
            U=10,
            seed=0,
        )
        sonic = make_sonic(fs)
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
            U=10,
            epsilon=eps_truth,
            seed=0,
        )

        sonic = make_sonic(fs)
        burst_data = {"u1": u.reshape(1, -1), "u2": v.reshape(1, -1), "u3": w.reshape(1, -1), "coords": "xyz"}
        eps_calc = Sonic.dissipation(sonic, burst_data, f_low=2, f_high=10, henjes_correction=False)

        # Accounting for different Kolmogorov constants in the synthetic spectra vs Edson method
        eps_calc *= 0.53 / (1.5 * 18 / 55)
        npt.assert_allclose(eps_truth, eps_calc, rtol=1e-1)

    def test_dissipation_close_to_theory_henjes(self):
        fs = 32
        eps_truth = 1e-3
        u, v, w, _, truth = generate_wave_turb_burst(
            fs=fs,
            duration_s=600,
            a=0.0,
            U=10,
            epsilon=eps_truth,
            seed=0,
        )

        sonic = make_sonic(fs)
        burst_data = {"u1": u.reshape(1, -1), "u2": v.reshape(1, -1), "u3": w.reshape(1, -1), "coords": "xyz"}
        eps_calc = Sonic.dissipation(sonic, burst_data, f_low=2, f_high=10, henjes_correction=True)

        # Accounting for different Kolmogorov constants in the synthetic spectra vs Edson method
        eps_calc *= 0.53 / (1.5 * 18 / 55)
        npt.assert_allclose(eps_truth, eps_calc, rtol=1e-1)


def _write_sonic_npy(path, n_samples=64):
    data = {
        "U": np.zeros(n_samples),
        "V": np.zeros(n_samples),
        "W": np.zeros(n_samples),
        "TIME": np.arange(n_samples, dtype=float),
    }
    np.save(path, data, allow_pickle=True)

def test_load_sonic_burst_from_dat():
    folderpath = f"{Path(__file__).parent}/testdata"
    files = glob.glob(f"{folderpath}/*.dat")
    name_map = {"u1": "U", "u2": "V", "u3": "W", "Ts": "Ts"}
    sonic = Sonic(files=files, name_map=name_map, fs=32, names=["U", "V", "W", "Ts", "Checksum", "Error"], sep=r"\s+")
    assert sonic.fs == 32
    assert sonic.name_map == name_map
    assert "sonic_sample.dat" in sonic.files[0]

    # Testing burst loading
    burst = sonic.load_burst(0)
    assert "u1" in burst.keys()
    assert "u2" in burst.keys()
    assert "u3" in burst.keys()
    assert "Ts" in burst.keys()
    assert burst["coords"] == "xyz"
    assert burst["u1"].shape[0] == sonic.n_heights

def test_buoyancy_flux():
    folderpath = f"{Path(__file__).parent}/testdata"
    files = glob.glob(f"{folderpath}/*.dat")
    name_map = {"u1": "U", "u2": "V", "u3": "W", "Ts": "Ts"}
    sonic = Sonic(files=files, name_map=name_map, fs=32, names=["U", "V", "W", "Ts", "Checksum", "Error"], sep=r"\s+")
    burst = sonic.load_burst(0)
    B = sonic.buoyancy_flux(burst)
    assert len(B) == sonic.n_heights
    npt.assert_almost_equal(B.item(), 0.001564, decimal=6)  # Regression test

def test_subsample(tmp_path):
    files = []
    for i in range(3):
        p = str(tmp_path / f"burst_{i}.npy")
        _write_sonic_npy(p)
        files.append(p)

    name_map = {"u1": "U", "u2": "V", "u3": "W", "time": "TIME"}
    sonic_full = Sonic(files=files, name_map=name_map, fs=1.0, z=10.0)
    sonic_subsampled = sonic_full.subsample(start_idx=0, end_idx=2)

    assert len(sonic_full.files) == 3
    assert len(sonic_subsampled.files) == 2
    assert eq_except(sonic_subsampled, sonic_full, "files")
