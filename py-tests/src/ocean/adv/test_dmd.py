import types
import numpy as np
import numpy.testing as npt
import scipy.io as sio
from pathlib import Path

from ocean.adv import ADV

TEST_DATA_DIR = f"{Path(__file__).parent}/testdata/dmd"
SYNTH_DATA_PATH = f"{TEST_DATA_DIR}/SynthData.mat"
OUTPUT_PATH = f"{TEST_DATA_DIR}/output.mat"


def _load_reference_data():
    syn = sio.loadmat(SYNTH_DATA_PATH)
    out = sio.loadmat(OUTPUT_PATH)
    u_tot = syn["u_tot"].ravel()
    fs = float(syn["fs"].ravel()[0])
    u_wave_ref = out["u_wave_dmd"].ravel()
    u_turb_ref = out["u_turb_dmd"].ravel()
    return u_tot, fs, u_wave_ref, u_turb_ref


def _make_adv(fs, n_heights=1):
    """Return a minimal namespace that satisfies ADV.dmd's attribute requirements."""
    return types.SimpleNamespace(fs=fs, n_heights=n_heights)


class TestDMD:
    def test_wave_matches_matlab_reference(self):
        """Wave reconstruction should match MATLAB output to machine precision."""
        u_tot, fs, u_wave_ref, _ = _load_reference_data()
        adv = _make_adv(fs)

        result = ADV.dmd(
            adv,
            u=u_tot,
            v=u_tot,
            w=u_tot,
            f_wave_low=0.117,
            f_wave_high=0.6,
            rank_truncation=131,
            time_delay_size=1500,
            return_time_series=True,
        )

        npt.assert_allclose(result["u_wave"], u_wave_ref, atol=1e-10)

    def test_turb_matches_matlab_reference(self):
        """Turbulence reconstruction should match MATLAB output to machine precision."""
        u_tot, fs, _, u_turb_ref = _load_reference_data()
        adv = _make_adv(fs)

        result = ADV.dmd(
            adv,
            u=u_tot,
            v=u_tot,
            w=u_tot,
            f_wave_low=0.117,
            f_wave_high=0.6,
            rank_truncation=131,
            time_delay_size=1500,
            return_time_series=True,
        )

        npt.assert_allclose(result["u_turb"], u_turb_ref, atol=1e-10)

    def test_reconstruction_identity(self):
        """Wave + turbulence should equal raw signal (minus last sample)."""
        u_tot, fs, _, _ = _load_reference_data()
        raw = u_tot - np.mean(u_tot)
        adv = _make_adv(fs)

        result = ADV.dmd(
            adv,
            u=u_tot,
            v=u_tot,
            w=u_tot,
            f_wave_low=0.117,
            f_wave_high=0.6,
            rank_truncation=131,
            time_delay_size=1500,
            return_time_series=True,
        )

        reconstructed = result["u_wave"] + result["u_turb"]
        npt.assert_allclose(reconstructed, raw[:-1], atol=1e-10)

    def test_output_shape(self):
        """Output time series should be length N-1."""
        u_tot, fs, _, _ = _load_reference_data()
        N = len(u_tot)
        adv = _make_adv(fs)

        result = ADV.dmd(
            adv,
            u=u_tot,
            v=u_tot,
            w=u_tot,
            f_wave_low=0.117,
            f_wave_high=0.6,
            rank_truncation=131,
            time_delay_size=1500,
            return_time_series=True,
        )

        assert result["u_wave"].shape == (N - 1,)
        assert result["u_turb"].shape == (N - 1,)

    def test_default_output_keys(self):
        """Default output should contain only Reynolds stress keys."""
        u_tot, fs, _, _ = _load_reference_data()
        adv = _make_adv(fs)

        result = ADV.dmd(
            adv, u=u_tot, v=u_tot, w=u_tot, f_wave_low=0.117, f_wave_high=0.6, rank_truncation=131, time_delay_size=1500
        )

        expected_keys = {
            "uu_turb",
            "vv_turb",
            "ww_turb",
            "uw_turb",
            "vw_turb",
            "uv_turb",
            "uu_wave",
            "vv_wave",
            "ww_wave",
            "uw_wave",
            "vw_wave",
            "uv_wave",
        }
        assert set(result.keys()) == expected_keys

    def test_float_rank_truncation(self):
        """Float rank_truncation should select modes by relative singular value magnitude."""
        u_tot, fs, _, _ = _load_reference_data()
        adv = _make_adv(fs)

        result = ADV.dmd(
            adv,
            u=u_tot,
            v=u_tot,
            w=u_tot,
            f_wave_low=0.117,
            f_wave_high=0.6,
            rank_truncation=0.05,
            time_delay_size=1500,
            return_time_series=True,
        )

        # Should still produce the right output shape
        assert result["u_wave"].shape == (len(u_tot) - 1,)
        assert result["u_turb"].shape == (len(u_tot) - 1,)
