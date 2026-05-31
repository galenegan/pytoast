import numpy.testing as npt
import numpy as np
import pytest

from utils.wave_utils import get_wavenumber, get_cg, wave_stats, jones_monismith_correction
from testhelpers.synth_utils import generate_wave_turb_burst, pierson_moskowitz

class TestDispersion:
    def test_dispersion_relation_deep_limit(self):
        omega = 2 * np.pi / 10
        h = 1000
        k = get_wavenumber(omega, h)
        k_deep_water = omega**2 / 9.81
        npt.assert_equal(k, k_deep_water)


    def test_dispersion_relation_shallow_limit(self):
        omega = 2 * np.pi / 2000
        h = 10
        k = get_wavenumber(omega, h)
        k_shallow_water = omega / np.sqrt(9.81 * h)
        npt.assert_almost_equal(k, k_shallow_water, decimal=4)

    def test_group_velocity_shallow_limit(self):
        omega = 2 * np.pi / 2000
        h = 10
        k = get_wavenumber(omega, h)
        cg = get_cg(k, h)
        npt.assert_almost_equal(cg, np.sqrt(9.81 * h), decimal=4)

    def test_group_velocity_deep_limit(self):
        omega = 2 * np.pi / 10
        h = 1000
        k = get_wavenumber(omega, h)
        cg = get_cg(k, h)
        npt.assert_almost_equal(cg, 9.81 / omega / 2, decimal=4)

class TestWaveStats:
    @pytest.fixture
    def wave_data(self):
        P_etaeta, eta, u, w, p = pierson_moskowitz(
            Hs=2,
            fp=0.1,
            h=10,
            z=-1,
            fs=16,
            duration_s=600,
            seed=0
        )
        return P_etaeta, eta, u, w, p

    def test_wave_stats_bulk_params(self, wave_data):
        P_etaeta, eta, u, w, p = wave_data
        df = 16 / len(eta)
        v = np.zeros_like(u)
        out = wave_stats(
            u, v, p, fs=16, mab=9, sea_correction=True
        )

        # Consistency between all the Hsig parameters
        npt.assert_almost_equal(4 * np.sqrt(np.sum(P_etaeta * df)), out["Hsig_all"], decimal=2)
        npt.assert_almost_equal(4 * np.std(eta), out["Hsig_all"], decimal=2)
        npt.assert_almost_equal(out["Hsig_all"], 2, decimal=2)

        # Peak period
        npt.assert_almost_equal(out["fp_all"], 0.1, decimal=2)

        # Directions and spread
        npt.assert_almost_equal(out["dir1_all"], 0, decimal=4)
        npt.assert_almost_equal(out["dir2_all"], 0, decimal=4)
        npt.assert_almost_equal(out["spread2_all"], 0, decimal=4)