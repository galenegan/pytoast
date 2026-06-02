import warnings

import numpy.testing as npt
import numpy as np
import pytest

from pytoast.utils.wave_utils import get_wavenumber, get_cg, jones_monismith_correction, wave_stats
from testhelpers.synth_utils import pierson_moskowitz


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

    def test_get_cg_zero_wavenumber_returns_zero_no_warning(self):
        k = np.array([0.0, 0.5, 1.0, 2.0])
        h = 10.0
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            cg = get_cg(k, h)
        assert cg[0] == 0.0
        cg_ref = get_cg(k[1:], h)
        npt.assert_allclose(cg[1:], cg_ref, rtol=1e-12)

    def test_get_wavenumber_no_overflow_at_high_frequency(self):
        omega = np.array([0.0, 2 * np.pi * 5.0, 2 * np.pi * 10.0])
        h = 1000.0
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            k = get_wavenumber(omega, h)
        assert k[0] == 0.0
        npt.assert_allclose(k[1:], omega[1:] ** 2 / 9.81, rtol=1e-6)


class TestWaveStats:
    @pytest.fixture
    def wave_data(self):
        P_etaeta, eta, u, w, p = pierson_moskowitz(Hs=2, fp=0.1, h=10, z=-1, fs=16, duration_s=600, seed=0)
        return P_etaeta, eta, u, w, p

    def test_wave_stats_bulk_params(self, wave_data):
        P_etaeta, eta, u, w, p = wave_data
        df = 16 / len(eta)
        v = np.zeros_like(u)
        out = wave_stats(u, v, p, fs=16, mab=9, sea_correction=True)

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

    def test_wave_stats_spectral(self, wave_data):
        P_etaeta, eta, u, w, p = wave_data
        df = 16 / len(eta)
        v = np.zeros_like(u)
        out = wave_stats(u, v, p, fs=16, mab=9, sea_correction=True)

        # Just verifying some shapes
        shape_f = out["f"].shape
        assert out["P_uu"].shape == shape_f
        assert out["P_pp"].shape == shape_f
        assert out["P_uv"].shape == shape_f
        assert out["P_pv"].shape == shape_f

    def test_sea_correction(self, wave_data):
        P_etaeta, eta, u, w, p = wave_data
        df = 16 / len(eta)
        v = np.zeros_like(u)
        out_corrected = wave_stats(u, v, p, fs=16, mab=2, sea_correction=True)

        out_uncorrected = wave_stats(u, v, p, fs=16, mab=2, sea_correction=False)

        # Bulk statistics integrated below cutoff should be similar
        npt.assert_almost_equal(out_corrected["Hsig_all"], out_uncorrected["Hsig_all"], decimal=2)
        npt.assert_almost_equal(out_corrected["Tm_all"], out_uncorrected["Tm_all"], decimal=1)

        # Corrected P_etaeta should not blow up
        assert all(out_corrected["P_etaeta"] < 10)
        assert np.max(out_corrected["P_etaeta"]) < np.max(out_uncorrected["P_etaeta"])


class TestJonesMonismithCorrection:
    @staticmethod
    def _build_pressure_spectrum(f, fp, noise_floor):
        """Synthetic pressure spectrum: Gaussian peak at fp on a flat noise floor."""
        peak = np.exp(-((f - fp) ** 2) / (2 * (0.02**2)))
        return peak + noise_floor

    def test_below_cutoff_unchanged(self):
        f = np.linspace(0, 1.0, 501)
        fp = 0.12
        S_pp = self._build_pressure_spectrum(f, fp=fp, noise_floor=1e-3)
        S_etaeta = S_pp * 4.0

        S_corrected = jones_monismith_correction(S_etaeta, S_pp, f, f_cutoff=0.5)

        # The replacement is m * f^-4 above some cutoff index. At and below that
        # index, the output must equal the input.
        diff = np.where(S_corrected != S_etaeta)[0]
        assert diff.size > 0, "expected replacement above some cutoff"
        cutoff_idx = diff[0]
        assert f[cutoff_idx] >= 1.1 * fp - 1e-9
        npt.assert_array_equal(S_corrected[:cutoff_idx], S_etaeta[:cutoff_idx])

    def test_high_frequency_tail_follows_f_minus_4(self):
        f = np.linspace(0, 1.0, 501)
        fp = 0.12
        S_pp = self._build_pressure_spectrum(f, fp=fp, noise_floor=1e-3)
        S_etaeta = S_pp * 4.0

        S_corrected = jones_monismith_correction(S_etaeta, S_pp, f, f_cutoff=0.5)

        # Identify the cutoff index (first index where output differs from input).
        cutoff_idx = int(np.where(S_corrected != S_etaeta)[0][0])
        m = S_corrected[cutoff_idx] * f[cutoff_idx] ** 4

        # Above the cutoff, S_corrected must equal m * f^-4 exactly (skip f=0).
        expected_tail = m * f[cutoff_idx:] ** (-4)
        npt.assert_allclose(S_corrected[cutoff_idx:], expected_tail, rtol=1e-12)

        # Tail must be monotonically non-increasing.
        assert np.all(np.diff(S_corrected[cutoff_idx:]) <= 0)

    def test_shape_preserved_and_input_not_mutated(self):
        f = np.linspace(0, 1.0, 257)
        fp = 0.15
        S_pp = self._build_pressure_spectrum(f, fp=fp, noise_floor=1e-3)
        S_etaeta = S_pp * 2.0
        S_etaeta_orig = S_etaeta.copy()

        S_corrected = jones_monismith_correction(S_etaeta, S_pp, f)

        assert S_corrected.shape == S_etaeta.shape
        npt.assert_array_equal(S_etaeta, S_etaeta_orig)  # input not mutated
