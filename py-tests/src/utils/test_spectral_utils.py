import numpy.testing as npt
import pytest

from pytoast.utils.spectral_utils import *

def test_window_len():
    N = 1024
    window_len = get_window_len(N, num_windows=10)
    assert window_len == int(2048 / 11)

    window_len = get_window_len(N, num_windows=1)
    assert window_len == N

@pytest.mark.parametrize("N,num_windows", [(1024, 8), (2000, 4), (500, 1), (10000, 16)])
def test_window_len_formula(N, num_windows):
    assert get_window_len(N, num_windows) == int(2 * N / (num_windows + 1))

def test_window_len_zero_windows():
    assert get_window_len(1000, 0) == 2000

def test_frequency_range():
    f = np.linspace(0, 10, 100)
    start_idx, end_idx = get_frequency_range(f)
    npt.assert_array_equal([0, len(f)], [start_idx, end_idx])

    start_idx, end_idx = get_frequency_range(f, f_low=0.5)
    assert start_idx == np.nanargmin(np.abs(f - 0.5))
    assert end_idx == len(f)

    start_idx, end_idx = get_frequency_range(f, f_high=0.5)
    assert end_idx == np.nanargmin(np.abs(f - 0.5))
    assert start_idx == 0

    start_idx, end_idx = get_frequency_range(f, -1, -0.5)
    assert start_idx == 0
    assert end_idx == 0

def test_frequency_range_both_bounds_inside():
    f = np.linspace(0, 10, 100)
    start_idx, end_idx = get_frequency_range(f, f_low=3.0, f_high=7.0)
    assert start_idx == np.nanargmin(np.abs(f - 3.0))
    assert end_idx == np.nanargmin(np.abs(f - 7.0))

def test_frequency_range_reversed():
    f = np.linspace(0, 10, 100)
    start_idx, end_idx = get_frequency_range(f, f_low=7.0, f_high=3.0)
    assert start_idx > end_idx
    assert start_idx == np.nanargmin(np.abs(f - 7.0))
    assert end_idx == np.nanargmin(np.abs(f - 3.0))

def test_frequency_range_bounds_outside():
    f = np.linspace(0, 10, 100)
    start_idx, _ = get_frequency_range(f, f_low=15.0)
    assert start_idx == len(f) - 1

    _, end_idx = get_frequency_range(f, f_high=-5.0)
    assert end_idx == 0

class TestPSD:
    def test_psd_sine_defaults(self):
        t = np.linspace(0, 4, 100)
        y = np.sin(2 * np.pi * t)
        f, P = psd(y, fs=25)
        assert np.argmax(P) == np.argmin(np.abs(f - 1))

    def test_psd_errors_without_fs(self):
        with pytest.raises(TypeError, match="fs"):
            psd(np.ones(10))

    def test_psd_sine_one_sided(self):
        t = np.linspace(0, 4, 100)
        y = np.sin(2 * np.pi * t)
        f1, P1 = psd(y, fs=25)
        f2, P2 = psd(y, fs=25, onesided=False)
        npt.assert_array_equal(f1[:-1], f2[:len(f1) - 1])
        npt.assert_array_equal(P1[1:-1], P2[1:len(P1) - 1] * 2)

    def test_psd_parseval(self):
        rng = np.random.default_rng(0)
        N = 8192
        x = rng.standard_normal(N)
        f, P = psd(x, fs=1.0)
        integral = np.sum(P) * (f[1] - f[0])
        npt.assert_allclose(integral, np.var(x), rtol=0.15)

    def test_psd_dc_for_constant(self):
        N = 1024
        f, P = psd(np.ones(N), fs=10)
        assert np.argmax(P) == 0
        assert np.sum(P[:5]) > 0.99 * np.sum(P)

    def test_psd_trend_leakage_without_detrend(self):
        fs = 25
        t = np.arange(0, 100, 1 / fs)
        f0 = 5.0
        y_sine = np.sin(2 * np.pi * f0 * t)
        y_with_trend = y_sine + 0.5 * t

        f, P_sine = psd(y_sine, fs=fs)
        assert np.argmax(P_sine) == np.argmin(np.abs(f - f0))

        f, P_trend = psd(y_with_trend, fs=fs)
        assert np.argmax(P_trend) == np.argmin(np.abs(f - 0))

    def test_psd_window_len_overrides_num_windows(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(4096)
        f, P = psd(x, fs=1.0, window_len=256, num_windows=2)
        assert len(f) == 256 // 2 + 1
        assert len(P) == len(f)

    def test_psd_2d_longest_axis_is_time(self):
        fs = 50
        N = 4096
        t = np.arange(N) / fs
        amps = np.array([1.0, 2.0, 3.0])
        x = np.array([a * np.sin(2 * np.pi * 1.0 * t) for a in amps])
        f, P = psd(x, fs=fs)
        assert P.shape[0] == 3
        assert P.shape[1] == len(f)
        peak_bin = np.argmin(np.abs(f - 1.0))
        for row in range(3):
            assert np.argmax(P[row]) == peak_bin


class TestCSD:
    def test_csd_self_matches_psd(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(4096)
        f_p, P = psd(x, fs=1.0)
        f_c, Pxy = csd(x, x, fs=1.0)
        npt.assert_array_equal(f_p, f_c)
        npt.assert_allclose(np.real(Pxy), P, rtol=1e-10, atol=1e-12)
        npt.assert_allclose(np.imag(Pxy), 0.0, atol=1e-10)

    def test_csd_known_phase_offset(self):
        fs = 100.0
        N = 4096
        t = np.arange(N) / fs
        f0 = 5.0
        phi = np.pi / 4
        x = np.sin(2 * np.pi * f0 * t)
        y = np.sin(2 * np.pi * f0 * t - phi)
        f, Pxy = csd(x, y, fs=fs, window_len=200)
        peak_bin = np.argmin(np.abs(f - f0))
        npt.assert_allclose(np.angle(Pxy[peak_bin]), -phi, atol=0.05)

    def test_csd_uncorrelated_signals(self):
        rng = np.random.default_rng(3)
        N = 8192
        x = rng.standard_normal(N)
        y = rng.standard_normal(N)
        f, Pxy = csd(x, y, fs=1.0)
        _, Pxx = psd(x, fs=1.0)
        _, Pyy = psd(y, fs=1.0)
        coherence = np.abs(Pxy) ** 2 / (Pxx * Pyy)
        assert np.mean(coherence) < 0.5
