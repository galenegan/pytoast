import pytest
import numpy as np
import numpy.testing as npt
from pytoast.utils.despike_utils import threshold, goring_nikora, recursive_gaussian
from pytoast.utils.interp_utils import naninterp, interp_rows


@pytest.fixture
def noisy_data():
    """1-D Gaussian noise of length 1000 with 10 spike samples (value=100).

    Returns
    -------
    out : np.ndarray
        1-D array, shape (1000,). The 10 spike indices are deterministic
        (seeded via `np.random.default_rng(42)` for the noise, separate seeded
        choice for the spikes).
    og_max, og_min : float
        Max/min of the pre-spike noise.
    """
    rng = np.random.default_rng(42)
    out = rng.standard_normal(1000)
    og_max = float(np.max(out))
    og_min = float(np.min(out))
    spike_rng = np.random.default_rng(7)
    bad_indices = spike_rng.choice(np.arange(1000), 10, replace=False)
    out[bad_indices] = 100.0
    return out, og_max, og_min


class TestDespike:
    def test_threshold_despike(self, noisy_data):
        data, og_max, og_min = noisy_data
        filtered = threshold(np.atleast_2d(data), threshold_min=-20, threshold_max=20)
        assert np.all(filtered <= og_max)
        assert np.all(filtered >= og_min)

    def test_goring_nikora(self, noisy_data):
        data, og_max, og_min = noisy_data
        filtered = goring_nikora(np.atleast_2d(data), remaining_spikes=0, max_iter=10)
        assert np.all(filtered <= og_max)
        assert np.all(filtered >= og_min)

    def test_recursive_gaussian(self, noisy_data):
        data, og_max, og_min = noisy_data
        filtered = recursive_gaussian(np.atleast_2d(data), alpha=3, max_iter=10)
        assert np.all(filtered <= og_max)
        assert np.all(filtered >= og_min)


class TestInterp:
    def test_naninterp(self, noisy_data):
        data, _, _ = noisy_data
        data[data > 10] = np.nan
        out = naninterp(data)
        assert np.all(np.isfinite(out))

    def test_naninterp_rejects_2d(self):
        with pytest.raises(ValueError, match="1-D"):
            naninterp(np.zeros((2, 5)))

    def test_interp_rows_all_nans(self):
        data = np.full((10, 10), np.nan)
        data_out = interp_rows(data)
        npt.assert_array_equal(data_out, data)

    def test_interp_rows_2d(self, noisy_data):
        data, _, _ = noisy_data
        data[data > 10] = np.nan
        data_2d = np.stack([data, data], axis=0)
        out = interp_rows(data_2d)
        assert out.shape == data_2d.shape
        assert np.all(np.isfinite(out))


class TestShapeContract:
    """Verify the (N-D with time on the last axis) contract for all despike /
    interp entry points."""

    def _make_signal_with_spike(self, shape):
        """Build a deterministic array of the given shape with one spike in
        each (..., time) row at index 5."""
        n = shape[-1]
        rng = np.random.default_rng(0)
        a = rng.standard_normal(shape)
        a[..., 5] = 100.0
        return a

    @pytest.mark.parametrize("shape", [(64,), (3, 64), (2, 3, 64)])
    @pytest.mark.parametrize(
        "fn,kwargs",
        [
            (threshold, {"threshold_min": -10, "threshold_max": 10}),
            (goring_nikora, {"remaining_spikes": 0, "max_iter": 5}),
            (recursive_gaussian, {"alpha": 3, "max_iter": 5}),
        ],
    )
    def test_despike_shape_roundtrip(self, fn, kwargs, shape):
        a = self._make_signal_with_spike(shape)
        out = fn(a, **kwargs)
        assert out.shape == shape, f"{fn.__name__}: shape {out.shape} != {shape}"
        assert out.dtype == np.float64
        # Spike samples must be pulled back inside +/- 10.
        assert np.all(np.abs(out[..., 5]) < 10.0)

    @pytest.mark.parametrize("shape", [(8,), (2, 8), (2, 3, 8)])
    def test_interp_rows_shape_roundtrip(self, shape):
        a = np.zeros(shape)
        a[..., 3] = np.nan
        a[..., 4] = np.nan
        # Surrounding values: 0; interpolation should yield 0.
        out = interp_rows(a)
        assert out.shape == shape
        assert out.dtype == np.float64
        assert np.all(np.isfinite(out))

    def test_int_input_coerced(self):
        a = np.array([[0, 0, 1000, 0, 0]], dtype=np.int64)
        out = threshold(a, threshold_min=-10, threshold_max=10)
        assert out.dtype == np.float64
        # Spike replaced by interpolation between zeros -> 0
        assert out[0, 2] == 0.0

    @pytest.mark.parametrize("fn", [threshold, goring_nikora, recursive_gaussian, interp_rows])
    def test_zero_d_rejected(self, fn):
        with pytest.raises(ValueError):
            fn(np.array(1.0))

    def test_input_not_mutated(self):
        a = np.array([[0.0, 0.0, 1000.0, 0.0, 0.0]])
        a_orig = a.copy()
        _ = threshold(a, threshold_min=-10, threshold_max=10)
        npt.assert_array_equal(a, a_orig)
