import numpy as np
import pandas as pd
import numpy.testing as npt
import pytest

from pytoast.utils.io_utils import results_to_dataset
from pytoast.utils.base_instrument import BaseInstrument


N_BURSTS = 4
N_HEIGHTS = 6
N_FREQ = 5


@pytest.fixture
def burst_times():
    return np.arange(N_BURSTS).astype("datetime64[s]")


@pytest.fixture
def z():
    return np.linspace(0.5, 3.0, N_HEIGHTS)


@pytest.fixture
def freq():
    return np.linspace(0.1, 2.0, N_FREQ)


def test_scalar_per_burst(burst_times, z):
    results = [{"eps": float(i)} for i in range(N_BURSTS)]
    ds = results_to_dataset(results, burst_times, z=z)
    assert ds["eps"].dims == ("burst_time",)
    assert ds["eps"].shape == (N_BURSTS,)
    npt.assert_array_equal(ds["eps"].values, np.arange(N_BURSTS, dtype=float))


def test_height_per_burst_attaches_z(burst_times, z):
    rng = np.random.default_rng(0)
    results = [{"u_mean": rng.normal(size=N_HEIGHTS)} for _ in range(N_BURSTS)]
    ds = results_to_dataset(results, burst_times, z=z)
    assert ds["u_mean"].dims == ("burst_time", "z")
    assert ds["u_mean"].shape == (N_BURSTS, N_HEIGHTS)
    npt.assert_array_equal(ds["z"].values, z)


def test_staggered_height_becomes_z_mid(burst_times, z):
    results = [{"uw": np.ones(N_HEIGHTS - 1) * i} for i in range(N_BURSTS)]
    ds = results_to_dataset(results, burst_times, z=z)
    assert ds["uw"].dims == ("burst_time", "z_mid")
    assert ds["uw"].shape == (N_BURSTS, N_HEIGHTS - 1)
    npt.assert_allclose(ds["z_mid"].values, 0.5 * (z[:-1] + z[1:]))


def test_freq_per_burst(burst_times, freq):
    results = [{"Suu": np.ones(N_FREQ) * i} for i in range(N_BURSTS)]
    ds = results_to_dataset(results, burst_times, freq=freq)
    assert ds["Suu"].dims == ("burst_time", "freq")
    assert ds["Suu"].shape == (N_BURSTS, N_FREQ)
    npt.assert_array_equal(ds["freq"].values, freq)


def test_height_freq_2d(burst_times, z, freq):
    rng = np.random.default_rng(1)
    results = [{"Suw": rng.normal(size=(N_HEIGHTS, N_FREQ))} for _ in range(N_BURSTS)]
    ds = results_to_dataset(results, burst_times, z=z, freq=freq)
    assert ds["Suw"].dims == ("burst_time", "z", "freq")
    assert ds["Suw"].shape == (N_BURSTS, N_HEIGHTS, N_FREQ)


def test_missing_key_fills_with_nan(burst_times, z):
    results = [
        {"eps": 1.0, "u_mean": np.arange(N_HEIGHTS, dtype=float)},
        {"eps": 2.0},
        {"u_mean": np.arange(N_HEIGHTS, dtype=float) + 10.0},
        {"eps": 4.0, "u_mean": np.arange(N_HEIGHTS, dtype=float) + 20.0},
    ]
    ds = results_to_dataset(results, burst_times, z=z)
    npt.assert_array_equal(ds["eps"].values, np.array([1.0, 2.0, np.nan, 4.0]))
    assert np.all(np.isnan(ds["u_mean"].values[1]))
    npt.assert_array_equal(ds["u_mean"].values[0], np.arange(N_HEIGHTS, dtype=float))


def test_ambiguous_1d_shape_raises(burst_times):
    z_eq = np.arange(N_FREQ, dtype=float)
    freq_eq = np.arange(N_FREQ, dtype=float)
    results = [{"ambig": np.zeros(N_FREQ)} for _ in range(N_BURSTS)]
    with pytest.raises(ValueError, match="Ambiguous"):
        results_to_dataset(results, burst_times, z=z_eq, freq=freq_eq)


def test_coord_reserved_keys_ignored(burst_times, z):
    results = [{"time": np.arange(3), "z": z, "eps": float(i)} for i in range(N_BURSTS)]
    ds = results_to_dataset(results, burst_times, z=z)
    assert "time" not in ds.data_vars
    assert "z" not in ds.data_vars
    assert "eps" in ds.data_vars


def test_shape_mismatch_raises(burst_times, z):
    results = [
        {"u_mean": np.zeros(N_HEIGHTS)},
        {"u_mean": np.zeros(N_HEIGHTS + 1)},
    ] + [{"u_mean": np.zeros(N_HEIGHTS)} for _ in range(N_BURSTS - 2)]
    with pytest.raises(ValueError, match="Shape mismatch"):
        results_to_dataset(results, burst_times, z=z)


def test_empty_results_raises():
    with pytest.raises(ValueError, match="non-empty"):
        results_to_dataset([], np.array([0]))


def test_burst_times_length_mismatch_raises(z):
    results = [{"eps": 1.0}, {"eps": 2.0}]
    with pytest.raises(ValueError, match="burst_times"):
        results_to_dataset(results, np.array([0, 1, 2]), z=z)


def test_attrs_attached(burst_times, z):
    results = [{"eps": 1.0} for _ in range(N_BURSTS)]
    ds = results_to_dataset(results, burst_times, z=z, attrs={"instrument": "ADV", "fs": 16.0})
    assert ds.attrs["instrument"] == "ADV"
    assert ds.attrs["fs"] == 16.0


def test_unknown_1d_length_uses_keyed_dim(burst_times):
    results = [{"mode_amp": np.zeros(7)} for _ in range(N_BURSTS)]
    ds = results_to_dataset(results, burst_times)
    assert ds["mode_amp"].dims == ("burst_time", "mode_amp_dim")
    assert ds["mode_amp"].shape == (N_BURSTS, 7)


def test_time_detection():
    time = pd.Timestamp.now()
    time_format = BaseInstrument.detect_time_format(time)
    assert time_format == "datetime"

    time = 8e5
    time_format = BaseInstrument.detect_time_format(time)
    assert time_format == "matlab"

    time = "2026-05-31T00:00:00Z"
    time_format = BaseInstrument.detect_time_format(time)
    assert time_format == "datestring"

    time = pd.Timestamp.now().timestamp()
    time_format = BaseInstrument.detect_time_format(time)
    assert time_format == "epoch"
