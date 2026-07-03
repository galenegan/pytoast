import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from pytoast.utils.base_instrument import BaseInstrument


FS = 16.0
N_HEIGHTS = 4
N_SAMPLES = 32
N_BURSTS = 3


def _make_monolithic(path, burst_dim="burst"):
    rng = np.random.default_rng(42)
    u1 = rng.normal(size=(N_BURSTS, N_HEIGHTS, N_SAMPLES))
    u2 = rng.normal(size=(N_BURSTS, N_HEIGHTS, N_SAMPLES))
    p = rng.normal(size=(N_BURSTS, N_SAMPLES))
    base = np.datetime64("2024-01-01T00:00:00")
    dt = np.timedelta64(int(1e9 / FS), "ns")
    time = base + dt * np.arange(N_BURSTS * N_SAMPLES).reshape(N_BURSTS, N_SAMPLES)
    z = np.linspace(0.5, 2.0, N_HEIGHTS)
    ds = xr.Dataset(
        {
            "u1": ((burst_dim, "height", "sample"), u1),
            "u2": ((burst_dim, "height", "sample"), u2),
            "p": ((burst_dim, "sample"), p),
            "t": ((burst_dim, "sample"), time),
            "z": (("height",), z),
        }
    )
    ds.to_netcdf(path)
    return u1, u2, p, z


def _make_single_burst_nc(path, i):
    rng = np.random.default_rng(i)
    u1 = rng.normal(size=(N_HEIGHTS, N_SAMPLES))
    u2 = rng.normal(size=(N_HEIGHTS, N_SAMPLES))
    p = rng.normal(size=(N_SAMPLES,))
    base = np.datetime64("2024-01-01T00:00:00") + np.timedelta64(i, "h")
    dt = np.timedelta64(int(1e9 / FS), "ns")
    time = base + dt * np.arange(N_SAMPLES)
    z = np.linspace(0.5, 2.0, N_HEIGHTS)
    ds = xr.Dataset(
        {
            "u1": (("height", "sample"), u1),
            "u2": (("height", "sample"), u2),
            "p": (("sample",), p),
            "t": (("sample",), time),
            "z": (("height",), z),
        }
    )
    ds.to_netcdf(path)
    return u1, u2, p


def _make_npy_burst(path, seed=0, n_heights=N_HEIGHTS, extra=None):
    rng = np.random.default_rng(seed)
    u1 = rng.normal(size=(n_heights, N_SAMPLES))
    time = 1.7e9 + np.arange(N_SAMPLES) / FS
    data = {"u1": u1, "u1_cm": u1 * 100.0, "t": time}
    if extra:
        data.update(extra)
    np.save(path, data)
    return u1


NAME_MAP = {"u1": "u1", "u2": "u2", "p": "p", "time": "t", "z": "z"}


def test_monolithic_nc_round_trip(tmp_path):
    path = str(tmp_path / "mono.nc")
    u1, u2, p, z = _make_monolithic(path)

    inst = BaseInstrument(
        files=path,
        name_map=NAME_MAP,
        fs=FS,
        burst_dim="burst",
    )

    assert inst.n_bursts == N_BURSTS
    assert inst.num_samples_per_burst == N_SAMPLES
    npt.assert_array_equal(inst.z, z)
    assert inst.file_type == "nc"

    for i in range(N_BURSTS):
        burst = inst.load_burst(i)
        npt.assert_array_equal(burst["u1"], u1[i])
        npt.assert_array_equal(burst["u2"], u2[i])
        npt.assert_array_equal(burst["p"][0], p[i])


def test_monolithic_requires_single_nc(tmp_path):
    p1 = str(tmp_path / "a.nc")
    p2 = str(tmp_path / "b.nc")
    _make_single_burst_nc(p1, 0)
    _make_single_burst_nc(p2, 1)
    with pytest.raises(ValueError, match="single .nc"):
        BaseInstrument(files=[p1, p2], name_map=NAME_MAP, fs=FS, burst_dim="burst")


def test_monolithic_rejects_bad_dim(tmp_path):
    path = str(tmp_path / "mono.nc")
    _make_monolithic(path)
    with pytest.raises(ValueError, match="not found"):
        BaseInstrument(files=path, name_map=NAME_MAP, fs=FS, burst_dim="nope")


def test_multi_file_nc_one_burst_per_file(tmp_path):
    paths = []
    expected = []
    for i in range(N_BURSTS):
        p = str(tmp_path / f"burst_{i}.nc")
        paths.append(p)
        expected.append(_make_single_burst_nc(p, i))

    inst = BaseInstrument(files=paths, name_map=NAME_MAP, fs=FS)

    assert inst.n_bursts == N_BURSTS
    assert inst.file_type == "nc"
    for i, (u1, u2, p) in enumerate(expected):
        burst = inst.load_burst(i)
        npt.assert_array_equal(burst["u1"], u1)
        npt.assert_array_equal(burst["u2"], u2)
        npt.assert_array_equal(burst["p"][0], p)


def test_to_dataset_round_trip(tmp_path):
    path = str(tmp_path / "mono.nc")
    _make_monolithic(path)
    inst = BaseInstrument(files=path, name_map=NAME_MAP, fs=FS, burst_dim="burst")

    results = [{"eps": float(i), "u_mean": np.arange(N_HEIGHTS, dtype=float) + i} for i in range(N_BURSTS)]
    burst_times = np.arange(N_BURSTS).astype("datetime64[s]")

    out = str(tmp_path / "out.nc")
    inst.to_netcdf(out, results, burst_times)

    loaded = xr.open_dataset(out)
    assert loaded["eps"].dims == ("burst_time",)
    assert loaded["u_mean"].dims == ("burst_time", "z")
    npt.assert_array_equal(loaded["eps"].values, np.arange(N_BURSTS, dtype=float))
    assert loaded.attrs["instrument"] == "BaseInstrument"
    assert loaded.attrs["fs"] == FS
    loaded.close()


def test_lambda_name_map_npy(tmp_path):
    path = str(tmp_path / "burst_0.npy")
    _make_npy_burst(path)

    inst_str = BaseInstrument(files=[path], name_map={"u1": "u1", "time": "t"}, fs=FS)
    inst_lam = BaseInstrument(files=[path], name_map={"u1": lambda d: d["u1_cm"] / 100.0, "time": "t"}, fs=FS)

    burst_str = inst_str.load_burst(0)
    burst_lam = inst_lam.load_burst(0)

    # Lambda-extracted variables get the same 2-D shape normalization as string-keyed ones
    assert burst_lam["u1"].ndim == 2
    npt.assert_allclose(burst_lam["u1"], burst_str["u1"])
    npt.assert_array_equal(burst_lam["time"], burst_str["time"])


def test_lambda_name_map_nc_plain_array(tmp_path):
    path = str(tmp_path / "burst_0.nc")
    u1, _, _ = _make_single_burst_nc(path, 0)

    # A lambda returning a plain numpy array (not a DataArray) must work for .nc files,
    # including during __init__ when z is inferred from the first non-time variable
    name_map = {"u1": lambda ds: ds["u1"].values * 2.0, "time": "t"}
    inst = BaseInstrument(files=[path], name_map=name_map, fs=FS)

    burst = inst.load_burst(0)
    assert burst["u1"].ndim == 2
    npt.assert_allclose(burst["u1"], u1 * 2.0)


def test_transformation_matrix_not_transposed(tmp_path):
    path = str(tmp_path / "burst_0.npy")
    matrix = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    _make_npy_burst(path, n_heights=3, extra={"tm": matrix})

    # Matrix key listed first: z and num_samples inference must skip it
    name_map = {"transformation_matrix": "tm", "u1": "u1", "time": "t"}
    inst = BaseInstrument(files=[path], name_map=name_map, fs=FS)
    assert inst.n_heights == 3
    assert inst.num_samples_per_burst == N_SAMPLES

    # Square (3, 3) matrix with n_heights == 3 must not be transposed by the time-last heuristic
    burst = inst.load_burst(0)
    npt.assert_array_equal(burst["transformation_matrix"], matrix)

    # Same key selection when "time" is absent and fs is given
    inst_no_time = BaseInstrument(files=[path], name_map={"transformation_matrix": "tm", "u1": "u1"}, fs=FS)
    assert inst_no_time.num_samples_per_burst == N_SAMPLES


def test_transformation_matrices_3d_shape_preserved(tmp_path):
    path = str(tmp_path / "burst_0.npy")
    rng = np.random.default_rng(7)
    matrices = rng.normal(size=(3, 3, 3))  # one (3, 3) matrix per instrument in the stack
    _make_npy_burst(path, n_heights=3, extra={"tms": matrices})

    name_map = {"u1": "u1", "time": "t", "transformation_matrices": "tms"}
    inst = BaseInstrument(files=[path], name_map=name_map, fs=FS)

    burst = inst.load_burst(0)
    npt.assert_array_equal(burst["transformation_matrices"], matrices)


def test_to_dataset_with_freq(tmp_path):
    path = str(tmp_path / "mono.nc")
    _make_monolithic(path)
    inst = BaseInstrument(files=path, name_map=NAME_MAP, fs=FS, burst_dim="burst")

    freq = np.linspace(0.1, 1.0, 5)
    results = [{"Suu": np.ones((N_HEIGHTS, 5)) * i} for i in range(N_BURSTS)]
    burst_times = np.arange(N_BURSTS).astype("datetime64[s]")

    ds = inst.to_dataset(results, burst_times, freq=freq)
    assert ds["Suu"].dims == ("burst_time", "z", "freq")
    assert ds["Suu"].shape == (N_BURSTS, N_HEIGHTS, 5)
    npt.assert_array_equal(ds["freq"].values, freq)
