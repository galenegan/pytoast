import numpy as np
import numpy.testing as npt
import pytest
from pathlib import Path
import glob
import pytoast.utils.sea_thermo as sea_thermo
from pytoast.ocean.ctd import CTD
from pytoast.utils.base_instrument import ZConvention
from testhelpers.stub_utils import eq_except, make_ctd

NAME_MAP = {"sp": "PSAL", "t": "TEMP", "p": "PRES", "time": "TIME"}
ALL_DERIVED = {"sa", "ct", "rho", "sigma0", "alpha", "beta", "sound_speed", "t_freezing", "cp", "nu", "z"}
ctd = make_ctd(NAME_MAP)


def _burst(n_heights=1, n_samples=4, keys=("sp", "t", "p")):
    shape = (n_heights, n_samples)
    values = {
        "sp": np.full(shape, 35.0),
        "t": np.full(shape, 15.0),
        "p": np.full(shape, 100.0),
    }
    return {k: values[k] for k in keys}


def test_sp_only_adds_only_sa():
    out = ctd.derive(_burst(keys=("sp",)))
    assert "sa" in out
    assert ALL_DERIVED - {"sa"} & out.keys() == set()


def test_full_inputs_single_height_no_N2():
    out = ctd.derive(_burst(n_heights=1))
    assert ALL_DERIVED <= out.keys()
    assert "N2" not in out


def test_full_inputs_multi_height_adds_N2():
    n_heights, n_samples = 3, 4
    out = make_ctd(NAME_MAP, n_heights=n_heights).derive(_burst(n_heights=n_heights, n_samples=n_samples))
    assert "N2" in out
    assert out["N2"].shape == (n_heights - 1, n_samples)


def test_p_only_adds_z_but_no_thermo():
    out = ctd.derive(_burst(keys=("t", "p")))
    assert "z" in out
    assert not (ALL_DERIVED - {"z"}) & out.keys()


def test_lat_forwarded_to_depth_from_pressure():
    burst = _burst(keys=("p",))
    burst["lat"] = 45.0
    out = ctd.derive(burst)
    expected = sea_thermo.depth_from_pressure(burst["p"], 45.0)
    npt.assert_allclose(out["z"], expected)
    assert not np.allclose(out["z"], sea_thermo.depth_from_pressure(burst["p"]))


def test_mutates_in_place():
    burst = _burst()
    out = ctd.derive(burst)
    assert out is burst


def test_empty_burst_returns_empty():
    out = ctd.derive({})
    assert out == {}


def _write_ctd_npy(path, n_samples=8):
    data = {
        "PSAL": np.full(n_samples, 35.0),
        "TEMP": np.full(n_samples, 15.0),
        "PRES": np.full(n_samples, 100.0),
        "TIME": np.arange(n_samples, dtype=float),
    }
    np.save(path, data, allow_pickle=True)


def test_subsample(tmp_path):
    files = []
    for i in range(3):
        p = str(tmp_path / f"burst_{i}.npy")
        _write_ctd_npy(p)
        files.append(p)

    ctd_full = CTD(files=files, name_map={"sp": "PSAL", "t": "TEMP", "p": "PRES", "time": "TIME"}, fs=1.0, z=1.0)
    ctd_subsampled = ctd_full.subsample(start_idx=0, end_idx=2)

    assert len(ctd_full.files) == 3
    assert len(ctd_subsampled.files) == 2
    assert eq_except(ctd_subsampled, ctd_full, "files")


def test_load_and_derive_from_mat():
    folder = f"{Path(__file__).parent}/testdata"
    files = glob.glob(f"{folder}/*.mat")
    ctd = CTD(
        files=files,
        name_map={"sp": "sal", "t": "temp", "p": "depth", "time": "time"},
        fs=1.0,
        z=[2.5, 3.3],
        z_convention="depth",
    )

    # Loading as a burst
    burst = ctd.load_burst(0)
    data = ctd.derive(burst)
    assert all([key in data.keys() for key in ALL_DERIVED])
    assert burst["N2"].shape[0] == 1


###############################
# CTD.validate_inputs
###############################


def _ctd_valid_kwargs(tmp_path):
    f = tmp_path / "fake.mat"
    f.write_bytes(b"")
    return {
        "files": [str(f)],
        "name_map": {"sp": "PSAL", "t": "TEMP", "p": "PRES"},
        "fs": 1.0,
        "z": [1.0],
        "z_convention": ZConvention.DEPTH,
    }


def test_ctd_validate_inputs_happy_path(tmp_path):
    kw = _ctd_valid_kwargs(tmp_path)
    assert CTD.validate_inputs(**kw) is None


def test_ctd_validate_inputs_files_not_list_raises(tmp_path):
    kw = _ctd_valid_kwargs(tmp_path)
    kw["files"] = (kw["files"][0],)
    with pytest.raises(TypeError):
        CTD.validate_inputs(**kw)


def test_ctd_validate_inputs_bad_extension_raises(tmp_path):
    f = tmp_path / "fake.zip"
    f.write_bytes(b"")
    kw = _ctd_valid_kwargs(tmp_path)
    kw["files"] = [str(f)]
    with pytest.raises(ValueError):
        CTD.validate_inputs(**kw)


def test_ctd_validate_inputs_missing_file_raises(tmp_path):
    kw = _ctd_valid_kwargs(tmp_path)
    kw["files"] = [str(tmp_path / "missing.mat")]
    with pytest.raises(FileNotFoundError):
        CTD.validate_inputs(**kw)


def test_ctd_validate_inputs_name_map_not_dict_raises(tmp_path):
    kw = _ctd_valid_kwargs(tmp_path)
    kw["name_map"] = "not a dict"
    with pytest.raises(TypeError):
        CTD.validate_inputs(**kw)


def test_ctd_validate_inputs_no_time_and_no_fs_raises(tmp_path):
    kw = _ctd_valid_kwargs(tmp_path)
    kw["fs"] = None
    with pytest.raises(ValueError):
        CTD.validate_inputs(**kw)


def test_ctd_validate_inputs_z_wrong_type_raises(tmp_path):
    kw = _ctd_valid_kwargs(tmp_path)
    kw["z"] = "bad"
    with pytest.raises(TypeError):
        CTD.validate_inputs(**kw)


def test_ctd_validate_inputs_z_list_non_numeric_raises(tmp_path):
    kw = _ctd_valid_kwargs(tmp_path)
    kw["z"] = [1.0, None]
    with pytest.raises(TypeError):
        CTD.validate_inputs(**kw)


def test_ctd_validate_inputs_fs_wrong_type_raises(tmp_path):
    kw = _ctd_valid_kwargs(tmp_path)
    kw["fs"] = "1"
    with pytest.raises(TypeError):
        CTD.validate_inputs(**kw)


def test_ctd_validate_inputs_data_keys_wrong_type_raises(tmp_path):
    kw = _ctd_valid_kwargs(tmp_path)
    kw["data_keys"] = 3
    with pytest.raises(TypeError):
        CTD.validate_inputs(**kw)


def test_ctd_validate_inputs_bad_z_convention_raises(tmp_path):
    kw = _ctd_valid_kwargs(tmp_path)
    kw["z_convention"] = ZConvention.MAS
    with pytest.raises(ValueError):
        CTD.validate_inputs(**kw)
