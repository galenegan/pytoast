import glob
import os
import numpy as np
import pytest

from ocean.adv import ADV
from utils.base_instrument import ZConvention
from testhelpers.stub_utils import eq_except


def test_mat_list():
    """Test loading a list of .mat files."""
    name_map = {"u1": "E", "u2": "N", "u3": "w", "p": "P2", "time": "dn"}

    # Get test data files
    test_dir = os.path.dirname(__file__)
    files = glob.glob(os.path.join(test_dir, "testdata", "mat_list", "*.mat"))
    files.sort()  # Ensure consistent ordering

    # Check we have test files
    assert len(files) > 0, f"No test data files found in {test_dir}/testdata/"

    # Heights for 6 instruments
    mean_depth = 13
    m_below_surface = np.linspace(1.8, 7.2, 6)
    mabs = [mean_depth - mbs for mbs in m_below_surface]

    # Test loading the data
    adv = ADV(files, name_map, fs=32, z=mabs)

    # Basic assertions
    assert adv.fs == 32
    assert "u1" in adv.name_map
    assert "u2" in adv.name_map
    assert "u3" in adv.name_map
    assert "p" in adv.name_map

    # Check dimensions
    assert adv.n_bursts == len(files)  # Number of bursts
    assert adv.n_heights == 6  # Number of heights

    # Loading a burst
    burst = adv.load_burst(0)
    assert "u1" in burst.keys()
    assert "u2" in burst.keys()
    assert "u3" in burst.keys()
    assert "p" in burst.keys()
    assert "time" in burst.keys()
    assert burst["coords"] == "xyz"
    assert burst["u1"].shape[0] == adv.n_heights


def test_subsample():
    name_map = {"u1": "E", "u2": "N", "u3": "w", "p": "P2", "time": "dn"}
    test_dir = os.path.dirname(__file__)
    files = sorted(glob.glob(os.path.join(test_dir, "testdata", "mat_list", "*.mat")))
    mean_depth = 13
    mabs = [mean_depth - mbs for mbs in np.linspace(1.8, 7.2, 6)]

    adv = ADV(files, name_map, fs=32, z=mabs)
    adv_subsampled = adv.subsample(start_idx=0, end_idx=2)

    assert len(adv.files) == len(files)
    assert len(adv_subsampled.files) == 2
    assert eq_except(adv_subsampled, adv, "files")


def test_npy_list():
    """Test loading a list of .npy files."""
    name_map = {"u1": "E", "u2": "N", "u3": "w", "p": "P2", "time": "dn"}

    # Get test data files
    test_dir = os.path.dirname(__file__)
    files = glob.glob(os.path.join(test_dir, "testdata", "mat_list", "*.mat"))
    files.sort()  # Ensure consistent ordering

    # Check we have test files
    assert len(files) > 0, f"No test data files found in {test_dir}/testdata/"

    # Heights for 6 instruments
    mean_depth = 13
    m_below_surface = np.linspace(1.8, 7.2, 6)
    mabs = [mean_depth - mbs for mbs in m_below_surface]

    # Test loading the data
    adv = ADV(files, name_map, fs=32, z=mabs)

    # Basic assertions
    assert adv.fs == 32
    assert "u1" in adv.name_map
    assert "u2" in adv.name_map
    assert "u3" in adv.name_map
    assert "p" in adv.name_map

    # Check dimensions
    assert adv.n_bursts == len(files)  # Number of bursts
    assert adv.n_heights == 6  # Number of heights

    # Loading a burst
    burst = adv.load_burst(0)
    assert "u1" in burst.keys()
    assert "u2" in burst.keys()
    assert "u3" in burst.keys()
    assert "p" in burst.keys()
    assert "time" in burst.keys()
    assert burst["coords"] == "xyz"
    assert burst["u1"].shape[0] == adv.n_heights
###############################
# ADV.validate_inputs
###############################


def _adv_valid_kwargs(tmp_path):
    f = tmp_path / "fake.mat"
    f.write_bytes(b"")
    return {
        "files": [str(f)],
        "name_map": {"u1": "U", "u2": "V", "u3": "W"},
        "deployment_type": "fixed",
        "fs": 32.0,
        "z": [1.0],
        "z_convention": ZConvention.MAB,
    }


def test_adv_validate_inputs_happy_path(tmp_path):
    kw = _adv_valid_kwargs(tmp_path)
    assert ADV.validate_inputs(**kw) is None


def test_adv_validate_inputs_files_not_list_raises(tmp_path):
    kw = _adv_valid_kwargs(tmp_path)
    kw["files"] = (kw["files"][0],)  # tuple, not list or str
    with pytest.raises(TypeError):
        ADV.validate_inputs(**kw)


def test_adv_validate_inputs_bad_extension_raises(tmp_path):
    f = tmp_path / "fake.txt"
    f.write_bytes(b"")
    kw = _adv_valid_kwargs(tmp_path)
    kw["files"] = [str(f)]
    with pytest.raises(ValueError):
        ADV.validate_inputs(**kw)


def test_adv_validate_inputs_missing_file_raises(tmp_path):
    kw = _adv_valid_kwargs(tmp_path)
    kw["files"] = [str(tmp_path / "does_not_exist.mat")]
    with pytest.raises(FileNotFoundError):
        ADV.validate_inputs(**kw)


def test_adv_validate_inputs_name_map_not_dict_raises(tmp_path):
    kw = _adv_valid_kwargs(tmp_path)
    kw["name_map"] = [("u1", "U")]
    with pytest.raises(TypeError):
        ADV.validate_inputs(**kw)


def test_adv_validate_inputs_no_time_and_no_fs_raises(tmp_path):
    kw = _adv_valid_kwargs(tmp_path)
    kw["fs"] = None
    # name_map has no "time" key in the baseline
    with pytest.raises(ValueError):
        ADV.validate_inputs(**kw)


def test_adv_validate_inputs_z_wrong_type_raises(tmp_path):
    kw = _adv_valid_kwargs(tmp_path)
    kw["z"] = "not a number"
    with pytest.raises(TypeError):
        ADV.validate_inputs(**kw)


def test_adv_validate_inputs_z_list_non_numeric_raises(tmp_path):
    kw = _adv_valid_kwargs(tmp_path)
    kw["z"] = [1.0, "two", 3.0]
    with pytest.raises(TypeError):
        ADV.validate_inputs(**kw)


def test_adv_validate_inputs_fs_wrong_type_raises(tmp_path):
    kw = _adv_valid_kwargs(tmp_path)
    kw["fs"] = "32"
    with pytest.raises(TypeError):
        ADV.validate_inputs(**kw)


def test_adv_validate_inputs_data_keys_wrong_type_raises(tmp_path):
    kw = _adv_valid_kwargs(tmp_path)
    kw["data_keys"] = 7
    with pytest.raises(TypeError):
        ADV.validate_inputs(**kw)


def test_adv_validate_inputs_bad_deployment_type_raises(tmp_path):
    kw = _adv_valid_kwargs(tmp_path)
    kw["deployment_type"] = "cast"
    with pytest.raises(ValueError):
        ADV.validate_inputs(**kw)


def test_adv_validate_inputs_missing_required_name_map_key_raises(tmp_path):
    kw = _adv_valid_kwargs(tmp_path)
    kw["name_map"] = {"u1": "U", "u2": "V"}  # missing "u3"
    with pytest.raises(ValueError):
        ADV.validate_inputs(**kw)


def test_adv_validate_inputs_bad_source_coords_raises(tmp_path):
    kw = _adv_valid_kwargs(tmp_path)
    with pytest.raises(ValueError):
        ADV.validate_inputs(source_coords="polar", **kw)


def test_adv_validate_inputs_bad_orientation_raises(tmp_path):
    kw = _adv_valid_kwargs(tmp_path)
    with pytest.raises(ValueError):
        ADV.validate_inputs(orientation="sideways", **kw)


def test_adv_validate_inputs_bad_z_convention_raises(tmp_path):
    kw = _adv_valid_kwargs(tmp_path)
    kw["z_convention"] = ZConvention.MAS
    with pytest.raises(ValueError):
        ADV.validate_inputs(**kw)


def test_adv_validate_inputs_bad_water_depth_raises(tmp_path):
    kw = _adv_valid_kwargs(tmp_path)
    with pytest.raises(ValueError):
        ADV.validate_inputs(water_depth="deep", **kw)
