import glob
import os

import numpy as np
import pytest

from ocean.adcp import ADCP
from utils.base_instrument import ZConvention
from testhelpers.rotate_utils import nortek_4beam_T
from testhelpers.stub_utils import eq_except


NAME_MAP = {
    "u1": "Burst_VelBeam1",
    "u2": "Burst_VelBeam2",
    "u3": "Burst_VelBeam3",
    "u4": "Burst_VelBeam4",
    "u5": "IBurst_VelBeam5",
    "p": "Burst_Pressure",
    "time": "Burst_TimeStamp",
    "z": "Burst_Range",
    "heading": "Burst_Heading",
    "pitch": "Burst_Pitch",
    "roll": "Burst_Roll",
}

T_NORTEK = nortek_4beam_T()

N_SAMPLES = 4096
N_HEIGHTS = 32


def _testdata_glob():
    test_dir = os.path.dirname(__file__)
    files = sorted(glob.glob(os.path.join(test_dir, "testdata", "BBASIT_0078_burst*.mat")))
    assert files
    return files


def _synth_oneburst_path():
    return os.path.join(os.path.dirname(__file__), "testdata", "synth_oneburst.mat")


def _make_adcp(files=None, **overrides):
    if files is None:
        files = _testdata_glob()
    kwargs = dict(
        files=files,
        name_map=NAME_MAP,
        source_coords="beam",
        orientation="up",
        manufacturer="nortek",
        data_keys="Data",
    )
    kwargs.update(overrides)
    return ADCP(**kwargs)


def test_mat_list_loads():
    files = _testdata_glob()
    adcp = _make_adcp(files=files)

    assert adcp.fs == 4.0
    assert adcp.n_bursts == len(files)
    assert adcp.n_heights == N_HEIGHTS
    assert adcp.beam_keys == ["u1", "u2", "u3", "u4", "u5"]
    assert adcp.num_beams == 5
    assert adcp.manufacturer == "nortek"
    assert adcp.source_coords == "beam"


def test_load_burst_no_preprocessing():
    adcp = _make_adcp()
    burst = adcp.load_burst(0)

    for key in ["u1", "u2", "u3", "u4", "u5"]:
        assert burst[key].shape == (N_HEIGHTS, N_SAMPLES), key
    for key in ["p", "time", "heading", "pitch", "roll"]:
        assert burst[key].shape[-1] == N_SAMPLES, key
    assert burst["coords"] == "beam"


def test_load_burst_with_despike():
    """goring_nikora should remove the injected spikes from the synthetic
    fixture."""
    adcp = _make_adcp(files=[_synth_oneburst_path()])
    adcp.set_preprocess_opts({"despike": {"method": "goring_nikora"}})
    burst = adcp.load_burst(0)

    assert burst["u1"].shape == (1, 1024)
    assert np.all(np.isfinite(burst["u1"]))
    # Pre-despike the synthetic data has spikes of magnitude 5.0 at known indices;
    # after despiking the max magnitude should be well below the spike level.
    assert np.max(np.abs(burst["u1"])) < 1.0


def test_beam_to_xyz_transform():
    adcp = _make_adcp()
    adcp.set_preprocess_opts({"rotate": {"coords_out": "xyz", "transformation_matrix": T_NORTEK}})
    burst = adcp.load_burst(0)

    assert burst["coords"] == "xyz"
    for key in ["u1", "u2", "u3", "u4"]:
        assert burst[key].shape == (N_HEIGHTS, N_SAMPLES)
    # The xyz vertical channel (u3) carries less variance than the streamwise
    # horizontal (u1) for this deployment.
    u3_var = np.var(burst["u3"], axis=1).mean()
    u1_var = np.var(burst["u1"], axis=1).mean()
    assert u3_var < u1_var


def test_flow_rotation_align_streamwise():
    adcp = _make_adcp()
    adcp.set_preprocess_opts(
        {
            "rotate": {
                "coords_out": "xyz",
                "transformation_matrix": T_NORTEK,
                "flow_rotation": "align_streamwise",
            }
        }
    )
    burst = adcp.load_burst(0)
    # Streamwise alignment puts the mean current entirely on u1; mean(u2) and mean(u3)
    # should be ~0 to single-precision floating-point tolerance.
    u1_speed = np.max(np.abs(np.mean(burst["u1"], axis=1)))
    assert np.max(np.abs(np.mean(burst["u2"], axis=1))) < 1e-6 * max(u1_speed, 1.0)
    assert np.max(np.abs(np.mean(burst["u3"], axis=1))) < 1e-6 * max(u1_speed, 1.0)


def test_flow_rotation_in_beam_raises():
    adcp = _make_adcp()
    adcp.set_preprocess_opts({"rotate": {"flow_rotation": "align_streamwise"}})
    with pytest.raises(ValueError, match="beam coordinates"):
        adcp.load_burst(0)


def test_invalid_inputs_raise():
    files = _testdata_glob()

    bad_name_map = {k: v for k, v in NAME_MAP.items() if k != "u1"}
    with pytest.raises(ValueError, match="u1"):
        ADCP(files=files, name_map=bad_name_map, data_keys="Data")

    with pytest.raises(ValueError, match="source_coords"):
        ADCP(files=files, name_map=NAME_MAP, source_coords="bogus", data_keys="Data")

    with pytest.raises(ValueError, match="orientation"):
        ADCP(files=files, name_map=NAME_MAP, orientation="sideways", data_keys="Data")

    with pytest.raises(ValueError, match="manufacturer"):
        ADCP(files=files, name_map=NAME_MAP, manufacturer="acme", data_keys="Data")


def test_subsample():
    adcp = _make_adcp()
    adcp_subsampled = adcp.subsample(start_idx=0, end_idx=2)
    original_files = getattr(adcp, "files")
    subsampled_files = getattr(adcp_subsampled, "files")
    assert len(original_files) == 3
    assert len(subsampled_files) == 2
    assert eq_except(adcp_subsampled, adcp, "files")


###############################
# ADCP.validate_inputs
###############################


def _adcp_valid_kwargs(tmp_path):
    f = tmp_path / "fake.mat"
    f.write_bytes(b"")
    return {
        "files": [str(f)],
        "name_map": {"u1": "U", "u2": "V", "u3": "W"},
        "deployment_type": "fixed",
        "fs": 16.0,
        "z": [1.0, 2.0, 3.0],
        "z_convention": ZConvention.MAB,
    }


def test_adcp_validate_inputs_happy_path(tmp_path):
    kw = _adcp_valid_kwargs(tmp_path)
    assert ADCP.validate_inputs(**kw) is None


def test_adcp_validate_inputs_files_not_list_raises(tmp_path):
    kw = _adcp_valid_kwargs(tmp_path)
    kw["files"] = (kw["files"][0],)
    with pytest.raises(TypeError):
        ADCP.validate_inputs(**kw)


def test_adcp_validate_inputs_bad_extension_raises(tmp_path):
    f = tmp_path / "fake.xyz"
    f.write_bytes(b"")
    kw = _adcp_valid_kwargs(tmp_path)
    kw["files"] = [str(f)]
    with pytest.raises(ValueError):
        ADCP.validate_inputs(**kw)


def test_adcp_validate_inputs_missing_file_raises(tmp_path):
    kw = _adcp_valid_kwargs(tmp_path)
    kw["files"] = [str(tmp_path / "missing.mat")]
    with pytest.raises(FileNotFoundError):
        ADCP.validate_inputs(**kw)


def test_adcp_validate_inputs_name_map_not_dict_raises(tmp_path):
    kw = _adcp_valid_kwargs(tmp_path)
    kw["name_map"] = ("u1", "U")
    with pytest.raises(TypeError):
        ADCP.validate_inputs(**kw)


def test_adcp_validate_inputs_no_time_and_no_fs_raises(tmp_path):
    kw = _adcp_valid_kwargs(tmp_path)
    kw["fs"] = None
    with pytest.raises(ValueError):
        ADCP.validate_inputs(**kw)


def test_adcp_validate_inputs_z_wrong_type_raises(tmp_path):
    kw = _adcp_valid_kwargs(tmp_path)
    kw["z"] = "bad"
    with pytest.raises(TypeError):
        ADCP.validate_inputs(**kw)


def test_adcp_validate_inputs_z_list_non_numeric_raises(tmp_path):
    kw = _adcp_valid_kwargs(tmp_path)
    kw["z"] = [1.0, "two"]
    with pytest.raises(TypeError):
        ADCP.validate_inputs(**kw)


def test_adcp_validate_inputs_fs_wrong_type_raises(tmp_path):
    kw = _adcp_valid_kwargs(tmp_path)
    kw["fs"] = "16"
    with pytest.raises(TypeError):
        ADCP.validate_inputs(**kw)


def test_adcp_validate_inputs_data_keys_wrong_type_raises(tmp_path):
    kw = _adcp_valid_kwargs(tmp_path)
    kw["data_keys"] = 5
    with pytest.raises(TypeError):
        ADCP.validate_inputs(**kw)


def test_adcp_validate_inputs_bad_deployment_type_raises(tmp_path):
    kw = _adcp_valid_kwargs(tmp_path)
    kw["deployment_type"] = "cast"
    with pytest.raises(ValueError):
        ADCP.validate_inputs(**kw)


def test_adcp_validate_inputs_missing_required_name_map_key_raises(tmp_path):
    kw = _adcp_valid_kwargs(tmp_path)
    kw["name_map"] = {"u1": "U", "u3": "W"}  # missing u2
    with pytest.raises(ValueError):
        ADCP.validate_inputs(**kw)


def test_adcp_validate_inputs_bad_source_coords_raises(tmp_path):
    kw = _adcp_valid_kwargs(tmp_path)
    with pytest.raises(ValueError):
        ADCP.validate_inputs(source_coords="polar", **kw)


def test_adcp_validate_inputs_bad_orientation_raises(tmp_path):
    kw = _adcp_valid_kwargs(tmp_path)
    with pytest.raises(ValueError):
        ADCP.validate_inputs(orientation="sideways", **kw)


def test_adcp_validate_inputs_bad_beam_angle_raises(tmp_path):
    kw = _adcp_valid_kwargs(tmp_path)
    with pytest.raises(ValueError):
        ADCP.validate_inputs(beam_angle="twenty", **kw)


def test_adcp_validate_inputs_bad_z_convention_raises(tmp_path):
    kw = _adcp_valid_kwargs(tmp_path)
    kw["z_convention"] = ZConvention.MAS
    with pytest.raises(ValueError):
        ADCP.validate_inputs(**kw)


def test_adcp_validate_inputs_bad_manufacturer_raises(tmp_path):
    kw = _adcp_valid_kwargs(tmp_path)
    with pytest.raises(ValueError):
        ADCP.validate_inputs(manufacturer="acme", **kw)
