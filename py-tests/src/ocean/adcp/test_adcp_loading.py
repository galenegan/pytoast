import glob
import os

import numpy as np
import pytest

from ocean.adcp import ADCP


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

T_NORTEK = np.array(
    [
        [1.1831, 0.0, 0.5518, 0.0],
        [0.0, -1.1831, 0.0, 0.5518],
        [-1.1831, 0.0, 0.5518, 0.0],
        [0.0, 1.1831, 0.0, 0.5518],
    ]
)

N_SAMPLES = 4096
N_HEIGHTS = 32


def _testdata_glob():
    test_dir = os.path.dirname(__file__)
    files = sorted(glob.glob(os.path.join(test_dir, "testdata", "BBASIT_0078_burst*.mat")))
    assert files, f"No truncated BBASIT_0078_burst*.mat fixtures in {test_dir}/testdata/ — run _truncate_source.py"
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
    # Pre-despike the synthetic fixture has spikes of magnitude 5.0 at known indices;
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
    # horizontal (u1) for this upward-looking deployment.
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
