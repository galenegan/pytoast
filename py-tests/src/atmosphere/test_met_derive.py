import numpy as np
import numpy.testing as npt
import pytest
import pytoast.utils.air_thermo as air_thermo
from pytoast.atmosphere.met import Met
from pytoast.utils.base_instrument import ZConvention
from testhelpers.stub_utils import eq_except, make_met


NAME_MAP = {"t": "TEMP", "p": "PRES", "rh": "RH", "time": "TIME"}
T_ONLY = {"cp", "L_v", "nu", "theta"}
T_P = T_ONLY | {"e_s", "rho_air_dry"}
T_P_RH = T_P | {"e", "rho_v", "w", "q", "t_v", "rho_air"}

met = make_met(NAME_MAP)


def _burst(n_heights=1, n_samples=4, keys=("t", "p", "rh"), sp=None):
    shape = (n_heights, n_samples)
    values = {
        "t": np.full(shape, 20.0),
        "p": np.full(shape, 1013.0),
        "rh": np.full(shape, 70.0),
    }
    out = {k: values[k] for k in keys}
    if sp is not None:
        out["sp"] = np.full(shape, sp)
    return out


def test_t_only():
    out = met.derive(_burst(keys=("t",)))
    assert T_ONLY <= out.keys()
    assert not (T_P_RH - T_ONLY) & out.keys()


def test_t_and_p():
    out = met.derive(_burst(keys=("t", "p")))
    assert T_P <= out.keys()
    assert not (T_P_RH - T_P) & out.keys()


def test_t_p_rh_full_set():
    out = met.derive(_burst(keys=("t", "p", "rh")))
    assert T_P_RH <= out.keys()


def test_sp_forwarded_to_saturation_vapor_pressure():
    burst_no_sp = _burst()
    burst_sp = _burst(sp=35.0)
    out_no_sp = met.derive(burst_no_sp)
    out_sp = met.derive(burst_sp)
    expected = air_thermo.saturation_vapor_pressure(burst_sp["t"], burst_sp["p"], burst_sp["sp"])
    npt.assert_allclose(out_sp["e_s"], expected)
    assert not np.allclose(out_sp["e_s"], out_no_sp["e_s"])


def test_no_t_returns_unchanged():
    burst = _burst(keys=("p", "rh"))
    original_keys = set(burst.keys())
    out = met.derive(burst)
    assert set(out.keys()) == original_keys


def test_theta_broadcasts_over_heights():
    n_heights, n_samples = 3, 5
    out = make_met(NAME_MAP, n_heights=n_heights).derive(_burst(n_heights=n_heights, n_samples=n_samples, keys=("t",)))
    assert out["theta"].shape == (n_heights, n_samples)


def test_mutates_in_place():
    burst = _burst()
    out = met.derive(burst)
    assert out is burst


def _write_met_npy(path, n_samples=8):
    data = {
        "TEMP": np.full(n_samples, 20.0),
        "PRES": np.full(n_samples, 1013.0),
        "RH": np.full(n_samples, 70.0),
        "TIME": np.arange(n_samples, dtype=float),
    }
    np.save(path, data, allow_pickle=True)


def test_subsample(tmp_path):
    files = []
    for i in range(3):
        p = str(tmp_path / f"burst_{i}.npy")
        _write_met_npy(p)
        files.append(p)

    met_full = Met(files=files, name_map=NAME_MAP, fs=1.0, z=2.0)
    met_subsampled = met_full.subsample(start_idx=0, end_idx=2)

    assert len(met_full.files) == 3
    assert len(met_subsampled.files) == 2
    assert eq_except(met_subsampled, met_full, "files")


###############################
# Met.validate_inputs
###############################


def _met_valid_kwargs(tmp_path):
    f = tmp_path / "fake.mat"
    f.write_bytes(b"")
    return {
        "files": [str(f)],
        "name_map": {"t": "TEMP", "p": "PRES", "rh": "RH"},
        "deployment_type": "fixed",
        "fs": 1.0,
        "z": [10.0],
        "z_convention": ZConvention.MAS,
    }


def test_met_validate_inputs_happy_path(tmp_path):
    kw = _met_valid_kwargs(tmp_path)
    assert Met.validate_inputs(**kw) is None


def test_met_validate_inputs_files_not_list_raises(tmp_path):
    kw = _met_valid_kwargs(tmp_path)
    kw["files"] = (kw["files"][0],)
    with pytest.raises(TypeError):
        Met.validate_inputs(**kw)


def test_met_validate_inputs_bad_extension_raises(tmp_path):
    f = tmp_path / "fake.txt"
    f.write_bytes(b"")
    kw = _met_valid_kwargs(tmp_path)
    kw["files"] = [str(f)]
    with pytest.raises(ValueError):
        Met.validate_inputs(**kw)


def test_met_validate_inputs_missing_file_raises(tmp_path):
    kw = _met_valid_kwargs(tmp_path)
    kw["files"] = [str(tmp_path / "missing.mat")]
    with pytest.raises(FileNotFoundError):
        Met.validate_inputs(**kw)


def test_met_validate_inputs_name_map_not_dict_raises(tmp_path):
    kw = _met_valid_kwargs(tmp_path)
    kw["name_map"] = "TEMP"
    with pytest.raises(TypeError):
        Met.validate_inputs(**kw)


def test_met_validate_inputs_no_time_and_no_fs_raises(tmp_path):
    kw = _met_valid_kwargs(tmp_path)
    kw["fs"] = None
    with pytest.raises(ValueError):
        Met.validate_inputs(**kw)


def test_met_validate_inputs_z_wrong_type_raises(tmp_path):
    kw = _met_valid_kwargs(tmp_path)
    kw["z"] = "ten"
    with pytest.raises(TypeError):
        Met.validate_inputs(**kw)


def test_met_validate_inputs_z_list_non_numeric_raises(tmp_path):
    kw = _met_valid_kwargs(tmp_path)
    kw["z"] = [10.0, "twenty"]
    with pytest.raises(TypeError):
        Met.validate_inputs(**kw)


def test_met_validate_inputs_fs_wrong_type_raises(tmp_path):
    kw = _met_valid_kwargs(tmp_path)
    kw["fs"] = "1"
    with pytest.raises(TypeError):
        Met.validate_inputs(**kw)


def test_met_validate_inputs_data_keys_wrong_type_raises(tmp_path):
    kw = _met_valid_kwargs(tmp_path)
    kw["data_keys"] = 11
    with pytest.raises(TypeError):
        Met.validate_inputs(**kw)


def test_met_validate_inputs_bad_deployment_type_raises(tmp_path):
    kw = _met_valid_kwargs(tmp_path)
    kw["deployment_type"] = "cast"
    with pytest.raises(ValueError):
        Met.validate_inputs(**kw)


def test_met_validate_inputs_bad_z_convention_raises(tmp_path):
    kw = _met_valid_kwargs(tmp_path)
    kw["z_convention"] = ZConvention.MAB
    with pytest.raises(ValueError):
        Met.validate_inputs(**kw)
