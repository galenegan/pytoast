import numpy as np
import numpy.testing as npt

from ocean.ctd import CTD
import utils.sea_thermo as sea_thermo


NAME_MAP = {"sp": "PSAL", "t": "TEMP", "p": "PRES", "time": "TIME"}


def _stub(n_heights=1):
    inst = CTD.__new__(CTD)
    inst.fs = 1.0
    inst.z = np.linspace(-1.0, -float(n_heights), n_heights)
    inst.name_map = NAME_MAP
    return inst


def _burst(n_heights=1, n_samples=4, keys=("sp", "t", "p")):
    shape = (n_heights, n_samples)
    values = {
        "sp": np.full(shape, 35.0),
        "t": np.full(shape, 15.0),
        "p": np.full(shape, 100.0),
    }
    return {k: values[k] for k in keys}


ALL_DERIVED = {"sa", "ct", "rho", "sigma0", "alpha", "beta", "sound_speed", "t_freezing", "cp", "nu", "z"}


def test_sp_only_adds_only_sa():
    out = _stub().derive(_burst(keys=("sp",)))
    assert "sa" in out
    assert ALL_DERIVED - {"sa"} & out.keys() == set()


def test_full_inputs_single_height_no_N2():
    out = _stub(n_heights=1).derive(_burst(n_heights=1))
    assert ALL_DERIVED <= out.keys()
    assert "N2" not in out


def test_full_inputs_multi_height_adds_N2():
    n_heights, n_samples = 3, 4
    out = _stub(n_heights=n_heights).derive(_burst(n_heights=n_heights, n_samples=n_samples))
    assert "N2" in out
    assert out["N2"].shape == (n_heights - 1, n_samples)


def test_p_only_adds_z_but_no_thermo():
    out = _stub().derive(_burst(keys=("t", "p")))
    assert "z" in out
    assert not (ALL_DERIVED - {"z"}) & out.keys()


def test_lat_forwarded_to_depth_from_pressure():
    burst = _burst(keys=("p",))
    burst["lat"] = 45.0
    out = _stub().derive(burst)
    expected = sea_thermo.depth_from_pressure(burst["p"], 45.0)
    npt.assert_allclose(out["z"], expected)
    assert not np.allclose(out["z"], sea_thermo.depth_from_pressure(burst["p"]))


def test_mutates_in_place():
    burst = _burst()
    out = _stub().derive(burst)
    assert out is burst


def test_empty_burst_returns_empty():
    out = _stub().derive({})
    assert out == {}
