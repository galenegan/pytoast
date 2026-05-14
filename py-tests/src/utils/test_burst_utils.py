import numpy as np
import pytest

from utils.burst_utils import get_beams, get_uvw


def _burst(coords, n=3):
    d = {f"u{i}": np.arange(10) + i for i in range(1, n + 1)}
    d["coords"] = coords
    return d


def test_get_uvw_enu_returns_arrays():
    burst = _burst("enu")
    u, v, w = get_uvw(burst)
    assert np.array_equal(u, burst["u1"])
    assert np.array_equal(v, burst["u2"])
    assert np.array_equal(w, burst["u3"])


def test_get_uvw_xyz_default_allows():
    burst = _burst("xyz")
    u, v, w = get_uvw(burst)
    assert np.array_equal(u, burst["u1"])


def test_get_uvw_beam_raises():
    burst = _burst("beam")
    with pytest.raises(ValueError, match="coords"):
        get_uvw(burst)


def test_get_uvw_missing_coords_raises():
    burst = {f"u{i}": np.zeros(5) for i in range(1, 4)}
    with pytest.raises(ValueError, match="coords"):
        get_uvw(burst)


def test_get_uvw_strict_enu_rejects_xyz():
    burst = _burst("xyz")
    with pytest.raises(ValueError):
        get_uvw(burst, allowed_coords=("enu",))


def test_get_beams_5beam():
    burst = _burst("beam", n=5)
    beams = get_beams(burst, 5)
    assert len(beams) == 5
    for i, b in enumerate(beams, start=1):
        assert np.array_equal(b, burst[f"u{i}"])


def test_get_beams_wrong_coords_raises():
    burst = _burst("xyz", n=4)
    with pytest.raises(ValueError, match="beam"):
        get_beams(burst, 4)
