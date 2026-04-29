import numpy as np
import types

from atmosphere.met import Met
from ocean.adcp import ADCP
from ocean.ctd import CTD
from testhelpers.rotate_utils import nortek_4beam_T

def make_adv(fs, n_heights=1):
    """Return a minimal namespace that satisfies ADV.covariance attribute
    requirements."""
    return types.SimpleNamespace(fs=fs, n_heights=n_heights)

def make_sonic(fs, n_heights=1):
    """Return a minimal namespace that satisfies ADV.covariance attribute
    requirements."""
    return types.SimpleNamespace(fs=fs, n_heights=n_heights)

def make_met(name_map, n_heights=1):
    inst = Met.__new__(Met)
    inst.fs = 1.0
    inst.z = np.linspace(2.0, 2.0 + float(n_heights - 1), n_heights)
    inst.name_map = name_map
    return inst

def make_ctd(name_map, n_heights=1):
    inst = CTD.__new__(CTD)
    inst.fs = 1.0
    inst.z = np.linspace(-1.0, -float(n_heights), n_heights)
    inst.name_map = name_map
    return inst

def make_adcp(
    fs,
    z,
    num_beams=5,
    manufacturer="nortek",
    beam_angle=25.0,
    transformation_matrix=None,
    source_coords="beam",
    orientation="up",
):
    """Return a real ADCP instance (created via __new__, no file I/O) populated
    with the attributes the covariance/dissipation/shear methods touch.

    Bypassing __init__ lets tests drive ADCP.method(adcp, burst, ...)
    directly with synthetic burst data, while still resolving
    self._apply_coord_transform via the real method on the class.
    """
    inst = ADCP.__new__(ADCP)
    inst.fs = float(fs)
    inst.z = np.asarray(z, dtype=float)
    inst.manufacturer = manufacturer
    inst.beam_angle = float(beam_angle)
    inst.source_coords = source_coords
    inst.orientation = orientation
    inst.name_map = {f"u{i}": f"beam{i}" for i in range(1, num_beams + 1)}
    T = transformation_matrix if transformation_matrix is not None else nortek_4beam_T(beam_angle)
    inst._rotate = {"transformation_matrix": T}
    inst._preprocess_enabled = False
    return inst
