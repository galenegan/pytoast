import numpy as np
import types

from atmosphere.met import Met
from ocean.adcp import ADCP
from ocean.ctd import CTD

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


def _standard_nortek_T(beam_angle_deg=25.0):
    """
    Beam->xyz transformation matrix matching the standard ADCP beam-projection
    convention assumed by the covariance and dissipation derivations in adcp.py.

    Beams 1, 3 lie in the x-z plane: b1 = sin(theta) u + cos(theta) w, b3 = -sin(theta) u + cos(theta) w.
    Beams 2, 4 lie in the y-z plane: b2 = sin(theta) v + cos(theta) w, b4 = -sin(theta) v + cos(theta) w.
    Inverting gives x = (b1 - b3) / (2 sin theta), z1 = (b1 + b3) / (2 cos theta), and the y / z2
    analogs. This T (and the derived xyz) is consistent with the radial beam projection that
    underlies the Stacey, Guerra-Thomson, and McMillan formulas used in src/ocean/adcp.py.
    """
    theta = np.deg2rad(beam_angle_deg)
    a = 1.0 / (2.0 * np.sin(theta))
    b = 1.0 / (2.0 * np.cos(theta))
    return np.array(
        [
            [a, 0.0, -a, 0.0],
            [0.0, a, 0.0, -a],
            [b, 0.0, b, 0.0],
            [0.0, b, 0.0, b],
        ]
    )


_NORTEK_T_DEFAULT = _standard_nortek_T(25.0)


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
    T = transformation_matrix if transformation_matrix is not None else _NORTEK_T_DEFAULT
    inst._rotate = {"transformation_matrix": T}
    inst._preprocess_enabled = False
    return inst
