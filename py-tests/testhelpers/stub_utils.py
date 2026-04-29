import numpy as np
import types

from atmosphere.met import Met
from ocean.ctd import CTD

def make_adv(fs, n_heights=1):
    """Return a minimal namespace that satisfies ADV.covariance attribute requirements."""
    return types.SimpleNamespace(fs=fs, n_heights=n_heights)

def make_sonic(fs, n_heights=1):
    """Return a minimal namespace that satisfies ADV.covariance attribute requirements."""
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
