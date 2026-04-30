import numpy.testing as npt
import numpy as np
from utils.wave_utils import get_wavenumber


def test_dispersion_relation_deep_limit():
    omega = 2 * np.pi / 10
    h = 1000
    k = get_wavenumber(omega, h)
    k_deep_water = omega**2 / 9.81
    npt.assert_equal(k, k_deep_water)


def test_dispersion_relation_shallow_limit():
    omega = 2 * np.pi / 1000
    h = 10
    k = get_wavenumber(omega, h)
    k_shallow_water = omega / np.sqrt(9.81 * h)
    npt.assert_almost_equal(k, k_shallow_water, decimal=4)
