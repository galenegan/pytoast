import pytest
import numpy as np
from utils.despike_utils import threshold, goring_nikora, recursive_gaussian

@pytest.fixture
def noisy_data():
    rng = np.random.default_rng(42)
    out = rng.standard_normal(1000)
    og_max = np.max(out)
    og_min = np.min(out)
    bad_indices = np.random.choice(np.arange(1000), 10, replace=False)
    out[bad_indices] = 100
    return np.atleast_2d(out), og_max, og_min


def test_threshold_despike(noisy_data):
    data, og_max, og_min = noisy_data
    filtered = threshold(data, threshold_min=-20, threshold_max=20)
    assert np.all(filtered <= og_max)
    assert np.all(filtered >= og_min)

def test_goring_nikora(noisy_data):
    data, og_max, og_min = noisy_data
    filtered = goring_nikora(data, remaining_spikes=0, max_iter=10)
    assert np.all(filtered <= og_max)
    assert np.all(filtered >= og_min)

def test_recursive_gaussian(noisy_data):
    data, og_max, og_min = noisy_data
    filtered = recursive_gaussian(data, alpha=3, max_iter=10)
    assert np.all(filtered <= og_max)
    assert np.all(filtered >= og_min)