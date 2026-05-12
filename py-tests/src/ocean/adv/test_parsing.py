import glob
import os
import numpy as np

from ocean.adv import ADV
from testhelpers.stub_utils import eq_except


def test_mat_list():
    """Test loading a list of .mat files."""
    name_map = {"u1": "E", "u2": "N", "u3": "w", "p": "P2", "time": "dn"}

    # Get test data files
    test_dir = os.path.dirname(__file__)
    files = glob.glob(os.path.join(test_dir, "testdata", "mat_list", "*.mat"))
    files.sort()  # Ensure consistent ordering

    # Check we have test files
    assert len(files) > 0, f"No test data files found in {test_dir}/testdata/"

    # Heights for 6 instruments
    mean_depth = 13
    m_below_surface = np.linspace(1.8, 7.2, 6)
    mabs = [mean_depth - mbs for mbs in m_below_surface]

    # Test loading the data
    adv = ADV(files, name_map, fs=32, z=mabs)

    # Basic assertions
    assert adv.fs == 32
    assert "u1" in adv.name_map
    assert "u2" in adv.name_map
    assert "u3" in adv.name_map
    assert "p" in adv.name_map

    # Check dimensions
    assert adv.n_bursts == len(files)  # Number of bursts
    assert adv.n_heights == 6  # Number of heights

    # Loading a burst
    burst = adv.load_burst(0)
    assert "u1" in burst.keys()
    assert "u2" in burst.keys()
    assert "u3" in burst.keys()
    assert "p" in burst.keys()
    assert "time" in burst.keys()
    assert burst["coords"] == "xyz"
    assert burst["u1"].shape[0] == adv.n_heights


def test_subsample():
    name_map = {"u1": "E", "u2": "N", "u3": "w", "p": "P2", "time": "dn"}
    test_dir = os.path.dirname(__file__)
    files = sorted(glob.glob(os.path.join(test_dir, "testdata", "mat_list", "*.mat")))
    mean_depth = 13
    mabs = [mean_depth - mbs for mbs in np.linspace(1.8, 7.2, 6)]

    adv = ADV(files, name_map, fs=32, z=mabs)
    adv_subsampled = adv.subsample(start_idx=0, end_idx=2)

    assert len(adv.files) == len(files)
    assert len(adv_subsampled.files) == 2
    assert eq_except(adv_subsampled, adv, "files")


def test_npy_list():
    """Test loading a list of .npy files."""
    name_map = {"u1": "E", "u2": "N", "u3": "w", "p": "P2", "time": "dn"}

    # Get test data files
    test_dir = os.path.dirname(__file__)
    files = glob.glob(os.path.join(test_dir, "testdata", "mat_list", "*.mat"))
    files.sort()  # Ensure consistent ordering

    # Check we have test files
    assert len(files) > 0, f"No test data files found in {test_dir}/testdata/"

    # Heights for 6 instruments
    mean_depth = 13
    m_below_surface = np.linspace(1.8, 7.2, 6)
    mabs = [mean_depth - mbs for mbs in m_below_surface]

    # Test loading the data
    adv = ADV(files, name_map, fs=32, z=mabs)

    # Basic assertions
    assert adv.fs == 32
    assert "u1" in adv.name_map
    assert "u2" in adv.name_map
    assert "u3" in adv.name_map
    assert "p" in adv.name_map

    # Check dimensions
    assert adv.n_bursts == len(files)  # Number of bursts
    assert adv.n_heights == 6  # Number of heights

    # Loading a burst
    burst = adv.load_burst(0)
    assert "u1" in burst.keys()
    assert "u2" in burst.keys()
    assert "u3" in burst.keys()
    assert "p" in burst.keys()
    assert "time" in burst.keys()
    assert burst["coords"] == "xyz"
    assert burst["u1"].shape[0] == adv.n_heights
