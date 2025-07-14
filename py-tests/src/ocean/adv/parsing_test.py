import glob
import pytest
import os
import numpy as np

from ocean.adv import ADV


def test_mat_list():
    """Test loading a list of .mat files"""
    name_map = {"u": "E", "v": "N", "w": "w", "p": "P2", "time": "dn"}

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
    adv = ADV.from_raw(files, name_map, fs=32, z=mabs)

    # Basic assertions
    assert adv.ds.attrs["fs"] == 32
    assert "u" in adv.ds.data_vars
    assert "v" in adv.ds.data_vars
    assert "w" in adv.ds.data_vars
    assert "p" in adv.ds.data_vars

    # Check dimensions
    assert adv.u.shape[0] == len(files)  # Number of bursts
    assert adv.u.shape[1] == 6  # Number of heights
    assert adv.u.shape[2] > 0  # Time dimension
