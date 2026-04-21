import pytest
import numpy.testing as npt
import scipy.io as sio
from pathlib import Path

from boundaries.bbl import styles

####################
# Reference data
####################

TEST_DATA_DIR = f"{Path(__file__).parent}/testdata"
TEST_DATA_PATH = f"{TEST_DATA_DIR}/test_styles_input.mat"
OUTPUT_DATA_PATH = f"{TEST_DATA_DIR}/test_styles_output.mat"

# Column order in BBLMPRMS (bblm02.m L207-208):
# Ro mu epsilon z1ozn z2ozn zroz1 zroz2 fofx kbs kbr znot ub ab ur
_BBLM_COLS = [
    "Ro",
    "mu",
    "epsilon",
    "z1_over_z0",
    "z2_over_z0",
    "zr_over_z1",
    "zr_over_z2",
    "fofx",
    "kbs",
    "kbr",
    "z0",
    "ub",
    "ab",
    "ur",
]

D_MEDIAN = 0.0004  # m (0.04 cm, from bblm02.m)
RTOL = 5e-3


############
# Fixtures
############


@pytest.fixture(scope="module")
def ref():
    out = sio.loadmat(OUTPUT_DATA_PATH)["BBLMPRMS"]
    out[:, 8:11] /= 100  # Convert kbs, kbr, z0 from cm to m
    return {k: out[:, i] for i, k in enumerate(_BBLM_COLS)}


@pytest.fixture(scope="module")
def results():
    inp = sio.loadmat(TEST_DATA_PATH)["DATA"].astype(float)  # (n, 6): Time Ub Ab Ur Zr Deg
    inp[:, 1:5] /= 100  # Convert Ub, Ab, Ur, Zr from cm(/s) to m(/s)
    return [
        styles(ub=inp[i, 1], ab=inp[i, 2], ur=inp[i, 3], zr=inp[i, 4], deg=inp[i, 5], d_median=D_MEDIAN)
        for i in range(len(inp))
    ]


###########
# Tests
###########


@pytest.mark.parametrize(
    "key",
    [
        "Ro",
        "mu",
        "epsilon",
        "z1_over_z0",
        "z2_over_z0",
        "zr_over_z1",
        "zr_over_z2",
        "kbs",
        "kbr",
        "z0",
    ],
)
def test_styles(key, results, ref):
    npt.assert_allclose([r[key] for r in results], ref[key], rtol=RTOL)
