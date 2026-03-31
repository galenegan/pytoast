import pytest
import numpy as np
import numpy.testing as npt
import scipy.io as sio
from pathlib import Path

from boundaries.bbl import styles
# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

BBLM_DIR = f"{Path(__file__).parent}/testdata"
TEST_DATA_PATH = f"{BBLM_DIR}/test2.mat"
OUTPUT_DATA_PATH = f"{BBLM_DIR}/model_output_file2017.mat"

# Column indices in BBLMPRMS (from bblm02.m L207-208):
# Ro mu epsilon z1ozn z2ozn zroz1 zroz2 fofx kbs kbr znot ub ab ur
_COL = {
    k: i
    for i, k in enumerate(
        [
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
    )
}

D_MEDIAN = 0.0004  # m (0.04 cm, from bblm02.m)
RTOL = 5e-3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ref():
    out = sio.loadmat(OUTPUT_DATA_PATH)["BBLMPRMS"]
    out[:, 8:11] /= 100  # Convert kbs, kbr, z0 from cm to m; cols 11-13 (ub, ab, ur) unused
    return out


@pytest.fixture(scope="module")
def results():
    inp = sio.loadmat(TEST_DATA_PATH)["DATA"].astype(float)  # (n, 6): Time Ub Ab Ur Zr Deg
    inp[:, 1:5] /= 100  # Convert Ub, Ab, Ur, Zr from cm(/s) to m(/s); Time and Deg unchanged
    return [
        styles(ub=inp[i, 1], ab=inp[i, 2], ur=inp[i, 3], zr=inp[i, 4], deg=inp[i, 5], d_median=D_MEDIAN)
        for i in range(len(inp))
    ]


# ---------------------------------------------------------------------------
# Nondimensional outputs
# ---------------------------------------------------------------------------


def test_Ro(results, ref):
    npt.assert_allclose([r["Ro"] for r in results], ref[:, _COL["Ro"]], rtol=RTOL)


def test_mu(results, ref):
    npt.assert_allclose([r["mu"] for r in results], ref[:, _COL["mu"]], rtol=RTOL)


def test_epsilon(results, ref):
    npt.assert_allclose([r["epsilon"] for r in results], ref[:, _COL["epsilon"]], rtol=RTOL)


def test_z1_over_z0(results, ref):
    npt.assert_allclose([r["z1_over_z0"] for r in results], ref[:, _COL["z1_over_z0"]], rtol=RTOL)


def test_z2_over_z0(results, ref):
    npt.assert_allclose([r["z2_over_z0"] for r in results], ref[:, _COL["z2_over_z0"]], rtol=RTOL)


def test_zr_over_z1(results, ref):
    npt.assert_allclose([r["zr_over_z1"] for r in results], ref[:, _COL["zr_over_z1"]], rtol=RTOL)


def test_zr_over_z2(results, ref):
    npt.assert_allclose([r["zr_over_z2"] for r in results], ref[:, _COL["zr_over_z2"]], rtol=RTOL)


# ---------------------------------------------------------------------------
# Roughness lengths
# ---------------------------------------------------------------------------


def test_kbs(results, ref):
    npt.assert_allclose([r["kbs"] for r in results], ref[:, _COL["kbs"]], rtol=RTOL)


def test_kbr(results, ref):
    npt.assert_allclose([r["kbr"] for r in results], ref[:, _COL["kbr"]], rtol=RTOL)


def test_z0(results, ref):
    npt.assert_allclose([r["z0"] for r in results], ref[:, _COL["z0"]], rtol=RTOL)
