import numpy as np
import numpy.testing as npt
import scipy.io as sio

from boundaries.bbl.styles_2017 import styles

BBLM_DIR = "/Users/ea-gegan/Documents/gitrepos/pytoast/src/boundaries/bbl/bblm02"
TEST_DATA_PATH = f"{BBLM_DIR}/test2.mat"
OUTPUT_DATA_PATH = f"{BBLM_DIR}/model_output_file2017.mat"

# Column indices in BBLMPRMS (from bblm02.m L207-208):
# Ro mu epsilon z1ozn z2ozn zroz1 zroz2 fofx kbs kbr znot ub ab ur
_COL = {
    k: i
    for i, k in enumerate(
        ["Ro", "mu", "epsilon", "z1_over_z0", "z2_over_z0", "zr_over_z1", "zr_over_z2", "fofx", "kbs", "kbr", "z0", "ub", "ab", "ur"]
    )
}
D_MEDIAN = 0.0004  # m (from bblm02.m)


def _load():
    inp = sio.loadmat(TEST_DATA_PATH)["DATA"].astype(float)  # (m, 6): Time Ub Ab Ur Zr Deg
    inp[:, 1:5] /= 100  # Convert Ub, Ab, Ur, Zr from cm(/s) to m(/s); Time and Deg unchanged
    out = sio.loadmat(OUTPUT_DATA_PATH)
    out["BBLMPRMS"][:, 8:11] /= 100  # Convert kbs, kbr, z0 from cm to m; cols 11-13 (ub, ab, ur) unused in tests
    return inp, out["BBLMPRMS"]


class TestStylesVsMatlab:
    """Verify Python port against MATLAB bblm02 reference output (test2.mat)."""

    def setup_method(self):
        inp, self.ref = _load()
        self.results = [
            styles(
                ub=inp[i, 1],
                ab=inp[i, 2],
                ur=inp[i, 3],
                zr=inp[i, 4],
                deg=inp[i, 5],
                d_median=D_MEDIAN,
            )
            for i in range(len(inp))
        ]

    def _col(self, key):
        return self.ref[:, _COL[key]]

    def _check(self, key, rtol=5e-3):
        computed = np.array([r[key] for r in self.results])
        npt.assert_allclose(computed, self._col(key), rtol=rtol, err_msg=f"Mismatch on '{key}'")

    def test_Ro(self):
        self._check("Ro")

    def test_mu(self):
        self._check("mu")

    def test_epsilon(self):
        self._check("epsilon")

    def test_z1_over_z0(self):
        self._check("z1_over_z0")

    def test_z2_over_z0(self):
        self._check("z2_over_z0")

    def test_zr_over_z1(self):
        self._check("zr_over_z1")

    def test_zr_over_z2(self):
        self._check("zr_over_z2")

    def test_kbs(self):
        self._check("kbs")

    def test_kbr(self):
        self._check("kbr")

    def test_z0(self):
        self._check("z0")
