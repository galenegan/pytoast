import numpy as np
import numpy.testing as npt
from pathlib import Path
from boundaries.bbl import madsen

# From Chris Sherwood/Dan Nowacki implementation
test_path = f"{Path(__file__).parent}/testdata"
ref = np.load(f"{test_path}/m94_output.npy", allow_pickle=True).item()



def test_madsen():
    output = madsen(
        ub_r=0.1, omega_r=2 * np.pi / 5, uc_r=0.3, phi_c=0, phi_wr=0, z_r=1, kN=0.05
    )
    npt.assert_almost_equal(output["ustar_c"], ref["ustrc"], decimal=3)
    npt.assert_almost_equal(output["ustar_wm"], ref["ustrwm"], decimal=3)
    npt.assert_almost_equal(output["ustar_wc"], ref["ustrr"], decimal=3)
    npt.assert_almost_equal(output["f_wc"], ref["fwc"], decimal=3)
