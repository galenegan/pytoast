from ocean.adv import ADV
import time
import glob
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    t0 = time.time()

    mean_depth = 13
    m_below_surface = np.linspace(1.8, 7.2, 6)
    mabs = [mean_depth - mbs for mbs in m_below_surface]

    # Testing this out
    files = glob.glob("/Users/ea-gegan/Documents/gitrepos/tke-budget/data/adv_fall/*.mat")
    files.sort()
    files = files[200:210]

    # Name map:
    name_map = {"u1": "E", "u2": "N", "u3": "w", "p": "P2", "time": "dn"}
    adv = ADV(files, name_map, fs=32, z=mabs, source_coords="enu", orientation="up")

    T = np.array([[2896, 2896, 0], [-2896, 2896, 0], [-2896, -2896, 5792]], dtype=float)

    # If necessary, scale the transformation matrix to floating point values
    T /= 4096.0

    pre_opts = {
        "despike": {"method": "threshold", "threshold_min": -0.2, "threshold_max": 0.2},
        "rotate": {
            "flow_rotation": "align_streamwise",
            # "coords_out": "xyz",
            # "transformation_matrices": [T] * 6,
            # "constant_hpr": [(0.0, 1.0, 1.1)] * 6,
        },
    }
    adv.set_preprocess_opts(pre_opts)

    uwb = np.empty((adv.n_bursts * 6,))
    uwp = np.empty((adv.n_bursts * 6,))
    for ii in range(adv.n_bursts):
        burst = adv.load_burst(ii)
        cov1 = adv.covariance(burst, method="benilov")
        cov2 = adv.covariance(burst, method="phase")

        uwb[ii * 6 : (ii + 1) * 6] = cov1["uw_turb"]
        uwp[ii * 6 : (ii + 1) * 6] = cov2["uw_turb"]
    t1 = time.time()
    print(f"Time elapsed: {t1 - t0:.2f} seconds")

    plt.plot(uwb, uwp, "o")
    plt.show()
