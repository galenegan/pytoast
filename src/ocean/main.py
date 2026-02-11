


if __name__ == "__main__":
    import time

    t0 = time.time()
    mean_depth = 13
    m_below_surface = np.linspace(1.8, 7.2, 6)
    mabs = [mean_depth - mbs for mbs in m_below_surface]

    # Testing this out
    files = glob.glob("/Users/ea-gegan/Documents/gitrepos/tke-budget/data/adv_fall/*.mat")
    files.sort()
    files = files[:5]

    # Name map:
    name_map = {"u": "E", "v": "N", "w": "w", "p": "P2", "time": "dn"}
    adv = ADV.from_raw(files, name_map, fs=32, z=mabs, zarr_save_path="~/Desktop/adv_zarr_test")
    adv.despike()
    theta = adv.get_principal_axis()
    vel_maj, vel_min = adv.rotate_velocity(theta)
    adv.u, adv.v = vel_maj, vel_min

    wavestats = adv.directional_wave_statistics()

    # eps, noise, quality_flag = adv.dissipation(f_low=1.2, f_high=15, fs=32)
    # print(eps.values[:, 0])
    cov = adv.covariance(
        method="cov",
        fs=32,
    )
    cov0 = cov["uw"][:, 4].values
    cov = adv.covariance(method="spectral_integral", parallel=False, fs=32)
    cov1 = cov["uw"][:, 4].values

    import matplotlib.pyplot as plt

    one = np.linspace(np.nanmin(cov1), np.nanmax(cov1), 100)
    plt.plot(cov1, cov0, "o")
    plt.plot(one, one, "-")
    plt.show()
    # print(cov.values)
    # t1 = time.time()
    # print(f"finished processing 20 files in {t1 - t0:.2f} seconds")
    # adv0 = ADV.from_saved_zarr("~/Desktop/adv_zarr_test")
    # test2_0 = adv0.u[9, 0, :].values

    # import matplotlib.pyplot as plt
    #
    # plt.plot(vel_maj[9, 0, :].values)
    # plt.plot(adv.u[9, 0, :].values)
    # plt.show()
