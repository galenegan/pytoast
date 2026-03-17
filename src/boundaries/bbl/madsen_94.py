import numpy as np
from utils.rotate_utils import min_angle
from utils.constants import VON_KARMAN as kappa


def _fwc(relative_roughness, c_mu):
    # Wave-current friction factor
    if relative_roughness <= 0.2:
        print(f"Warning: Relative roughness {relative_roughness} is out of range [0.2, 10000]")
        return c_mu * np.exp(7.02 * 0.2 ** (-0.078) - 8.82)
    elif (relative_roughness > 0.2) & (relative_roughness <= 100):
        return c_mu * np.exp(7.02 * relative_roughness ** (-0.078) - 8.82)
    elif (relative_roughness > 100) & (relative_roughness <= 10000):
        return c_mu * np.exp(5.61 * relative_roughness ** (-0.109) - 7.30)
    else:
        print(f"Warning: Relative roughness {relative_roughness} is out of range [0.2, 10000]")
        return c_mu * np.exp(5.61 * 10000 ** (-0.109) - 7.30)


def madsen(ub_r, omega_r, uc_r, phi_c, phi_wr, z_r, kN, max_iter=10):
    """

    Parameters
    ----------
    ub_r
    omega_r
    u_c
    phi_c
    phi_wr
    z_r
    kN
    max_iter

    Returns
    -------

    """
    phi_wc = np.deg2rad(min_angle(phi_c - phi_wr))
    z0 = kN / 30

    iter = 0
    delta_fwc = np.inf
    c_mu = 1.0  # c_mu when mu = tau_c / tau_wr = 0
    while (iter < max_iter) and (delta_fwc > 0.01):
        relative_roughness = (c_mu * ub_r) / (kN * omega_r)

        # Wave-current friction factor
        f_wc = _fwc(relative_roughness, c_mu)

        # All the friction velocities
        ustar_wm = np.sqrt(0.5 * f_wc * ub_r**2)
        ustar_r = c_mu * ustar_wm**2

        if relative_roughness > 8:
            delta_wc = 2 * kappa * ustar_r / omega_r
        else:
            delta_wc = kN

        ustar_c = (
            (ustar_r / 2)
            * (np.log(z_r / delta_wc) / np.log(delta_wc / z0))
            * (-1 + np.sqrt(1 + (4 * kappa * np.log(delta_wc / z0) / np.log(z_r / delta_wc) ** 2) * (uc_r / ustar_r)))
        )

        mu = (ustar_c / ustar_wm) ** 2
        c_mu = np.sqrt(1 + 2 * mu * np.abs(np.cos(phi_wc)) + mu**2)
        f_wc_new = _fwc(relative_roughness, c_mu)
        delta_fwc = np.abs(f_wc - f_wc_new) / f_wc

    out = {
        "ustar_c": ustar_c,
        "ustar_wm": ustar_wm,
        "ustar_wc": ustar_r,
        "f_wc": f_wc_new,
    }
    return out
