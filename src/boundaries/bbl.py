import numpy as np
from scipy.special import jv, hankel1
from scipy.optimize import brentq, newton
from utils.constants import VON_KARMAN as kappa
from utils.rotate_utils import min_angle

# Default physical/model constants for Styles model (specific to this module for testing against Matlab source)
_NU = 1.19e-6  # kinematic viscosity of seawater @ 15 C (m^2/s)
_G = 9.81  # gravity (m/s^2)
_S = 2.65  # relative sediment density
_BETA = 0.7  # closure constant
_ALPHA = 0.3  # closure constant
_CON = 6.4  # closure constant
_KBR_DEF = 0.03  # default ripple roughness (m)


def _fwc_m94(relative_roughness, c_mu):
    # Wave-current friction factor (Eqs 32 - 33, Madsen 1994)
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


def _shields_critical(star):
    """Critical Shields parameter for initiation of sediment motion
    (shldc.m)."""
    if star < 1.5:
        return 0.0932 * star ** (-0.707)
    elif star < 4.0:
        return 0.0848 * star ** (-0.473)
    elif star < 10.0:
        return 0.0680 * star ** (-0.314)
    elif star < 34.0:
        return 0.033
    elif star < 270:
        return 0.0134 * star**0.255
    else:
        return 0.056


def _phi2_1(zeta_0, zeta_1, mp):
    """Nondimensional wave shear via Kelvin/Bessel functions (phi2_1.m).

    Parameters
    ----------
    zeta_0 : float
        Nondimensional roughness length (1 / (kappa * Ro))
    zeta_1 : float
        Nondimensional inner wave boundary layer height (= alpha)
    mp : complex
        Boundary parameter (delta + i*delta, delta = 1/sqrt(2*zeta_1))
    """
    x = np.array([2.0 * np.sqrt(zeta_0), 2.0 * np.sqrt(zeta_1)])
    y = x * np.exp(3j * np.pi / 4)

    J0 = jv(0, y)
    J1 = jv(1, y)
    H0 = hankel1(0, y)
    H1 = hankel1(1, y)

    ber = J0.real
    bei = J0.imag
    ker = (0.5 * np.pi * 1j * H0).real
    kei = (0.5 * np.pi * 1j * H0).imag
    ber1 = J1.real
    bei1 = J1.imag
    ker1 = (0.5 * np.pi * 1j * H1).real
    kei1 = (0.5 * np.pi * 1j * H1).imag

    berp = (ber1 + bei1) / np.sqrt(2)
    beip = (-ber1 + bei1) / np.sqrt(2)
    kerp = (ker1 + kei1) / np.sqrt(2)
    keip = (-ker1 + kei1) / np.sqrt(2)

    bnot = ber[0] + 1j * bei[0]
    knot = ker[0] + 1j * kei[0]
    bnotp = (berp[0] + 1j * beip[0]) / np.sqrt(zeta_0)
    knotp = (kerp[0] + 1j * keip[0]) / np.sqrt(zeta_0)

    b1 = ber[1] + 1j * bei[1]
    k1 = ker[1] + 1j * kei[1]
    b1p = (berp[1] + 1j * beip[1]) / np.sqrt(zeta_1)
    k1p = (kerp[1] + 1j * keip[1]) / np.sqrt(zeta_1)

    ll = mp * b1 + b1p
    nn = mp * k1 + k1p
    argi = bnotp * nn / (bnot * nn - knot * ll) + knotp * ll / (knot * ll - bnot * nn)
    gammai = -kappa * zeta_0 * argi
    return np.abs(gammai)


def _phi(zeta_0, zeta_1, mp):
    """Bessel solution or linear approximation depending on zeta_1/zeta_0."""
    if zeta_1 / zeta_0 > 1:
        return _phi2_1(zeta_0, zeta_1, mp)
    return np.abs(-kappa * zeta_1 * mp)


def _pwave(ab_over_z0, ub_over_ustar_wm, zeta_1, mp, max_iter=40, tol=1e-4):
    """Pure-wave ub/ustar_wm via secant method (pwave.m)."""

    def f(x):
        zeta_0 = 1.0 / (kappa * (ab_over_z0 / x))
        phi = _phi(zeta_0, zeta_1, mp)
        return x - 1.0 / phi

    return newton(f, x0=ub_over_ustar_wm, x1=ub_over_ustar_wm * 0.5, tol=tol, maxiter=max_iter)


def _bstress2(ub_over_ustar_wc, ab_over_z0, zr_over_z0, ub_over_kappa_ur, theta, alpha_loc, zeta_1, mp):
    """BBL stress parameters and ub / ustar_wc residual fx (bstress2.m).

    Returns
    -------
    (Ro, mu, epsilon, z1_over_z0, z2_over_z0, zr_over_z1, zr_over_z2, fx)
    """
    Ro = ab_over_z0 / ub_over_ustar_wc
    zeta_0 = 1.0 / (kappa * Ro)
    phi = _phi(zeta_0, zeta_1, mp)

    mu = np.sqrt(ub_over_ustar_wc * phi)
    eps2 = -(mu**2) * np.abs(np.cos(theta)) + np.sqrt(1.0 - mu**4 * np.sin(theta) ** 2)
    epsilon = np.sqrt(eps2)

    Ro_r = Ro / zr_over_z0
    zr_over_z1 = 1.0 / (alpha_loc * kappa * Ro_r)  # Invoking alpha_loc = z1 / L_cw; L_cw = kappa ustar_wc a_b / u_b
    zr_over_z2 = epsilon * zr_over_z1
    z1_over_z0 = alpha_loc * kappa * Ro
    z2_over_z0 = z1_over_z0 / epsilon

    # Six velocity-profile cases (Table 1, Styles et al. (2017)). Each of these expressions is the deviation between
    # sigma = ub / ustar_wc and the estimate of ub / ustar_wc (Eq. 2-16)
    if zr_over_z2 > 1 and z1_over_z0 > 1:
        fx = (
            ub_over_kappa_ur * epsilon * (np.log(zr_over_z2) + 1 - epsilon + epsilon * np.log(z1_over_z0))
            - ub_over_ustar_wc
        )
    elif zr_over_z2 <= 1 and zr_over_z1 > 1 and z1_over_z0 > 1:
        fx = ub_over_kappa_ur * epsilon**2 * (zr_over_z1 - 1 + np.log(z1_over_z0)) - ub_over_ustar_wc
    elif zr_over_z1 <= 1 and z1_over_z0 > 1:
        fx = ub_over_kappa_ur * epsilon**2 * np.log(zr_over_z0) - ub_over_ustar_wc
    elif zr_over_z2 > 1 and z1_over_z0 <= 1 and z2_over_z0 > 1:
        fx = ub_over_kappa_ur * epsilon * (np.log(zr_over_z2) + 1 - 1.0 / z2_over_z0) - ub_over_ustar_wc
    elif zr_over_z2 <= 1 and zr_over_z1 > 1 and z1_over_z0 <= 1 and z2_over_z0 > 1:
        fx = ub_over_kappa_ur * epsilon**2 * (zr_over_z1 - 1.0 / z1_over_z0) - ub_over_ustar_wc
    elif zr_over_z2 > 1 and z2_over_z0 <= 1:
        fx = ub_over_kappa_ur * epsilon * np.log(zr_over_z0) - ub_over_ustar_wc
    else:
        raise ValueError(
            f"No velocity profile case matched: "
            f"zr_over_z1={zr_over_z1:.4g}, zr_over_z2={zr_over_z2:.4g}, z1_over_z0={z1_over_z0:.4g}, z2_over_z0={z2_over_z0:.4g}"
        )

    return Ro, mu, epsilon, z1_over_z0, z2_over_z0, zr_over_z1, zr_over_z2, fx


def madsen(ub_r, omega_r, uc_r, phi_c, phi_wr, z_r, kN, max_iter=10, tol=1e-4):
    """

    Parameters
    ----------
    ub_r
    omega_r
    uc_r
    phi_c
    phi_wr
    z_r
    kN
    max_iter
    tol

    Returns
    -------

    """
    phi_wc = np.deg2rad(min_angle(phi_c - phi_wr))
    z0 = kN / 30

    iter = 0
    delta_fwc = np.inf
    c_mu = 1.0  # c_mu when mu = tau_c / tau_wr = 0
    while (iter < max_iter) and (delta_fwc > tol):
        relative_roughness = (c_mu * ub_r) / (kN * omega_r)

        # Wave-current friction factor
        f_wc = _fwc_m94(relative_roughness, c_mu)

        # All the friction velocities
        ustar_wm = np.sqrt(0.5 * f_wc * ub_r**2)
        ustar_r = np.sqrt(c_mu * ustar_wm**2)

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
        f_wc_new = _fwc_m94(relative_roughness, c_mu)
        delta_fwc = np.abs(f_wc - f_wc_new) / f_wc

    out = {
        "ustar_c": ustar_c,
        "ustar_wm": ustar_wm,
        "ustar_wc": ustar_r,
        "f_wc": f_wc_new,
    }
    return out


def styles(
    ub,
    ab,
    ur,
    zr,
    deg,
    d_median,
    s=_S,
    nu=_NU,
    g=_G,
    beta=_BETA,
    Alpha=_ALPHA,
    Con=_CON,
    kbr_def=_KBR_DEF,
    max_iter=50,
    tol=1e-4,
):
    """Styles et al.

    (2017) combined wave-current bottom boundary layer model. This is a python port of the Matlab source code (https://cirp.usace.army.mil/products/bblm.php). It has been edited to remove unnecessary input parameters and
    some variable names have been changed for clarity. The custom bisection and secant solvers have also been
    replaced with scipy.optimize.brentq and scipy.optimize.newton, respectively. The output on the provided test data is
    within an acceptable tolerance despite these changes (see py-tests/). Finally, all inputs and outputs use SI units
    (m) rather than CGS units (centimeters, grams, seconds) like the source.

    Parameters
    ----------
    ub : float
        Bottom wave orbital velocity amplitude (m/s)
    ab : float
        Bottom wave excursion amplitude (m)
    ur : float
        Mean current speed at height zr (m/s)
    zr : float
        Height above bed where ur is measured (m)
    deg : float
        Angle between wave propagation and current direction (degrees)
    d_median : float
        Median grain diameter (m)
    s : float
        Relative sediment density (default 2.65)
    nu : float
        Kinematic viscosity (m^2/s; default 1.19e-6 for seawater at 15 C)
    g : float
        Gravitational acceleration (m/s^2; default 9.81)
    beta : float
        Closure constant (default 0.7)
    Alpha : float
        Closure constant (default 0.3)
    Con : float
        Closure constant (default 6.4)
    kbr_def : float
        Default ripple roughness used when Psi < psicr (m; default 0.03)
    max_iter : int
        Maximum bisection iterations (default 50)
    tol : float
        Bisection convergence tolerance (default 1e-4)

    Returns
    -------
    dict with keys:
        Ro        - internal friction Rossby number (Ab / (z0 * ub / ustar_wc))
        mu        - sqrt(ub_over_ustar_wc * phi); wave/combined stress ratio parameter
        epsilon   - ustar_c / ustar_wc
        z1_over_z0     - inner wave BBL height / hydraulic roughness
        z2_over_z0     - outer wave BBL height / hydraulic roughness
        zr_over_z1     - measurement height / inner BBL height
        zr_over_z2     - measurement height / outer BBL height
        kbs       - suspended sediment roughness (m)
        kbr       - ripple roughness (m)
        z0      - hydraulic roughness length (m)
        ustar_wm  - maximum wave shear velocity (m/s)
        ustar_c   - time-averaged current shear velocity (m/s)
        ustar_wc  - combined wave-current shear velocity (m/s)
    """
    # Skin friction Shields parameter (Madsen formula)
    arg_ole = ab / d_median
    fwc_skn = np.exp(5.61 * arg_ole ** (-0.109) - 7.30)
    psi_norm = (s - 1) * g * d_median
    psi = 0.5 * fwc_skn * (1.42 * ub) ** 2 / psi_norm

    # Critical Shields parameter
    star = d_median / (4 * nu) * np.sqrt(g * d_median * (s - 1))
    psicr = _shields_critical(star)

    # Ripple geometry and bottom roughness
    if psi - psicr <= 0:
        kbr = kbr_def
    else:
        chi = 4.0 * nu * ub**2 / (d_median * ((s - 1) * g * d_median) ** 1.5)
        if chi < 2:
            eta = ab * 0.30 * chi ** (-0.39)
        else:
            eta = ab * 0.45 * chi ** (-0.99)
        kbr = Con * eta

    kbs = ab * 0.0655 * (ub**2 / ((s - 1) * g * ab)) ** 1.4
    kb = d_median + kbr + kbs
    z0 = kb / 30.0

    # Derived input parameters
    omega = ub / ab
    theta = np.deg2rad(deg)
    ab_over_z0 = ab / z0
    zr_over_z0 = zr / z0
    ub_over_kappa_ur = ub / (kappa * ur)

    # Nondimensional heights -- using notation for zeta = z / L_cw from the paper rather than z0p from source code
    alpha_loc = Alpha * (1.0 + beta * kb / ab)
    zeta_1 = alpha_loc
    delta = 1.0 / np.sqrt(2.0 * zeta_1)
    mp = delta + 1j * delta

    # Bisection to find ub_over_ustar_wc

    lower_bound = 1e-6
    # Upper bound: pure-wave limit (pwave.m), uses empirical friction factor parameterization to estimate ub / ustar_wm
    t1 = -alpha_loc * kappa * zeta_1
    if ab_over_z0 < 6.25:
        ub_over_ustar_wm = 1.0 / abs(t1 * mp)
    elif ab_over_z0 < 10:
        ub_over_ustar_wm = np.exp(1.488) * ab_over_z0 ** (-0.653) * ab_over_z0 ** (0.185 * np.log(ab_over_z0))
    elif ab_over_z0 < 100:
        ub_over_ustar_wm = np.exp(0.4599) * ab_over_z0**0.1977 * ab_over_z0 ** (0.0085 * np.log(ab_over_z0))
    else:
        ub_over_ustar_wm = np.exp(0.13996) * ab_over_z0**0.3539 * ab_over_z0 ** (-0.0106 * np.log(ab_over_z0))

    upper_bound = _pwave(ab_over_z0, ub_over_ustar_wm, zeta_1, mp)

    # Wrapper function for the bisection with a cache for the extra variables we want
    _cache = [None]

    def _brent_f(x):
        result = _bstress2(x, ab_over_z0, zr_over_z0, ub_over_kappa_ur, theta, alpha_loc, zeta_1, mp)
        _cache[0] = result
        return result[-1]  # fx

    brentq(_brent_f, a=upper_bound, b=lower_bound, maxiter=max_iter, xtol=tol)
    Ro, mu, epsilon, z1_over_z0, z2_over_z0, zr_over_z1, zr_over_z2, _ = _cache[0]

    ustar_wc = Ro * z0 * omega
    ustar_c = epsilon * ustar_wc
    ustar_wm = mu * ustar_wc

    out = {
        "Ro": Ro,
        "mu": mu,
        "epsilon": epsilon,
        "z1_over_z0": z1_over_z0,
        "z2_over_z0": z2_over_z0,
        "zr_over_z1": zr_over_z1,
        "zr_over_z2": zr_over_z2,
        "kbs": kbs,
        "kbr": kbr,
        "z0": z0,
        "ustar_wm": ustar_wm,
        "ustar_c": ustar_c,
        "ustar_wc": ustar_wc,
    }

    return out
