"""Shared helpers for generating synthetic bursts with analytically known wave
and turbulence ground truth, used by the ADV, Sonic, and ADCP test modules."""

import numpy as np
import scipy.signal as sig
from utils.wave_utils import get_wavenumber
from utils.constants import GRAVITATIONAL_ACCELERATION as g, VON_KARMAN as KAPPA, WATER_DENSITY as rho0


def _timeseries_from_spectrum(P, N, fs, rng, phases=None, phase_offset=0.0):
    """Build a real time series with PSD `P` via random-phase IFFT.

    If `phases` (length N//2 - 1) is supplied, it is used in place of newly
    drawn random phases — pass the same array to multiple calls to make the resulting
    time series phase-coherent. `phase_offset` (radians) is added to every
    positive-frequency phase, useful for generating a 90-deg-shifted channel
    (e.g. vertical wave velocity vs. surface elevation) that remains
    phase-coherent with the unshifted channels.
    """
    amp = np.sqrt(N * fs * P)
    X = np.zeros(N, dtype=complex)
    nh = N // 2
    if phases is None:
        phases = rng.uniform(0, 2 * np.pi, nh - 1)
    X[1:nh] = amp[1:nh] * np.exp(1j * (phases + phase_offset))
    X[nh + 1 :] = np.conj(X[1:nh][::-1])
    if N % 2 == 0:
        X[nh] = amp[nh] * (1.0 if rng.random() > 0.5 else -1.0)
    return np.fft.ifft(X).real


def pierson_moskowitz(Hs, fp, h=10, z=0, fs=32, duration_s=600, seed=0):
    """Pierson-Moskowitz surface elevation and the associated linear-wave
    kinematics at vertical coordinate z (z=0 at the free surface, z=-h at the
    bed) in water depth h. u is the horizontal velocity along the direction
    of wave propagation; p is the total (hydrostatic + wave-induced) pressure.

    PSDs of u, w, and p are derived per-frequency from P_etaeta via linear-
    wave transfer functions:
        |H_u(f)|^2 = (omega * cosh(k(z+h)) / sinh(kh))^2
        |H_w(f)|^2 = (omega * sinh(k(z+h)) / sinh(kh))^2
        |H_p(f)|^2 = (rho*g * cosh(k(z+h)) / cosh(kh))^2
    """
    N = int(fs * duration_s)
    f = np.fft.fftfreq(N, d=1.0 / fs)
    rng = np.random.default_rng(seed)

    pos = f > 0
    P_etaeta = np.zeros_like(f)
    P_etaeta[pos] = (5 / 16) * Hs**2 * fp**4 * f[pos] ** (-5) * np.exp(-5 / 4 * (f[pos] / fp) ** (-4))

    omega = 2 * np.pi * f[pos]
    k = get_wavenumber(omega, h)
    ###########################################################
    # Hyperbolic transfer-function ratios written so they stay
    # finite at large kh (deep water). For z in [-h, 0]:
    #   cosh(k(z+h))/sinh(kh) = (e^{kz}+e^{-k(z+2h)}) / (1-e^{-2kh})
    #   sinh(k(z+h))/sinh(kh) = (e^{kz}-e^{-k(z+2h)}) / (1-e^{-2kh})
    #   cosh(k(z+h))/cosh(kh) = (e^{kz}+e^{-k(z+2h)}) / (1+e^{-2kh})
    ###########################################################
    e_kz = np.exp(k * z)
    e_neg = np.exp(-k * (z + 2 * h))
    e_2kh = np.exp(-2 * k * h)
    r_cs = (e_kz + e_neg) / (1 - e_2kh)  # cosh(k(z+h))/sinh(kh)
    r_ss = (e_kz - e_neg) / (1 - e_2kh)  # sinh(k(z+h))/sinh(kh)
    r_cc = (e_kz + e_neg) / (1 + e_2kh)  # cosh(k(z+h))/cosh(kh)

    H_u2 = (omega * r_cs) ** 2
    H_w2 = (omega * r_ss) ** 2
    H_p2 = (rho0 * g * r_cc) ** 2

    P_uu = np.zeros_like(f)
    P_ww = np.zeros_like(f)
    P_pp = np.zeros_like(f)
    P_uu[pos] = P_etaeta[pos] * H_u2
    P_ww[pos] = P_etaeta[pos] * H_w2
    P_pp[pos] = P_etaeta[pos] * H_p2

    # Share a single phase realization across channels so the
    # four time series are phase-coherent. Each is divided by 2
    # because the function takes a 2-sided spectrum
    phases = rng.uniform(0, 2 * np.pi, N // 2 - 1)
    eta = _timeseries_from_spectrum(P_etaeta / 2, N, fs, rng, phases=phases)
    u = _timeseries_from_spectrum(P_uu / 2, N, fs, rng, phases=phases)
    w = _timeseries_from_spectrum(P_ww / 2, N, fs, rng, phases=phases, phase_offset=np.pi / 2)
    # Pressure, adding hydrostatic component and converting to dbar
    p = _timeseries_from_spectrum(P_pp / 2, N, fs, rng, phases=phases)
    p += rho0 * g * np.abs(z)
    p /= 1e4
    return P_etaeta, eta, u, w, p



def generate_wave_turb_burst(
    fs=32,
    duration_s=600,
    a=0.10,
    f_wave=0.15,
    h=5.0,
    mab=1.0,
    rho=1020.0,
    sigma_uu=1.0e-2,
    sigma_vv=8.0e-3,
    sigma_ww=4.0e-3,
    sigma_uw=-1.0e-3,
    U=0.0,
    epsilon=None,
    f_cut_low=0.001,
    seed=0,
):
    """
    Synthesize a burst with a monochromatic linear surface wave
    superposed on wave-uncorrelated, Gaussian turbulence.

    Returns u, v, w (m/s), p (dbar), and a dict of ground-truth quantities.

    Turbulence model (controlled by `epsilon`):
    - `epsilon=None`: white Gaussian turbulence with the prescribed 3x3 Reynolds
      stress tensor imposed via Cholesky factorization.
    - `epsilon=float`: isotropic turbulence under the Taylor
      (frozen-turbulence) assumption with mean flow `U` along +x and
      dissipation `epsilon`, matching the theoretical spectra:
          S_uu(omega) = (9/55) alpha epsilon^(2/3) U^(2/3) omega^(-5/3)
          S_vv(omega) = S_ww(omega) = (12/55) alpha epsilon^(2/3) U^(2/3) omega^(-5/3)
      with alpha = 1.5 (Kolmogorov constant). The sigma_* kwargs are ignored in this
      mode; the resulting turbulent variances are integrated from the synthesized
      spectra above `f_cut_low`.

    Wave Reynolds stresses follow from linear wave theory for a unidirectional
    progressive wave (wave along +x, so v_wave = 0 and <u w>_wave = 0).
    """
    N = int(fs * duration_s)
    t = np.arange(N) / fs

    omega = 2 * np.pi * f_wave
    k = float(np.asarray(get_wavenumber(np.array([omega]), h)).ravel()[0])

    # Wave kinematics at height mab above the bed
    u_amp = a * omega * np.cosh(k * mab) / np.sinh(k * h)
    w_amp = a * omega * np.sinh(k * mab) / np.sinh(k * h)
    p_amp_pa = rho * g * a * np.cosh(k * mab) / np.cosh(k * h)

    phase = omega * t
    u_wave = u_amp * np.cos(phase)
    w_wave = w_amp * np.sin(phase)
    p_wave_dbar = (p_amp_pa / 1e4) * np.cos(phase)
    p_mean_dbar = rho * g * (h - mab) / 1e4

    rng = np.random.default_rng(seed)

    if epsilon is None:
        cov = np.array(
            [
                [sigma_uu, 0.0, sigma_uw],
                [0.0, sigma_vv, 0.0],
                [sigma_uw, 0.0, sigma_ww],
            ]
        )
        L = np.linalg.cholesky(cov)
        turb = L @ rng.standard_normal((3, N))
        u_turb, v_turb, w_turb = turb[0], turb[1], turb[2]
        sigma_uu_out, sigma_vv_out, sigma_ww_out = sigma_uu, sigma_vv, sigma_ww
        sigma_uw_out = sigma_uw
    else:
        if U <= 0:
            raise ValueError("U must be > 0 when epsilon is provided (Taylor frozen turbulence)")
        alpha = 1.5
        f_bin = np.fft.fftfreq(N, d=1.0 / fs)
        omega_abs = np.abs(2 * np.pi * f_bin)
        coef_uu = (9.0 / 55.0) * alpha * epsilon ** (2 / 3) * U ** (2 / 3) * 2 * np.pi
        coef_ww = (12.0 / 55.0) * alpha * epsilon ** (2 / 3) * U ** (2 / 3) * 2 * np.pi
        P_uu_f = np.zeros_like(omega_abs)
        P_ww_f = np.zeros_like(omega_abs)
        valid = omega_abs >= 2 * np.pi * f_cut_low
        P_uu_f[valid] = coef_uu * omega_abs[valid] ** (-5 / 3)
        P_ww_f[valid] = coef_ww * omega_abs[valid] ** (-5 / 3)

        u_turb = _timeseries_from_spectrum(P_uu_f, N, fs, rng)
        v_turb = _timeseries_from_spectrum(P_ww_f, N, fs, rng)
        w_turb = _timeseries_from_spectrum(P_ww_f, N, fs, rng)
        df = fs / N
        sigma_uu_out = float(np.sum(P_uu_f) * df)
        sigma_ww_out = float(np.sum(P_ww_f) * df)
        sigma_vv_out = sigma_ww_out
        sigma_uw_out = 0.0

    u = U + u_wave + u_turb
    v = v_turb
    w = w_wave + w_turb
    p = p_mean_dbar + p_wave_dbar

    truth = {
        "uu_wave": 0.5 * u_amp**2,
        "vv_wave": 0.0,
        "ww_wave": 0.5 * w_amp**2,
        "uw_wave": 0.0,
        "uv_wave": 0.0,
        "vw_wave": 0.0,
        "uu_turb": sigma_uu_out,
        "vv_turb": sigma_vv_out,
        "ww_turb": sigma_ww_out,
        "uw_turb": sigma_uw_out,
        "uv_turb": 0.0,
        "vw_turb": 0.0,
    }
    return u, v, w, p, truth


def generate_profile_burst(
    fs,
    duration_s,
    z,
    profile_type="wall_bounded",
    u_star=0.02,
    z0=1.0e-4,
    epsilon=1.0e-5,
    U=0.5,
    sigma_u_over_u_star=2.4,
    sigma_v_over_u_star=1.9,
    sigma_w_over_u_star=1.25,
    uw_over_u_star_sq=-1.0,
    f_cut_low=0.001,
    seed=0,
):
    """
    Synthesize a profile burst in xyz coordinates with known analytical statistics per-height.

    profile_type="wall_bounded":
        White-noise turbulence with a depth-independent prescribed Reynolds stress
        tensor and a log-layer mean U(z) = (u_star/kappa) log(z/z0).
        The truth `epsilon(z)` follows the log-layer balance u_star^3 / (kappa * z), but
        the synthesized turbulence is *spectrally white* (no Kolmogorov shape) and is
        intended for covariance/shear tests, not spectral dissipation fits.

    profile_type="isotropic":
        Kolmogorov isotropic turbulence with constant epsilon at every height,
        synthesized in frequency space under Taylor's frozen-turbulence hypothesis with
        mean U along +x. Mirrors the inner block of generate_wave_turb_burst.
        Use this for spectral dissipation tests. The true Reynolds stress tensor is
        diagonal (uw=0).

    Parameters
    ----------
    fs : float
        Sampling frequency (Hz).
    duration_s : float
        Burst duration (s).
    z : array-like
        Vertical coordinate (m above bed). For wall_bounded must be all positive.
    profile_type : {"wall_bounded", "isotropic"}
    u_star, z0 : float
        Friction velocity and roughness length for the wall_bounded mean profile.
    epsilon : float
        Dissipation rate for the isotropic case (m^2/s^3).
    U : float
        Mean flow used for Taylor frozen turbulence (m/s).
    sigma_*_over_u_star : float
        Normalized diagonal Reynolds stress components
    uw_over_u_star_sq : float
        Reynolds stress <u'w'> in units of u_star^2 (typically -1).
    f_cut_low : float
        Low-frequency cutoff for the spectral integration (Hz).
    seed : int
        RNG seed.

    Returns
    -------
    u1_xyz, u2_xyz, u3_xyz : np.ndarray, shape (n_heights, n_samples)
        Velocity components in xyz coordinates.
    truth : dict of np.ndarray, each shape (n_heights,)
        Keys: "U", "uu", "vv", "ww", "uw", "epsilon".
    """
    z = np.asarray(z, dtype=float)
    nz = len(z)
    N = int(fs * duration_s)
    rng = np.random.default_rng(seed)

    if profile_type == "wall_bounded":
        if np.any(z <= 0):
            raise ValueError("wall_bounded profile requires strictly positive z")
        U_z = (u_star / KAPPA) * np.log(z / z0)
        eps_z = u_star**3 / (KAPPA * z)
        sigma_uu = (sigma_u_over_u_star * u_star) ** 2
        sigma_vv = (sigma_v_over_u_star * u_star) ** 2
        sigma_ww = (sigma_w_over_u_star * u_star) ** 2
        sigma_uw = uw_over_u_star_sq * u_star**2

        cov = np.array(
            [
                [sigma_uu, 0.0, sigma_uw],
                [0.0, sigma_vv, 0.0],
                [sigma_uw, 0.0, sigma_ww],
            ]
        )
        L = np.linalg.cholesky(cov)
        u1 = np.empty((nz, N))
        u2 = np.empty((nz, N))
        u3 = np.empty((nz, N))
        for ii in range(nz):
            turb = L @ rng.standard_normal((3, N))
            u1[ii, :] = U_z[ii] + turb[0]
            u2[ii, :] = turb[1]
            u3[ii, :] = turb[2]

        truth = {
            "U": U_z,
            "uu": np.full(nz, sigma_uu),
            "vv": np.full(nz, sigma_vv),
            "ww": np.full(nz, sigma_ww),
            "uw": np.full(nz, sigma_uw),
            "epsilon": eps_z,
        }
        return u1, u2, u3, truth

    if profile_type == "isotropic":
        if U <= 0:
            raise ValueError("U must be > 0 for isotropic Taylor synthesis")
        alpha = 1.5
        f_bin = np.fft.fftfreq(N, d=1.0 / fs)
        omega_abs = np.abs(2 * np.pi * f_bin)
        coef_uu = (9.0 / 55.0) * alpha * epsilon ** (2 / 3) * U ** (2 / 3) * 2 * np.pi
        coef_ww = (12.0 / 55.0) * alpha * epsilon ** (2 / 3) * U ** (2 / 3) * 2 * np.pi
        P_uu_f = np.zeros_like(omega_abs)
        P_ww_f = np.zeros_like(omega_abs)
        valid = omega_abs >= 2 * np.pi * f_cut_low
        P_uu_f[valid] = coef_uu * omega_abs[valid] ** (-5 / 3)
        P_ww_f[valid] = coef_ww * omega_abs[valid] ** (-5 / 3)

        u1 = np.empty((nz, N))
        u2 = np.empty((nz, N))
        u3 = np.empty((nz, N))
        for ii in range(nz):
            u1[ii, :] = U + _timeseries_from_spectrum(P_uu_f, N, fs, rng)
            u2[ii, :] = _timeseries_from_spectrum(P_ww_f, N, fs, rng)
            u3[ii, :] = _timeseries_from_spectrum(P_ww_f, N, fs, rng)

        df = fs / N
        sigma_uu_out = float(np.sum(P_uu_f) * df)
        sigma_ww_out = float(np.sum(P_ww_f) * df)
        truth = {
            "U": np.full(nz, U),
            "uu": np.full(nz, sigma_uu_out),
            "vv": np.full(nz, sigma_ww_out),
            "ww": np.full(nz, sigma_ww_out),
            "uw": np.zeros(nz),
            "epsilon": np.full(nz, epsilon),
        }
        return u1, u2, u3, truth

    raise ValueError(f"Unknown profile_type {profile_type!r}")
