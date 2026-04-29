"""Shared helpers for generating synthetic bursts with analytically known wave
and turbulence ground truth, used by the ADV, Sonic, and ADCP test modules."""

import numpy as np

from utils.wave_utils import get_wavenumber
from utils.constants import GRAVITATIONAL_ACCELERATION as g, VON_KARMAN as KAPPA


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

        def _synth(P):
            # Helper function to generate synthetic timeseries from power spectrum P
            amp = np.sqrt(N * fs * P)
            X = np.zeros(N, dtype=complex)
            nh = N // 2
            ph = rng.uniform(0, 2 * np.pi, nh - 1)
            X[1:nh] = amp[1:nh] * np.exp(1j * ph)
            X[nh + 1 :] = np.conj(X[1:nh][::-1])
            if N % 2 == 0:
                X[nh] = amp[nh] * (1.0 if rng.random() > 0.5 else -1.0)
            return np.fft.ifft(X).real

        u_turb = _synth(P_uu_f)
        v_turb = _synth(P_ww_f)
        w_turb = _synth(P_ww_f)
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

        def _synth(P):
            amp = np.sqrt(N * fs * P)
            X = np.zeros(N, dtype=complex)
            nh = N // 2
            ph = rng.uniform(0, 2 * np.pi, nh - 1)
            X[1:nh] = amp[1:nh] * np.exp(1j * ph)
            X[nh + 1 :] = np.conj(X[1:nh][::-1])
            if N % 2 == 0:
                X[nh] = amp[nh] * (1.0 if rng.random() > 0.5 else -1.0)
            return np.fft.ifft(X).real

        u1 = np.empty((nz, N))
        u2 = np.empty((nz, N))
        u3 = np.empty((nz, N))
        for ii in range(nz):
            u1[ii, :] = U + _synth(P_uu_f)
            u2[ii, :] = _synth(P_ww_f)
            u3[ii, :] = _synth(P_ww_f)

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
