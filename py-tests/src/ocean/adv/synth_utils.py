"""
Shared helpers for synthesizing controlled ADV bursts with analytically known
wave and turbulence ground truth, used by the ADV test modules.
"""

import numpy as np

from utils.wave_utils import get_wavenumber
from utils.constants import GRAVITATIONAL_ACCELERATION as g


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
    u_d_mean=0.0,
    epsilon=None,
    f_cut_low=0.05,
    seed=0,
):
    """
    Synthesize a burst with a monochromatic linear surface wave
    superposed on wave-uncorrelated, Gaussian turbulence.

    Returns u, v, w (m/s), p (dbar), and a dict of ground-truth quantities.

    Turbulence model (controlled by `epsilon`):
    - `epsilon=None`: white Gaussian turbulence with the prescribed 3x3 Reynolds
      stress tensor imposed via Cholesky factorization.
    - `epsilon=float`: isotropic Kolmogorov turbulence in the Taylor
      (frozen-turbulence) limit with mean flow `u_d_mean` along +x and
      dissipation `epsilon`, matching Gerbi et al. (2009) Eqs. 13-14:
          S_uu(w) = (9/55)  a e^(2/3) / U_d^(2/3) |w|^(-5/3)
          S_vv(w) = S_ww(w) = (12/55) a e^(2/3) / U_d^(2/3) |w|^(-5/3)
      with a = 1.5 (Kolmogorov constant). The sigma_* kwargs are ignored in this
      mode; the resulting turbulent variances are integrated from the synthesized
      spectra above `f_cut_low` (low-frequency cutoff preventing the -5/3
      integral from diverging).

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
        if u_d_mean <= 0:
            raise ValueError("u_d_mean must be > 0 when epsilon is provided (Taylor hypothesis)")
        alpha = 1.5
        f_bin = np.fft.fftfreq(N, d=1.0 / fs)
        omega_abs = np.abs(2 * np.pi * f_bin)
        coef_uu = (9.0 / 55.0) * alpha * epsilon ** (2 / 3) / u_d_mean ** (2 / 3) * 2 * np.pi
        coef_ww = (12.0 / 55.0) * alpha * epsilon ** (2 / 3) / u_d_mean ** (2 / 3) * 2 * np.pi
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

        u_turb = _synth(P_uu_f)
        v_turb = _synth(P_ww_f)
        w_turb = _synth(P_ww_f)
        df = fs / N
        sigma_uu_out = float(np.sum(P_uu_f) * df)
        sigma_ww_out = float(np.sum(P_ww_f) * df)
        sigma_vv_out = sigma_ww_out
        sigma_uw_out = 0.0

    u = u_d_mean + u_wave + u_turb
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
