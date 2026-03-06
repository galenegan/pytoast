import numpy as np
from scipy.stats import median_abs_deviation
from utils.interp_utils import interp_rows


def threshold(
    u: np.ndarray,
    threshold_min: float = -3.0,
    threshold_max: float = 3.0,
) -> np.ndarray:
    """
    Threshold-based despiking

    Parameters
    ----------
    u : np.ndarray
        Velocity array to despike
    threshold_min : float, optional
        Value below which samples are set to `np.nan` and interpolated over
    threshold_max : float, optional
        Value above which samples are set to `np.nan` and interpolated over

    Returns
    -------
    u_out : np.ndarray
        Velocity array with spikes removed and interpolated over

    """
    u_out = u.copy()
    bad_rows = (u_out < threshold_min) | (u_out > threshold_max)
    u_out[bad_rows] = np.nan
    interp_rows(u_out)
    return u_out


def goring_nikora(
    u: np.ndarray,
    remaining_spikes: int = 5,
    max_iter: int = 10,
    robust_statistics: bool = False,
) -> np.ndarray:
    """
    Implements the Goring & Nikora (2002) phase-space de-spiking algorithm,
    returning modified velocity array

    Parameters
    ----------
    u : np.ndarray
        Velocity array to despike.

    remaining_spikes : int
        Iterations will stop once there are `remaining_spikes` or fewer bad samples

    max_iter : int
        Maximum number of iterations

    robust_statistics : bool
        If True, ellipsoid centers will be based on the median and axis lengths will be based on median absolute
        deviation as suggested by Wahl (2003). If False, mean and standard deviation are used, consistent with the
        original Goring & Nikora implementation.

    Returns
    -------
    u_out : np.ndarray
        Velocity array with spikes removed and interpolated over

    References
    ----------
    Goring, D. G., & Nikora, V. I. (2002). Despiking acoustic Doppler velocimeter data. Journal of hydraulic
        engineering, 128(1), 117-126.
    Wahl, T. L. (2003). Discussion of "Despiking acoustic doppler velocimeter data" by
        Derek G. Goring and Vladimir I. Nikora. Journal of Hydraulic Engineering, 129(6), 484-487.

    """

    def flag_bad_indices(u: np.ndarray) -> np.ndarray:
        """Flag spikes in a 2D array (n_heights, n_samples) using phase-space method."""
        # Gradients along time axis
        du = np.gradient(u, axis=1) / 2
        du2 = np.gradient(du, axis=1) / 2

        # Per-row statistics
        if robust_statistics:
            sigma_u = median_abs_deviation(u, axis=1, nan_policy="omit")
            sigma_du = median_abs_deviation(du, axis=1, nan_policy="omit")
            sigma_du2 = median_abs_deviation(du2, axis=1, nan_policy="omit")
            u_bar = np.nanmedian(u, axis=1)
            du_bar = np.nanmedian(du, axis=1)
            du2_bar = np.nanmedian(du2, axis=1)
        else:
            sigma_u = np.nanstd(u, axis=1)
            sigma_du = np.nanstd(du, axis=1)
            sigma_du2 = np.nanstd(du2, axis=1)
            u_bar = np.nanmean(u, axis=1)
            du_bar = np.nanmean(du, axis=1)
            du2_bar = np.nanmean(du2, axis=1)

        # Expected absolute maximum
        n = u.shape[1]
        lam = np.sqrt(2 * np.log(n))

        # Rotation angle per row
        theta = np.arctan(np.nansum(u * du2, axis=1) / np.nansum(u**2, axis=1))

        # Ellipse axes (unrotated)
        a1 = lam * sigma_u
        b1 = lam * sigma_du
        a3 = lam * sigma_du
        b3 = lam * sigma_du2

        # Rotated ellipse axes via batched 2x2 solve
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        A = np.empty((u.shape[0], 2, 2))
        A[:, 0, 0] = cos_t**2
        A[:, 0, 1] = sin_t**2
        A[:, 1, 0] = sin_t**2
        A[:, 1, 1] = cos_t**2
        b_vec = np.stack([(lam * sigma_u) ** 2, (lam * sigma_du2) ** 2], axis=1)  # (n_heights, 2)
        x = np.linalg.solve(A, b_vec[:, :, None]).squeeze(-1)  # (n_heights, 2)
        a2 = np.sqrt(x[:, 0])
        b2 = np.sqrt(x[:, 1])

        # Broadcast all (n_heights,) stats to (n_heights, 1)
        u_bar = u_bar[:, np.newaxis]
        du_bar = du_bar[:, np.newaxis]
        du2_bar = du2_bar[:, np.newaxis]
        a1 = a1[:, np.newaxis]
        b1 = b1[:, np.newaxis]
        a2 = a2[:, np.newaxis]
        b2 = b2[:, np.newaxis]
        a3 = a3[:, np.newaxis]
        b3 = b3[:, np.newaxis]
        cos_t = cos_t[:, np.newaxis]
        sin_t = sin_t[:, np.newaxis]

        # u vs du
        bad_u_du = (u - u_bar) ** 2 / a1**2 + (du - du_bar) ** 2 / b1**2 > 1

        # u vs du2 (rotated ellipse)
        bad_u_du2 = (
            (cos_t * (u - u_bar) + sin_t * (du2 - du2_bar)) ** 2 / a2**2
            + (sin_t * (u - u_bar) - cos_t * (du2 - du2_bar)) ** 2 / b2**2
        ) > 1

        # du vs du2
        bad_du_du2 = (du - du_bar) ** 2 / a3**2 + (du2 - du2_bar) ** 2 / b3**2 > 1

        return bad_u_du | bad_u_du2 | bad_du_du2

    u_out = u.copy()
    bad_index = flag_bad_indices(u_out)
    total_bad = np.sum(bad_index, axis=1)
    iterations = 0

    while np.any(total_bad > remaining_spikes) and iterations < max_iter:
        u_out[bad_index] = np.nan
        interp_rows(u_out)
        bad_index = flag_bad_indices(u_out)
        total_bad = np.sum(bad_index, axis=1)
        iterations += 1

    interp_rows(u_out)
    return u_out

def recursive_gaussian():
    pass
    # TODO: Implement recursive Gaussian despike

def kernel_density():
    pass
    # TODO: Implement kernel density despike