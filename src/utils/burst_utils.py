from typing import Dict, Tuple

import numpy as np


def get_uvw(
    burst_data: Dict[str, np.ndarray],
    allowed_coords: Tuple[str, ...] = ("xyz", "enu"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (u1, u2, u3) from a burst after validating its coord state.

    Parameters
    ----------
    burst_data : dict
        Burst data dictionary. Must contain keys "u1", "u2", "u3", "coords".
    allowed_coords : tuple of str, optional
        Coordinate frames in which it is valid to read u/v/w semantics from
        u1/u2/u3. Default is ("xyz", "enu") which matches ADV/Sonic covariance.
        Pass ("enu",) to require ENU strictly.

    Returns
    -------
    (u1, u2, u3) : tuple of np.ndarray

    Raises
    ------
    ValueError
        If burst_data["coords"] is missing or not in allowed_coords.
    """
    coords = burst_data.get("coords")
    if coords not in allowed_coords:
        raise ValueError(
            f"burst_data['coords']={coords!r} not in allowed_coords={allowed_coords!r}. "
            f"Apply a coord transform before calling this method."
        )
    return burst_data["u1"], burst_data["u2"], burst_data["u3"]


def get_beams(burst_data: Dict[str, np.ndarray], n: int) -> Tuple[np.ndarray, ...]:
    """Return (u1, ..., un) from a burst after validating coords == 'beam'.

    Parameters
    ----------
    burst_data : dict
        Burst data dictionary. Must contain keys "u1" through f"u{n}" and "coords".
    n : int
        Number of beams to return (e.g. 4 or 5 for ADCP).

    Returns
    -------
    tuple of np.ndarray
        Length-n tuple of beam velocity arrays.

    Raises
    ------
    ValueError
        If burst_data["coords"] != "beam".
    KeyError
        If any beam key u1..u{n} is missing from burst_data.
    """
    coords = burst_data.get("coords")
    if coords != "beam":
        raise ValueError(f"burst_data['coords']={coords!r}; get_beams requires 'beam' coordinates.")
    return tuple(burst_data[f"u{i}"] for i in range(1, n + 1))
