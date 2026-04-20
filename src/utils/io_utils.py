import numpy as np
import xarray as xr
from typing import Any, Dict, List, Optional, Tuple, Union

Numeric = Union[float, int, np.ndarray]

COORD_KEYS = {"time", "z", "freq"}


def results_to_dataset(
    results: List[Dict[str, Any]],
    burst_times: np.ndarray,
    z: Optional[np.ndarray] = None,
    freq: Optional[np.ndarray] = None,
    attrs: Optional[dict] = None,
) -> xr.Dataset:
    """
    Concatenate a list of per-burst result dictionaries into an xarray Dataset.

    Dimensions are inferred from the shapes of the values. The first axis of
    each output variable is always ``burst_time`` (length ``len(results)``).
    Inner axes are matched against ``z`` and ``freq`` when provided:

    =================================  ============================
    First non-None value shape         Output dims
    =================================  ============================
    scalar                             (burst_time,)
    (len(z),)                          (burst_time, z)
    (len(z) - 1,)                      (burst_time, z_mid)
    (len(freq),)                       (burst_time, freq)
    (len(z), len(freq))                (burst_time, z, freq)
    (len(z) - 1, len(freq))            (burst_time, z_mid, freq)
    =================================  ============================

    Keys ``"time"``, ``"z"``, and ``"freq"`` inside the per-burst dicts are
    treated as coordinates and are not written as data variables.

    Parameters
    ----------
    results : list of dict
        Per-burst result dictionaries. Keys missing from a given burst are
        filled with NaN in the output.
    burst_times : np.ndarray
        1D array of length ``len(results)`` giving a representative timestamp
        for each burst. Used as the ``burst_time`` coordinate.
    z : np.ndarray, optional
        Height coordinate, shape (n_heights,). Used to match inner axes of
        shape (n_heights,) or (n_heights - 1,).
    freq : np.ndarray, optional
        Frequency coordinate, shape (n_freq,). Used to match inner axes of
        shape (n_freq,).
    attrs : dict, optional
        Global attributes to attach to the returned Dataset.

    Returns
    -------
    xr.Dataset
        Dataset with variables keyed by result-dict keys and ``burst_time`` as
        the leading dimension.
    """
    if not isinstance(results, list) or len(results) == 0:
        raise ValueError("`results` must be a non-empty list of dicts")

    burst_times = np.asarray(burst_times)
    if burst_times.ndim != 1 or len(burst_times) != len(results):
        raise ValueError(f"`burst_times` must be 1D with length {len(results)}; got shape {burst_times.shape}")

    n_bursts = len(results)
    n_heights = int(len(z)) if z is not None else None
    n_freq = int(len(freq)) if freq is not None else None

    # Union of variable keys across all bursts, excluding coord-reserved names.
    var_keys: List[str] = []
    seen = set()
    for r in results:
        for k in r.keys():
            if k in COORD_KEYS or k in seen:
                continue
            seen.add(k)
            var_keys.append(k)

    coords: Dict[str, Any] = {"burst_time": ("burst_time", burst_times)}
    if z is not None:
        z_arr = np.asarray(z)
        coords["z"] = ("z", z_arr)
        if n_heights is not None and n_heights > 1:
            coords["z_mid"] = ("z_mid", 0.5 * (z_arr[:-1] + z_arr[1:]))
    if freq is not None:
        coords["freq"] = ("freq", np.asarray(freq))

    data_vars: Dict[str, Tuple[Tuple[str, ...], np.ndarray]] = {}
    for key in var_keys:
        canonical = _first_non_none_value(results, key)
        if canonical is None:
            continue  # all-None column; skip
        dims_inner, shape_inner = _infer_dims(canonical, key, n_heights=n_heights, n_freq=n_freq)
        full_shape = (n_bursts, *shape_inner)
        arr = np.full(full_shape, np.nan, dtype=float)
        for i, r in enumerate(results):
            v = r.get(key, None)
            if v is None:
                continue
            v_arr = np.asarray(v, dtype=float)
            if v_arr.shape != shape_inner:
                raise ValueError(
                    f"Shape mismatch for key {key!r} at burst {i}: expected {shape_inner}, got {v_arr.shape}"
                )
            arr[i] = v_arr
        dims = ("burst_time", *dims_inner)
        data_vars[key] = (dims, arr)

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    if attrs:
        ds.attrs.update(attrs)
    return ds


def _first_non_none_value(results: List[Dict[str, Any]], key: str) -> Optional[Any]:
    for r in results:
        v = r.get(key, None)
        if v is not None:
            return v
    return None


def _infer_dims(
    value: Any,
    key: str,
    n_heights: Optional[int],
    n_freq: Optional[int],
) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    """Return (inner_dim_names, inner_shape) for a per-burst value."""
    arr = np.asarray(value)
    shape = arr.shape

    if arr.ndim == 0:
        return (), ()

    if arr.ndim == 1:
        n = shape[0]
        candidates: List[str] = []
        if n_heights is not None and n == n_heights:
            candidates.append("z")
        if n_heights is not None and n_heights > 1 and n == n_heights - 1:
            candidates.append("z_mid")
        if n_freq is not None and n == n_freq:
            candidates.append("freq")
        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous shape for key {key!r}: length {n} matches multiple "
                f"coordinates ({candidates}). Ensure len(z) != len(freq) or "
                f"rename the key."
            )
        if len(candidates) == 1:
            return (candidates[0],), shape
        # Unknown 1D size — create a key-specific dim so users can still persist
        # custom quantities (e.g. mode amplitudes).
        return (f"{key}_dim",), shape

    if arr.ndim == 2:
        n0, n1 = shape
        if n_heights is not None and n_freq is not None:
            if n0 == n_heights and n1 == n_freq:
                return ("z", "freq"), shape
            if n_heights > 1 and n0 == n_heights - 1 and n1 == n_freq:
                return ("z_mid", "freq"), shape
        # Fallback: key-specific dims.
        return (f"{key}_dim0", f"{key}_dim1"), shape

    # ndim >= 3: generic dim names.
    return tuple(f"{key}_dim{i}" for i in range(arr.ndim)), shape
