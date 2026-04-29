"""One-shot helper to seed py-tests/src/ocean/adcp/testdata/ with truncated
copies of the Nortek AD2CP example bursts in /Users/ea-
gegan/Documents/MATLAB/adcp_example/.

Run manually once:
    .venv/bin/python py-tests/src/ocean/adcp/testdata/_truncate_source.py

Pytest does not collect this file (no `test_` prefix).
"""

import glob
import os

import numpy as np
import scipy.io as sio

N_KEEP = 4096
SOURCE_DIR = "/Users/ea-gegan/Documents/MATLAB/adcp_example"
DEST_DIR = os.path.dirname(os.path.abspath(__file__))

# Fields required by the ADCP name_map in src/ocean/main_adcp.py, plus Range.
KEEP_FIELDS = [
    "Burst_VelBeam1",
    "Burst_VelBeam2",
    "Burst_VelBeam3",
    "Burst_VelBeam4",
    "IBurst_VelBeam5",
    "Burst_Pressure",
    "Burst_TimeStamp",
    "Burst_Heading",
    "Burst_Pitch",
    "Burst_Roll",
    "Burst_Range",
]


def _truncate_field(name, arr):
    arr = np.asarray(arr)
    if name == "Burst_Range":
        return arr
    if arr.ndim == 1:
        return arr[:N_KEEP]
    return arr[:N_KEEP, ...]


def main():
    files = sorted(glob.glob(os.path.join(SOURCE_DIR, "BBASIT_0078.ad2cp.00000_*.mat")))
    if not files:
        raise SystemExit(f"No source .mat files in {SOURCE_DIR}")

    for ii, path in enumerate(files):
        raw = sio.loadmat(path, simplify_cells=True)
        data_key = next(k for k in raw if k.startswith("Data"))
        data = raw[data_key]

        truncated = {name: _truncate_field(name, data[name]) for name in KEEP_FIELDS if name in data}

        out_path = os.path.join(DEST_DIR, f"BBASIT_0078_burst{ii}.mat")
        sio.savemat(out_path, {"Data": truncated}, do_compression=True)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"wrote {out_path}  ({size_mb:.2f} MB)")

    # Synthetic single-burst single-height fixture for fast smoke tests.
    rng = np.random.default_rng(0)
    N = 1024
    fs = 16.0
    base = rng.standard_normal((N, 1)).astype(np.float32) * 0.05
    spike_idx = np.array([100, 250, 500, 800])
    with_spikes = base.copy()
    with_spikes[spike_idx, 0] = 5.0  # large spikes for despike test
    timestamp = (np.arange(N) / fs).reshape(-1, 1) + 1_700_000_000.0  # epoch seconds
    synth = {
        "Burst_VelBeam1": with_spikes,
        "Burst_VelBeam2": base + rng.standard_normal((N, 1)).astype(np.float32) * 0.05,
        "Burst_VelBeam3": base + rng.standard_normal((N, 1)).astype(np.float32) * 0.05,
        "Burst_VelBeam4": base + rng.standard_normal((N, 1)).astype(np.float32) * 0.05,
        "IBurst_VelBeam5": base + rng.standard_normal((N, 1)).astype(np.float32) * 0.05,
        "Burst_Pressure": np.full((N, 1), 10.0, dtype=np.float64),
        "Burst_TimeStamp": timestamp,
        "Burst_Heading": np.full((N, 1), 0.0, dtype=np.float32),
        "Burst_Pitch": np.full((N, 1), 0.0, dtype=np.float32),
        "Burst_Roll": np.full((N, 1), 0.0, dtype=np.float32),
        "Burst_Range": np.array([[1.0]], dtype=np.float32),
    }
    out_path = os.path.join(DEST_DIR, "synth_oneburst.mat")
    sio.savemat(out_path, {"Data": synth}, do_compression=True)
    print(f"wrote {out_path}  ({os.path.getsize(out_path) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
