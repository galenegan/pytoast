# pyTOAST: Python Toolkit for Ocean, Atmospheric, and Surface-wave Turbulence

> A pure-Python toolkit for analyzing observations of ocean and atmospheric
> turbulence and related bulk variables.

---

## Overview

`pytoast` provides instrument classes for common turbulence sensors --
acoustic Doppler velocimeters (ADV), acoustic Doppler current profilers
(ADCP), sonic anemometers, CTDs, and meteorological stations -- along with a
shared preprocessing pipeline (despiking, coordinate rotations) and a suite
of derived calculations: turbulent kinetic energy and dissipation, Reynolds
stresses with wave-turbulence decomposition, directional wave statistics
(Jones-Monismith-corrected), seawater and air thermodynamics (TEOS-10), and
boundary-layer flux parameterizations (COARE 3.6, Madsen and Styles benthic
boundary layer). The library is aimed at physical oceanographers and
boundary-layer meteorologists processing field observations.

---

## Installation

```bash
pip install pytoast
```

For development:

```bash
git clone git@github.com:galenegan/pytoast.git
cd pytoast
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.13
- Core dependencies (numpy, scipy, pandas, xarray, matplotlib, h5py, netCDF4,
  mat73) are installed automatically.

---

## Quick start

```python
from pytoast.ocean.adv import ADV

name_map = {
    "u1": "u", "u2": "v", "u3": "w",
    "p": "pressure", "time": "time",
}

adv = ADV(files="burst.mat", name_map=name_map, fs=16, z=[1.0])

adv.set_preprocess_opts({
    "despike": {"method": "goring_nikora"},
    "rotate":  {"flow_rotation": "align_streamwise"},
})

burst = adv.load_burst(0)
diss  = adv.dissipation(burst, f_low=1.0, f_high=4.0)
print(diss["eps"])     # TKE dissipation rate (m^2/s^3) at each height
```

See the [documentation](https://galenegan.github.io/pytoast/) for the full
API reference.

---

## Running tests

```bash
pytest
pytest --cov=src    # with coverage report
```

---

## Contributing

1. Fork & clone the repo.
2. Create a feature branch: `git checkout -b feature/my-improvement`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Run tests before pushing: `pytest`
5. Open a pull request.

---

## Citation

If you use this software in published work, please cite:

```bibtex
@software{pytoast,
  author  = {Galen Egan},
  title   = {pyTOAST: Python Toolkit for Ocean, Atmospheric, and Surface-wave Turbulence},
  year    = {2026},
  url     = {https://github.com/galenegan/pytoast},
}
```

---

## License

This project is licensed under the MIT License -- see [LICENSE](LICENSE) for
details.
