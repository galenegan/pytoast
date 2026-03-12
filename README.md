# pyTOAST: Python Toolkit for Ocean, Atmospheric, and Surface-wave Turbulence

> A pure python toolkit for analyzing observations of ocean and atmospheric turbulence and related bulk variables.  

---

## Overview

---

## Installation

```bash
# Clone and install in editable mode (recommended for development)
git clone git@github.com:galenegan/pytoast.git
cd pytoast
pip install -e ".[dev]"
```

If you only need to *use* the package (not develop it):

```bash
git clone git@github.com:galenegan/pytoast.git
```

### Requirements

- Python $\ge$ 3.13
- Core dependencies are installed automatically (see `pyproject.toml`).

---

## Quick start

```python
# Example code here that demonstrates using some core functionality
```

---

## Running tests

```bash
pytest                    # run all tests
pytest --cov=myproject    # with coverage report
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

This project is licensed under the MIT License — see [LICENSE](LICENSE) for
details.
