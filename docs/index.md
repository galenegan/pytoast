# PyTOAST: Python Toolkit for Ocean, Atmospheric, and Surface wave Turbulence

For full source code, please visit the [GitHub repository](https://github.com/galenegan/pytoast). Documentation is 
accessed from the menu on the left.

## Project layout

```
`-- src
    |-- atmosphere
    |   |-- met.py
    |   `-- sonic.py
    |-- boundaries
    |   |-- bbl.py
    |   `-- coare.py
    |-- ocean
    |   |-- adcp.py
    |   |-- adv.py
    |   `-- ctd.py
    `-- utils
        |-- air_thermo.py
        |-- base_instrument.py
        |-- burst_utils.py
        |-- constants.py
        |-- despike_utils.py
        |-- interp_utils.py
        |-- io_utils.py
        |-- rotate_utils.py
        |-- sea_thermo.py
        |-- spectral_utils.py
        `-- wave_utils.py
|-- py-tests
|   |-- src
|   |   |-- atmosphere
|   |   |   |-- test_met_derive.py
|   |   |   `-- test_sonic.py
|   |   |-- boundaries
|   |   |   |-- test_coare.py
|   |   |   |-- test_styles.py
|   |   |   `-- testdata
|   |   |-- ocean
|   |   |   |-- adcp
|   |   |   |   |-- test_adcp_calcs.py
|   |   |   |   |-- test_adcp_loading.py
|   |   |   |   `-- testdata
|   |   |   |-- adv
|   |   |   |   |-- test_covariance.py
|   |   |   |   |-- test_dmd.py
|   |   |   |   |-- test_gerbi.py
|   |   |   |   |-- test_parsing.py
|   |   |   |   `-- testdata
|   |   |   |       |-- dmd
|   |   |   |       |-- mat_list
|   |   |   |       `-- npy_list
|   |   |   `-- ctd
|   |   |       |-- test_ctd_derive.py
|   |   |       `-- testdata
|   |   `-- utils
|   |       |-- test_air_thermo.py
|   |       |-- test_base_instrument_io.py
|   |       |-- test_burst_utils.py
|   |       |-- test_despike_utils.py
|   |       |-- test_io_utils.py
|   |       |-- test_rotate_utils.py
|   |       |-- test_sea_thermo.py
|   |       |-- test_wave_utils.py
|   |       `-- testdata
|   `-- testhelpers
|       |-- rotate_utils.py
|       |-- stub_utils.py
|       `-- synth_utils.py
|-- LICENSE
|-- README.md
|-- mkdocs.yml
|-- pyproject.toml
```