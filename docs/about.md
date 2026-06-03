 The code is broadly organized by measurement location, with submodules for
associated sensors and parameterizations:

- Ocean
	- Acoustic Doppler Velocimeters (ADV)
	- Acoustic Doppler Current Profilers (ADCP)
	- CTD
- Atmosphere
	- Sonic anemometers
	- Bulk meteorological sensors
- Boundaries
	- Air-sea fluxes (COARE 3.6)
	- Bottom boundary layer models (Madsen, Styles)

After initializing the appropriate instrument class, some of the things you can calculate are:

- TKE dissipation (spectral and structure function methods)
- Reynolds stresses (including multiple wave-turbulence decompositions)
- Directional wave statistics
- Air and seawater thermodynamic properties

Some other nice features of the code base:

- Standard methods for loading, despiking, and rotating raw observations
- Utility functions for common spectral analysis and wave data processing methods
- Functionality for exporting derived statistics to NetCDF
- Documentation (see menu to the left)
- Example Jupyter notebooks demonstrating core functionality on test data (see `notebooks/`)
- An extensive test suite verifying correctness wherever possible (see `py-tests/`)

## References and Acknowledgements

References for all algorithms are included in source code docstrings and
the documentation on this site. Test data includes observations collected
as part of the following projects:

- NSF OCE-1736668 ([link](https://purl.stanford.edu/wv787xr0534))
- NSF OCE-202302 ([link 1](https://doi.org/10.26025/1912/66837), [link 2](https://doi.org/10.26025/1912/29583))