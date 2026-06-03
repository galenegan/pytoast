PyTOAST is a pure python toolkit for analyzing observations of ocean and
atmospheric turbulence and related bulk variables. The code is broadly
organized by measurement location, with submodules for associated sensors and
parameterizations:

- Ocean
	- Acoustic Doppler Velocimeters (ADV)
	- Acoustic Doppler Current Profilers (ADCP)
	- CTD
- Atmosphere
	- Sonic anemometers
	- Bulk meteorological sensors
- Boundaries
	- Wave measurements
	- Air-sea fluxes (COARE 3.6)
	- Bottom boundary layer models (Madsen, Styles)

After initializing the appropriate instrument class, some of the things you can calculate are:

- TKE dissipation (spectral and structure function methods)
- Reynolds stresses (including multiple wave-turbulence decompositions)
- Directional wave statistics
- Air and seawater thermodynamic properties

Standard methods for loading, despiking, and rotating raw observations are also included. Some other nice features of the code base:

- Documentation (see menu to the left)
- Example Jupyter notebooks (see `notebooks/`) demonstrating core functionality on test data
- An extensive test suite (see `py-tests/`) verifying correctness wherever possible