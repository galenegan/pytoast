"""Tests for coare36 and coare36_warm_layer in src/boundaries/coare.py.

Reference comparison uses coare36_reRference.npz generated from the
patched reference implementation (coare36vn_zrf_et.py) on all 2165 rows
of test_36_data.txt.
"""

import numpy as np
import numpy.testing as npt
import pytest
from pathlib import Path

from boundaries.coare import coare36, coare36_warm_layer, sea_surface_albedo

########
# Paths
########

TEST_DATA_DIR = f"{Path(__file__).parent}/testdata"
TEST_DATA_TXT = f"{TEST_DATA_DIR}/test_36_data.txt"
OUTPUT_DATA_NPZ = f"{TEST_DATA_DIR}/coare36_reference.npz"

RTOL_TIGHT = 1e-3  # 0.1% for momentum-dominated quantities
RTOL = 1e-2  # 1% for heat fluxes / skin quantities (TEOS-10 vs reference simple formulas)

# Keys where RTOL_TIGHT applies; everything else uses RTOL
_TIGHT_KEYS = frozenset(
    {"ustar", "Cd", "Urf", "UN", "U10", "U10N", "Cdn10", "whitecap_fraction", "wave_break_dissipation"}
)

# Per-key absolute tolerance (on top of rtol) for quantities where TEOS-10 and
# the reference simple thermodynamics diverge at a small number of points
_ATOL = {
    "hs": 0.1,  # W/m^2
    "h_buoyancy": 0.1,  # W/m^2
    "h_buoyancy_sonic": 0.1,  # W/m^2
    "dq_skin": 6e-3,  # g/kg
    "tstar": 1e-4,  # K; near-zero sign changes give large relative error
}

# All output keys present in the reference .npz (matches _REF_COL in gen_coare36_reference.py)
_ALL_REF_KEYS = [
    "ustar",
    "tau",
    "hs",
    "hl",
    "h_buoyancy",
    "h_buoyancy_sonic",
    "hl_webb",
    "tstar",
    "qstar",
    "z0",
    "z0t",
    "z0q",
    "Cd",
    "Ch",
    "Ce",
    "L",
    "zeta",
    "dT_skin",
    "dq_skin",
    "dz_skin",
    "Urf",
    "Trf",
    "Qrf",
    "RHrf",
    "UrfN",
    "TrfN",
    "QrfN",
    "lw_net",
    "sw_net",
    "Le",
    "rho_air",
    "UN",
    "U10",
    "U10N",
    "Cdn10",
    "Chn10",
    "Cen10",
    "hrain",
    "Qs",
    "evap",
    "T10",
    "T10N",
    "Q10",
    "Q10N",
    "RH10",
    "P10",
    "rho_air10",
    "gust",
    "whitecap_fraction",
    "wave_break_dissipation",
]


############
# Fixtures
############


@pytest.fixture(scope="module")
def test_data():
    """Raw input data from test_36_data.txt (2165 rows)."""
    d = np.genfromtxt(TEST_DATA_TXT, skip_header=1)
    return d


@pytest.fixture(scope="module")
def coare36_results(test_data):
    """Vectorized coare36 on all 2165 rows."""
    d = test_data
    return coare36(
        u=d[:, 1],
        z_u=d[:, 2],
        t=d[:, 3],
        z_t=d[:, 4],
        rh=d[:, 5],
        z_rh=d[:, 6],
        p=d[:, 7],
        ts=d[:, 8],
        sw_down=d[:, 9],
        lw_down=d[:, 10],
        julian_day=d[:, 0],
        lat=d[:, 11],
        lon=d[:, 12],
        pbl_height=d[:, 13],
        rain=d[:, 14],
        surface_salinity=d[:, 15],
        phase_speed=d[:, 16],
        h_sig=d[:, 17],
    )


@pytest.fixture(scope="module")
def ref():
    """Pre-computed reference outputs from the patched reference code."""
    return np.load(OUTPUT_DATA_NPZ)


@pytest.fixture(scope="module")
def warm_layer_results(test_data):
    """coare36_warm_layer on all 2165 rows (time-ordered)."""
    d = test_data
    return coare36_warm_layer(
        julian_day=d[:, 0],
        u=d[:, 1],
        z_u=d[:, 2],
        t=d[:, 3],
        z_t=d[:, 4],
        rh=d[:, 5],
        z_rh=d[:, 6],
        p=d[:, 7],
        ts=d[:, 8],
        sw_down=d[:, 9],
        lw_down=d[:, 10],
        lat=d[:, 11],
        lon=d[:, 12],
        pbl_height=d[:, 13],
        rain=d[:, 14],
        ts_depth=d[:, 19],
        surface_salinity=d[:, 15],
        phase_speed=d[:, 16],
        h_sig=d[:, 17],
    )


##############################################
# sea_surface_albedo - analytical checks
##############################################


class TestSeaSurfaceAlbedo:
    def test_night_albedo_zero(self):
        """Night (sun below horizon) -> all outputs zero."""
        # julian_day = 80.0: frac=0 -> utc=0, midnight at lon=0
        alb, trans, smax, alt = sea_surface_albedo(sw_down=300.0, julian_day=80.0, lat=0.0, lon=0.0)
        assert alb.item() == 0.0
        assert alt.item() == 0.0
        assert smax.item() == 0.0
        assert trans.item() == 0.0

    def test_solar_noon_equator_equinox(self):
        """Solar noon at equator during equinox -> altitude ~= 90 degrees,
        positive albedo."""
        # julian_day = 80.5: frac=0.5 -> utc=12, solar noon at lon=0
        # Vernal equinox around day 80: declination ~= 0 degrees
        _, _, _, alt = sea_surface_albedo(sw_down=800.0, julian_day=80.5, lat=0.0, lon=0.0)
        # sin(alt) ~= cos(lat-decl) ~= 1 -> alt ~= 90 degrees
        assert alt.item() > 80.0

    def test_sign_convention_east_west_symmetry(self):
        """Lon=+90 at utc=6 and lon=-90 at utc=18 give same-sign solar
        altitude.

        Both produce hour_angle ~= pi (solar noon) at lat=0, equinox.
        The two Julian days differ by 0.5 days so the declination
        changes by ~0.2 degrees, hence altitudes agree within 0.5
        degrees (atol).
        """
        _, _, _, alt_east = sea_surface_albedo(sw_down=800.0, julian_day=80.25, lat=0.0, lon=90.0)
        _, _, _, alt_west = sea_surface_albedo(sw_down=800.0, julian_day=80.75, lat=0.0, lon=-90.0)
        # Both should be near solar noon (altitude > 80 degrees)
        assert alt_east.item() > 80.0
        assert alt_west.item() > 80.0
        # Altitudes agree within 0.5 degrees -- small difference due to 0.5-day declination drift
        npt.assert_allclose(alt_east.item(), alt_west.item(), atol=0.5)


##########################
# coare36 vs reference
##########################


class TestCoare36VsReference:
    @pytest.mark.parametrize("key", _ALL_REF_KEYS)
    def test_output(self, key, coare36_results, ref):
        rtol = RTOL_TIGHT if key in _TIGHT_KEYS else RTOL
        npt.assert_allclose(coare36_results[key], ref[key], rtol=rtol, atol=_ATOL.get(key, 0.0))


##############################
# coare36 physical sanity
##############################


class TestCoare36Physical:
    def test_ustar_positive(self, coare36_results):
        assert np.all(coare36_results["ustar"] > 0), "Friction velocity must be positive"

    def test_dT_skin_non_negative(self, coare36_results):
        """Cool skin cools the surface -- dT_skin (depression) must be >= 0."""
        assert np.all(coare36_results["dT_skin"] >= 0), "dT_skin must be non-negative"

    def test_dz_skin_in_range(self, coare36_results):
        dz = coare36_results["dz_skin"]
        assert np.all(dz >= 0), "dz_skin must be non-negative"
        assert np.all(dz <= 0.01), "dz_skin must be <= 1 cm"


##########################################
# coare36_warm_layer physical sanity
##########################################


class TestWarmLayerPhysical:
    def test_dT_warm_non_negative(self, warm_layer_results):
        """Warm layer only adds heat -- temperature anomaly must be >= 0."""
        assert np.all(warm_layer_results["dT_warm"] >= 0)

    def test_dz_warm_in_range(self, warm_layer_results):
        dz = warm_layer_results["dz_warm"]
        assert np.all(dz > 0), "Warm layer depth must be positive"
        assert np.all(dz <= 19.0), "Warm layer depth must not exceed max_warm_depth=19 m"

    def test_dT_warm_to_skin_le_dT_warm(self, warm_layer_results):
        """Partial warm layer contribution cannot exceed total warm-layer
        anomaly."""
        npt.assert_array_less(
            warm_layer_results["dT_warm_to_skin"] - 1e-10,
            warm_layer_results["dT_warm"] + 1e-10,
            err_msg="dT_warm_to_skin must be <= dT_warm everywhere",
        )

    def test_warming_occurs(self, warm_layer_results, test_data):
        """Warm layer must produce positive heating at some point during
        daytime."""
        sw_down = test_data[:, 9]
        daytime = sw_down > 0
        assert np.any(warm_layer_results["dT_warm"][daytime] > 0), (
            "Expected warm layer heating during at least one daytime record"
        )
