import pytest
import numpy as np
import numpy.testing as npt

from utils.rotate_utils import (
    coord_transform_3_beam_nortek,
    coord_transform_4_beam_nortek,
    coord_transform_4_beam_rdi,
)

####################
# Shared constants
####################

# 3-beam T matrix from Nortek reference script
T_3BEAM = np.array([[2896, 2896, 0], [-2896, 2896, 0], [-2896, -2896, 5792]], dtype=float) / 4096.0

# Synthetic 4-beam Nortek T: x/y from opposing beam pairs, z1/z2 from each pair's vertical component
_theta = np.deg2rad(25.0)
_a = 1 / (2 * np.sin(_theta))
_b = 1 / (4 * np.cos(_theta))
T_4BEAM_NORTEK = np.array(
    [
        [_a, -_a, 0.0, 0.0],
        [0.0, 0.0, -_a, _a],
        [_b, _b, 0.0, 0.0],
        [0.0, 0.0, _b, _b],
    ]
)

HEADING = 120.0
PITCH = 5.0
ROLL = -2.0
N = 200


############
# Fixtures
############


@pytest.fixture
def beams3():
    rng = np.random.default_rng(42)
    return tuple(rng.standard_normal(N) for _ in range(3))


@pytest.fixture
def beams4():
    rng = np.random.default_rng(42)
    return tuple(rng.standard_normal(N) for _ in range(4))


@pytest.fixture
def zero_error_beams():
    """RDI beam data with b1+b2 = b3+b4, so the error velocity is zero.
    Round-trips through ENU are exact only in this case."""
    rng = np.random.default_rng(42)
    b1 = rng.standard_normal(N)
    b2 = rng.standard_normal(N)
    b3 = rng.standard_normal(N)
    b4 = b1 + b2 - b3
    return b1, b2, b3, b4


############################################
# 3-beam Nortek - round-trip (up orientation)
############################################


@pytest.mark.parametrize(
    "coords_in,coords_out",
    [
        ("beam", "xyz"),
        ("xyz", "beam"),
        ("beam", "enu"),
        ("enu", "beam"),
        ("xyz", "enu"),
        ("enu", "xyz"),
    ],
)
def test_3beam_nortek_round_trip_up(beams3, coords_in, coords_out):
    u1, u2, u3 = beams3
    r1, r2, r3 = coord_transform_3_beam_nortek(
        u1,
        u2,
        u3,
        HEADING,
        PITCH,
        ROLL,
        T_3BEAM,
        orientation="up",
        coords_in=coords_in,
        coords_out=coords_out,
    )
    v1, v2, v3 = coord_transform_3_beam_nortek(
        r1,
        r2,
        r3,
        HEADING,
        PITCH,
        ROLL,
        T_3BEAM,
        orientation="up",
        coords_in=coords_out,
        coords_out=coords_in,
    )
    npt.assert_allclose(v1, u1, atol=1e-10)
    npt.assert_allclose(v2, u2, atol=1e-10)
    npt.assert_allclose(v3, u3, atol=1e-10)


##############################################
# 3-beam Nortek - round-trip (down orientation)
##############################################


@pytest.mark.parametrize(
    "coords_in,coords_out",
    [
        ("beam", "enu"),
        ("enu", "beam"),
        ("xyz", "enu"),
        ("enu", "xyz"),
    ],
)
def test_3beam_nortek_round_trip_down(beams3, coords_in, coords_out):
    u1, u2, u3 = beams3
    r1, r2, r3 = coord_transform_3_beam_nortek(
        u1,
        u2,
        u3,
        HEADING,
        PITCH,
        ROLL,
        T_3BEAM,
        orientation="down",
        coords_in=coords_in,
        coords_out=coords_out,
    )
    v1, v2, v3 = coord_transform_3_beam_nortek(
        r1,
        r2,
        r3,
        HEADING,
        PITCH,
        ROLL,
        T_3BEAM,
        orientation="down",
        coords_in=coords_out,
        coords_out=coords_in,
    )
    npt.assert_allclose(v1, u1, atol=1e-10)
    npt.assert_allclose(v2, u2, atol=1e-10)
    npt.assert_allclose(v3, u3, atol=1e-10)


##########################
# 3-beam Nortek - identity
##########################


@pytest.mark.parametrize("coords", ["beam", "xyz", "enu"])
def test_3beam_nortek_identity(beams3, coords):
    u1, u2, u3 = beams3
    r1, r2, r3 = coord_transform_3_beam_nortek(
        u1,
        u2,
        u3,
        HEADING,
        PITCH,
        ROLL,
        T_3BEAM,
        coords_in=coords,
        coords_out=coords,
    )
    npt.assert_array_equal(r1, u1)
    npt.assert_array_equal(r2, u2)
    npt.assert_array_equal(r3, u3)


###################################################################
# 3-beam Nortek - forward transform sanity checks (fill in from reference data)
###################################################################


def test_3beam_nortek_forward_beam_to_xyz():
    pytest.skip("Fill in expected values from a reference dataset")


def test_3beam_nortek_forward_beam_to_enu():
    pytest.skip("Fill in expected values from a reference dataset")


############################################
# 4-beam Nortek - round-trip (up orientation)
############################################


@pytest.mark.parametrize(
    "coords_in,coords_out",
    [
        ("beam", "xyz"),
        ("xyz", "beam"),
        ("beam", "enu"),
        ("enu", "beam"),
        ("xyz", "enu"),
        ("enu", "xyz"),
    ],
)
def test_4beam_nortek_round_trip_up(beams4, coords_in, coords_out):
    u1, u2, u3, u4 = beams4
    r1, r2, r3, r4 = coord_transform_4_beam_nortek(
        u1,
        u2,
        u3,
        u4,
        HEADING,
        PITCH,
        ROLL,
        T_4BEAM_NORTEK,
        orientation="up",
        coords_in=coords_in,
        coords_out=coords_out,
    )
    v1, v2, v3, v4 = coord_transform_4_beam_nortek(
        r1,
        r2,
        r3,
        r4,
        HEADING,
        PITCH,
        ROLL,
        T_4BEAM_NORTEK,
        orientation="up",
        coords_in=coords_out,
        coords_out=coords_in,
    )
    npt.assert_allclose(v1, u1, atol=1e-10)
    npt.assert_allclose(v2, u2, atol=1e-10)
    npt.assert_allclose(v3, u3, atol=1e-10)
    npt.assert_allclose(v4, u4, atol=1e-10)


##############################################
# 4-beam Nortek - round-trip (down orientation)
##############################################


@pytest.mark.parametrize(
    "coords_in,coords_out",
    [
        ("beam", "enu"),
        ("enu", "beam"),
        ("xyz", "enu"),
        ("enu", "xyz"),
    ],
)
def test_4beam_nortek_round_trip_down(beams4, coords_in, coords_out):
    u1, u2, u3, u4 = beams4
    r1, r2, r3, r4 = coord_transform_4_beam_nortek(
        u1,
        u2,
        u3,
        u4,
        HEADING,
        PITCH,
        ROLL,
        T_4BEAM_NORTEK,
        orientation="down",
        coords_in=coords_in,
        coords_out=coords_out,
    )
    v1, v2, v3, v4 = coord_transform_4_beam_nortek(
        r1,
        r2,
        r3,
        r4,
        HEADING,
        PITCH,
        ROLL,
        T_4BEAM_NORTEK,
        orientation="down",
        coords_in=coords_out,
        coords_out=coords_in,
    )
    npt.assert_allclose(v1, u1, atol=1e-10)
    npt.assert_allclose(v2, u2, atol=1e-10)
    npt.assert_allclose(v3, u3, atol=1e-10)
    npt.assert_allclose(v4, u4, atol=1e-10)


##########################
# 4-beam Nortek - identity
##########################


@pytest.mark.parametrize("coords", ["beam", "xyz", "enu"])
def test_4beam_nortek_identity(beams4, coords):
    u1, u2, u3, u4 = beams4
    r1, r2, r3, r4 = coord_transform_4_beam_nortek(
        u1,
        u2,
        u3,
        u4,
        HEADING,
        PITCH,
        ROLL,
        T_4BEAM_NORTEK,
        coords_in=coords,
        coords_out=coords,
    )
    npt.assert_array_equal(r1, u1)
    npt.assert_array_equal(r2, u2)
    npt.assert_array_equal(r3, u3)
    npt.assert_array_equal(r4, u4)


###################################################################
# 4-beam Nortek - forward transform sanity checks (fill in from reference data)
###################################################################


def test_4beam_nortek_forward_beam_to_xyz():
    pytest.skip("Fill in expected values from a Signature reference dataset")


def test_4beam_nortek_forward_beam_to_enu():
    pytest.skip("Fill in expected values from a Signature reference dataset")


####################
# RDI - round-trips
#
# Note on ENU round-trip limitations:
#   The RDI error velocity (4th component) is computed in xyz space and cannot
#   be recovered once data is in ENU. As a consequence:
#   - xyz -> enu -> xyz: first 3 components round-trip; 4th returns as zero.
#   - beam -> enu -> beam: only exact when the input has zero error velocity
#     (b1 + b2 == b3 + b4). Use the zero_error_beams fixture for those tests.
####################


def test_rdi_beam_xyz_round_trip(beams4):
    u1, u2, u3, u4 = beams4
    r1, r2, r3, r4 = coord_transform_4_beam_rdi(
        u1,
        u2,
        u3,
        u4,
        HEADING,
        PITCH,
        ROLL,
        coords_in="beam",
        coords_out="xyz",
    )
    v1, v2, v3, v4 = coord_transform_4_beam_rdi(
        r1,
        r2,
        r3,
        r4,
        HEADING,
        PITCH,
        ROLL,
        coords_in="xyz",
        coords_out="beam",
    )
    npt.assert_allclose(v1, u1, atol=1e-10)
    npt.assert_allclose(v2, u2, atol=1e-10)
    npt.assert_allclose(v3, u3, atol=1e-10)
    npt.assert_allclose(v4, u4, atol=1e-10)


def test_rdi_xyz_enu_round_trip_first_three_components(beams4):
    u1, u2, u3, u4 = beams4
    r1, r2, r3, r4 = coord_transform_4_beam_rdi(
        u1,
        u2,
        u3,
        u4,
        HEADING,
        PITCH,
        ROLL,
        coords_in="xyz",
        coords_out="enu",
    )
    v1, v2, v3, v4 = coord_transform_4_beam_rdi(
        r1,
        r2,
        r3,
        r4,
        HEADING,
        PITCH,
        ROLL,
        coords_in="enu",
        coords_out="xyz",
    )
    npt.assert_allclose(v1, u1, atol=1e-10)
    npt.assert_allclose(v2, u2, atol=1e-10)
    npt.assert_allclose(v3, u3, atol=1e-10)
    npt.assert_array_equal(v4, np.zeros(N))  # error velocity unrecoverable from ENU


def test_rdi_beam_enu_round_trip_zero_error(zero_error_beams):
    u1, u2, u3, u4 = zero_error_beams
    r1, r2, r3, r4 = coord_transform_4_beam_rdi(
        u1,
        u2,
        u3,
        u4,
        HEADING,
        PITCH,
        ROLL,
        coords_in="beam",
        coords_out="enu",
    )
    v1, v2, v3, v4 = coord_transform_4_beam_rdi(
        r1,
        r2,
        r3,
        r4,
        HEADING,
        PITCH,
        ROLL,
        coords_in="enu",
        coords_out="beam",
    )
    npt.assert_allclose(v1, u1, atol=1e-10)
    npt.assert_allclose(v2, u2, atol=1e-10)
    npt.assert_allclose(v3, u3, atol=1e-10)
    npt.assert_allclose(v4, u4, atol=1e-10)


def test_rdi_enu_beam_enu_round_trip_zero_error(zero_error_beams):
    b1, b2, b3, b4 = zero_error_beams
    e1, e2, e3, e4 = coord_transform_4_beam_rdi(
        b1,
        b2,
        b3,
        b4,
        HEADING,
        PITCH,
        ROLL,
        coords_in="beam",
        coords_out="enu",
    )
    r1, r2, r3, r4 = coord_transform_4_beam_rdi(
        e1,
        e2,
        e3,
        e4,
        HEADING,
        PITCH,
        ROLL,
        coords_in="enu",
        coords_out="beam",
    )
    v1, v2, v3, v4 = coord_transform_4_beam_rdi(
        r1,
        r2,
        r3,
        r4,
        HEADING,
        PITCH,
        ROLL,
        coords_in="beam",
        coords_out="enu",
    )
    npt.assert_allclose(v1, e1, atol=1e-10)
    npt.assert_allclose(v2, e2, atol=1e-10)
    npt.assert_allclose(v3, e3, atol=1e-10)


################################
# RDI - error velocity passthrough
################################


def test_rdi_error_velocity_unchanged_xyz_to_enu(beams4):
    u1, u2, u3, u4 = beams4
    r1, r2, r3, r4 = coord_transform_4_beam_rdi(
        u1,
        u2,
        u3,
        u4,
        HEADING,
        PITCH,
        ROLL,
        coords_in="xyz",
        coords_out="enu",
    )
    npt.assert_array_equal(r4, u4)


def test_rdi_error_velocity_from_beam_passed_through_to_enu(beams4):
    u1, u2, u3, u4 = beams4
    theta = np.deg2rad(25.0)
    d = 1 / (2 * np.sin(theta)) / np.sqrt(2)
    expected_error = d * (u1 + u2 - u3 - u4)
    r1, r2, r3, r4 = coord_transform_4_beam_rdi(
        u1,
        u2,
        u3,
        u4,
        HEADING,
        PITCH,
        ROLL,
        coords_in="beam",
        coords_out="enu",
    )
    npt.assert_allclose(r4, expected_error, atol=1e-12)


################
# RDI - identity
################


@pytest.mark.parametrize("coords", ["beam", "xyz", "enu"])
def test_rdi_identity(beams4, coords):
    u1, u2, u3, u4 = beams4
    r1, r2, r3, r4 = coord_transform_4_beam_rdi(
        u1,
        u2,
        u3,
        u4,
        HEADING,
        PITCH,
        ROLL,
        coords_in=coords,
        coords_out=coords,
    )
    npt.assert_array_equal(r1, u1)
    npt.assert_array_equal(r2, u2)
    npt.assert_array_equal(r3, u3)
    npt.assert_array_equal(r4, u4)


######################################################################
# RDI - forward transform sanity checks (fill in from reference data)
######################################################################


def test_rdi_forward_beam_to_xyz():
    pytest.skip("Fill in expected values from an RDI reference dataset")


def test_rdi_forward_beam_to_enu():
    pytest.skip("Fill in expected values from an RDI reference dataset")
