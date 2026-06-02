import pytest
import numpy as np
import numpy.testing as npt

from utils.rotate_utils import (
    align_with_flow,
    align_with_principal_axis,
    apply_flow_rotation,
    coord_transform_3_beam_nortek,
    coord_transform_4_beam_nortek,
    coord_transform_4_beam_rdi,
    min_angle,
    rotate_velocity_by_theta,
)

from testhelpers.rotate_utils import nortek_3beam_T, nortek_4beam_T, rdi_4beam_T

####################
# Shared constants
####################
HEADING = 120.0
PITCH = 5.0
ROLL = -2.0
N = 200
T_3BEAM = nortek_3beam_T()
T_4BEAM_NORTEK = nortek_4beam_T()
T_4BEAM_RDI = rdi_4beam_T()


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
        ("beam", "xyz"),
        ("xyz", "beam"),
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


###################################################
# 3-beam Nortek - forward transform sanity checks
###################################################


def test_3beam_nortek_forward_xyz_to_enu():

    u1 = np.ones((N,))
    u2 = np.zeros((N,))
    u3 = np.zeros((N,))

    # Heading = 0 -> North, u1 and u2 will swap
    heading = 0.0
    (
        r1,
        r2,
        r3,
    ) = coord_transform_3_beam_nortek(
        u1,
        u2,
        u3,
        heading,
        pitch=0.0,
        roll=0.0,
        transformation_matrix=T_3BEAM,
        orientation="up",
        coords_in="xyz",
        coords_out="enu",
    )

    npt.assert_allclose(r1, u2, atol=1e-10)
    npt.assert_allclose(r2, u1, atol=1e-10)
    npt.assert_allclose(r3, u3, atol=1e-10)

    # Heading = 90 -> East, u1 = r1, u2 = r2, u3 = r3
    heading = 90.0
    (
        r1,
        r2,
        r3,
    ) = coord_transform_3_beam_nortek(
        u1,
        u2,
        u3,
        heading,
        pitch=0.0,
        roll=0.0,
        transformation_matrix=T_3BEAM,
        orientation="up",
        coords_in="xyz",
        coords_out="enu",
    )

    npt.assert_allclose(r1, u1, atol=1e-10)
    npt.assert_allclose(r2, u2, atol=1e-10)
    npt.assert_allclose(r3, u3, atol=1e-10)


def test_3beam_nortek_forward_beam_to_xyz():
    u1 = np.ones((N,))
    u2 = np.zeros((N,))
    u3 = np.zeros((N,))

    # Heading = 90.0, u1 = T_3BEAM[0, 0], u1 and u3 = -T_3BEAM[0, 0] = T_3BEAM[1, 0] = T_3BEAM[2, 0]
    heading = 90.0
    (
        r1,
        r2,
        r3,
    ) = coord_transform_3_beam_nortek(
        u1,
        u2,
        u3,
        heading,
        pitch=0.0,
        roll=0.0,
        transformation_matrix=T_3BEAM,
        orientation="up",
        coords_in="beam",
        coords_out="xyz",
    )

    npt.assert_allclose(r1, u1 * T_3BEAM[0, 0], atol=1e-10)
    npt.assert_allclose(r2, u1 * T_3BEAM[1, 0], atol=1e-10)
    npt.assert_allclose(r3, u1 * T_3BEAM[2, 0], atol=1e-10)


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
        ("beam", "xyz"),
        ("xyz", "beam"),
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


###################################################
# 4-beam Nortek - forward transform sanity checks
###################################################


def test_4beam_nortek_forward_beam_to_xyz():
    u1 = np.ones((N,))
    u2 = np.zeros((N,))
    u3 = np.zeros((N,))
    u4 = np.zeros((N,))
    r1, r2, r3, r4 = coord_transform_4_beam_nortek(
        u1,
        u2,
        u3,
        u4,
        heading=90,
        pitch=0,
        roll=0,
        transformation_matrix=T_4BEAM_NORTEK,
        orientation="down",
        coords_in="beam",
        coords_out="xyz",
    )

    npt.assert_allclose(u1 * T_4BEAM_NORTEK[0, 0], r1, atol=1e-10)
    npt.assert_allclose(u1 * T_4BEAM_NORTEK[1, 0], r2, atol=1e-10)
    npt.assert_allclose(u1 * T_4BEAM_NORTEK[2, 0], r3, atol=1e-10)
    npt.assert_allclose(u1 * T_4BEAM_NORTEK[3, 0], r4, atol=1e-10)


def test_4beam_nortek_forward_xyz_to_enu():
    # Heading = 90, orientation = up (no rotation)
    u1 = np.ones((N,))
    u2 = np.zeros((N,))
    u3 = np.ones((N,))
    u4 = np.ones((N,))
    r1, r2, r3, r4 = coord_transform_4_beam_nortek(
        u1,
        u2,
        u3,
        u4,
        heading=90,
        pitch=0,
        roll=0,
        transformation_matrix=T_4BEAM_NORTEK,
        orientation="up",
        coords_in="xyz",
        coords_out="enu",
    )

    npt.assert_allclose(u1, r1, atol=1e-10)
    npt.assert_allclose(u2, r2, atol=1e-10)
    npt.assert_allclose(u3, r3, atol=1e-10)
    npt.assert_allclose(u4, r4, atol=1e-10)

    # Heading = 0, orientation = down (rotate 90, flip vertical)
    u1 = np.ones((N,))
    u2 = np.zeros((N,))
    u3 = np.ones((N,))
    u4 = np.ones((N,))
    r1, r2, r3, r4 = coord_transform_4_beam_nortek(
        u1,
        u2,
        u3,
        u4,
        heading=0,
        pitch=0,
        roll=0,
        transformation_matrix=T_4BEAM_NORTEK,
        orientation="down",
        coords_in="xyz",
        coords_out="enu",
    )

    npt.assert_allclose(u1, r2, atol=1e-10)
    npt.assert_allclose(u2, r1, atol=1e-10)
    npt.assert_allclose(u3, -r3, atol=1e-10)
    npt.assert_allclose(u4, -r4, atol=1e-10)


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


#######################################
# RDI - forward transform sanity checks
#######################################


def test_rdi_forward_beam_to_xyz():
    u1 = np.ones((N,))
    u2 = np.zeros((N,))
    u3 = np.zeros((N,))
    u4 = np.zeros((N,))
    r1, r2, r3, r4 = coord_transform_4_beam_rdi(
        u1,
        u2,
        u3,
        u4,
        heading=0,
        pitch=0,
        roll=0,
        orientation="down",
        coords_in="beam",
        coords_out="xyz",
    )

    npt.assert_allclose(u1 * T_4BEAM_RDI[0, 0], r1, atol=1e-10)
    npt.assert_allclose(u1 * T_4BEAM_RDI[1, 0], r2, atol=1e-10)
    npt.assert_allclose(u1 * T_4BEAM_RDI[2, 0], r3, atol=1e-10)
    npt.assert_allclose(u1 * T_4BEAM_RDI[3, 0], r4, atol=1e-10)


def test_rdi_forward_xyz_to_enu():
    # For RDI, zero heading points along the Y axis, and in the upward orientation the z velocity points away from
    # up (see Fig 3 of ADCP Coordinate Transformation guide).

    # Heading = 0, orientation = down (no rotation)
    u1 = np.zeros((N,))
    u2 = np.ones((N,))
    u3 = np.ones((N,))
    u4 = u1 + u2 - u3
    r1, r2, r3, r4 = coord_transform_4_beam_rdi(
        u1,
        u2,
        u3,
        u4,
        heading=0,
        pitch=0,
        roll=0,
        transformation_matrix=T_4BEAM_RDI,
        orientation="down",
        coords_in="xyz",
        coords_out="enu",
    )

    npt.assert_allclose(u1, r1, atol=1e-10)
    npt.assert_allclose(u2, r2, atol=1e-10)
    npt.assert_allclose(u3, r3, atol=1e-10)
    npt.assert_allclose(u4, r4, atol=1e-10)

    # Heading = 90, orientation = up (rotate 90, flip vertical)
    u1 = np.zeros((N,))
    u2 = np.ones((N,))
    u3 = np.ones((N,))
    u4 = np.ones((N,))
    r1, r2, r3, r4 = coord_transform_4_beam_rdi(
        u1,
        u2,
        u3,
        u4,
        heading=90,
        pitch=0,
        roll=0,
        transformation_matrix=T_4BEAM_RDI,
        orientation="up",
        coords_in="xyz",
        coords_out="enu",
    )
    npt.assert_allclose(u1, r2, atol=1e-10)
    npt.assert_allclose(u2, r1, atol=1e-10)
    npt.assert_allclose(u3, -r3, atol=1e-10)
    npt.assert_allclose(u4, r4, atol=1e-10)


###############################
# min_angle (angle wrapping)
###############################


@pytest.mark.parametrize(
    "alpha,expected",
    [
        (0.0, 0.0),
        (45.0, 45.0),
        (179.999, 179.999),
        (180.0, -180.0),
        (-180.0, -180.0),
        (270.0, -90.0),
        (-270.0, 90.0),
        (540.0, -180.0),
        (-540.0, -180.0),
        (360.0, 0.0),
    ],
)
def test_min_angle_scalar_wrapping(alpha, expected):
    npt.assert_allclose(min_angle(alpha), expected, atol=1e-12)


def test_min_angle_array_wrapping():
    alphas = np.array([0.0, 90.0, 180.0, 270.0, -90.0, -180.0, -270.0, 720.0])
    expected = np.array([0.0, 90.0, -180.0, -90.0, -90.0, -180.0, 90.0, 0.0])
    npt.assert_allclose(min_angle(alphas), expected, atol=1e-12)


def test_min_angle_idempotent():
    rng = np.random.default_rng(7)
    alphas = rng.uniform(-1000, 1000, size=50)
    once = min_angle(alphas)
    twice = min_angle(once)
    npt.assert_allclose(once, twice, atol=1e-12)


########################################
# rotate_velocity_by_theta
########################################


def test_rotate_velocity_preserves_magnitude():
    rng = np.random.default_rng(0)
    M, N_samp = 3, 256
    u1 = rng.standard_normal((M, N_samp))
    u2 = rng.standard_normal((M, N_samp))
    u3 = rng.standard_normal((M, N_samp))
    theta_h = np.array([10.0, -45.0, 137.5])
    theta_v = np.array([5.0, -12.0, 30.0])

    u_rot, v_rot, w_rot = rotate_velocity_by_theta(u1, u2, u3, theta_h, theta_v)

    mag_in = np.sqrt(u1**2 + u2**2 + u3**2)
    mag_out = np.sqrt(u_rot**2 + v_rot**2 + w_rot**2)
    npt.assert_allclose(mag_in, mag_out, atol=1e-12)


def test_rotate_velocity_zero_angles_is_identity():
    rng = np.random.default_rng(1)
    u1 = rng.standard_normal((2, 64))
    u2 = rng.standard_normal((2, 64))
    u3 = rng.standard_normal((2, 64))
    u_rot, v_rot, w_rot = rotate_velocity_by_theta(u1, u2, u3, 0.0, 0.0)
    npt.assert_allclose(u_rot, u1, atol=1e-12)
    npt.assert_allclose(v_rot, u2, atol=1e-12)
    npt.assert_allclose(w_rot, u3, atol=1e-12)


def test_rotate_velocity_90deg_heading():
    # Pure u1 flow rotated by 90 degrees in the horizontal plane should land on -v_rot.
    A = 1.7
    u1 = np.full((1, 32), A)
    u2 = np.zeros((1, 32))
    u3 = np.zeros((1, 32))
    u_rot, v_rot, w_rot = rotate_velocity_by_theta(u1, u2, u3, 90.0, 0.0)
    npt.assert_allclose(u_rot, np.zeros_like(u1), atol=1e-12)
    npt.assert_allclose(v_rot, -u1, atol=1e-12)
    npt.assert_allclose(w_rot, np.zeros_like(u1), atol=1e-12)


def test_rotate_velocity_scalar_and_array_angles_equal():
    rng = np.random.default_rng(2)
    M, N_samp = 4, 128
    u1 = rng.standard_normal((M, N_samp))
    u2 = rng.standard_normal((M, N_samp))
    u3 = rng.standard_normal((M, N_samp))

    scalar_out = rotate_velocity_by_theta(u1, u2, u3, 30.0, 15.0)
    array_out = rotate_velocity_by_theta(u1, u2, u3, np.full(M, 30.0), np.full(M, 15.0))
    for s, a in zip(scalar_out, array_out):
        npt.assert_allclose(s, a, atol=1e-12)
        assert a.shape == (M, N_samp)


########################################
# align_with_principal_axis
########################################


def test_align_principal_axis_recovers_known_heading():
    # Build a 1-burst flow whose dominant variance lies along 45 deg.
    rng = np.random.default_rng(3)
    N_samp = 20000
    a = rng.standard_normal(N_samp) * 3.0  # large variance along major axis
    b = rng.standard_normal(N_samp) * 0.1  # small variance orthogonal
    angle = 45.0
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    u1 = (a * c - b * s).reshape(1, -1)
    u2 = (a * s + b * c).reshape(1, -1)
    u3 = np.zeros((1, N_samp))

    theta_h, theta_v = align_with_principal_axis(u1, u2, u3)
    assert theta_h.shape == (1,)
    assert theta_v.shape == (1,)
    # Principal axis is defined modulo 180 deg; collapse to the major-axis direction.
    recovered = (theta_h[0] + 180) % 180
    npt.assert_allclose(recovered, angle, atol=0.5)


def test_align_principal_axis_vertical_angle():
    # u3 = c * u_rot (with u_rot the streamwise component after heading rotation)
    # so theta_v should recover arctan(c).
    rng = np.random.default_rng(4)
    N_samp = 20000
    u_stream = rng.standard_normal(N_samp) * 2.0 + 1.5  # nonzero mean for u_rot_bar
    c_ratio = 0.25
    u1 = u_stream.reshape(1, -1)
    u2 = np.zeros((1, N_samp))
    u3 = (c_ratio * u_stream).reshape(1, -1)

    theta_h, theta_v = align_with_principal_axis(u1, u2, u3)
    expected_v = np.rad2deg(np.arctan(c_ratio))
    npt.assert_allclose(theta_v[0], expected_v, atol=0.5)


def test_align_principal_axis_shape_multi_burst():
    rng = np.random.default_rng(5)
    M, N_samp = 4, 1000
    u1 = rng.standard_normal((M, N_samp))
    u2 = rng.standard_normal((M, N_samp))
    u3 = rng.standard_normal((M, N_samp))
    theta_h, theta_v = align_with_principal_axis(u1, u2, u3)
    assert theta_h.shape == (M,)
    assert theta_v.shape == (M,)


########################################
# align_with_flow
########################################


def test_align_with_flow_recovers_mean_direction():
    # Constant mean flow with known components.
    M, N_samp = 1, 512
    u_bar, v_bar, w_bar = 1.0, 1.0, 0.2
    u1 = np.full((M, N_samp), u_bar)
    u2 = np.full((M, N_samp), v_bar)
    u3 = np.full((M, N_samp), w_bar)

    theta_h, theta_v = align_with_flow(u1, u2, u3)
    expected_h = np.rad2deg(np.arctan2(v_bar, u_bar))
    expected_v = np.rad2deg(np.arctan2(w_bar, np.sqrt(u_bar**2 + v_bar**2)))
    npt.assert_allclose(theta_h[0], expected_h, atol=1e-12)
    npt.assert_allclose(theta_v[0], expected_v, atol=1e-12)


def test_align_with_flow_zeroes_burst_mean_v_and_w():
    rng = np.random.default_rng(6)
    M, N_samp = 2, 4096
    u1 = rng.standard_normal((M, N_samp)) + 1.5
    u2 = rng.standard_normal((M, N_samp)) + 0.7
    u3 = rng.standard_normal((M, N_samp)) + 0.3

    theta_h, theta_v = align_with_flow(u1, u2, u3)
    _, v_rot, w_rot = rotate_velocity_by_theta(u1, u2, u3, theta_h, theta_v)
    npt.assert_allclose(np.mean(v_rot, axis=1), np.zeros(M), atol=1e-12)
    npt.assert_allclose(np.mean(w_rot, axis=1), np.zeros(M), atol=1e-12)


def test_align_with_flow_shape_multi_burst():
    rng = np.random.default_rng(8)
    M, N_samp = 3, 256
    u1 = rng.standard_normal((M, N_samp))
    u2 = rng.standard_normal((M, N_samp))
    u3 = rng.standard_normal((M, N_samp))
    theta_h, theta_v = align_with_flow(u1, u2, u3)
    assert theta_h.shape == (M,)
    assert theta_v.shape == (M,)


########################################
# apply_flow_rotation
########################################


def _make_burst(M=2, N_samp=512, seed=9):
    rng = np.random.default_rng(seed)
    return {
        "u1": rng.standard_normal((M, N_samp)) + 1.0,
        "u2": rng.standard_normal((M, N_samp)) + 0.3,
        "u3": rng.standard_normal((M, N_samp)) + 0.1,
    }


def test_apply_flow_rotation_align_streamwise():
    burst = _make_burst()
    M = burst["u1"].shape[0]
    out = apply_flow_rotation(burst, "align_streamwise")
    assert out["rotation"] == "align_streamwise"
    npt.assert_allclose(np.mean(out["u2"], axis=1), np.zeros(M), atol=1e-12)
    npt.assert_allclose(np.mean(out["u3"], axis=1), np.zeros(M), atol=1e-12)


def test_apply_flow_rotation_align_principal():
    burst = _make_burst()
    in_shape = burst["u1"].shape
    out = apply_flow_rotation(burst, "align_principal")
    assert out["rotation"] == "align_principal"
    for key in ("u1", "u2", "u3"):
        assert out[key].shape == in_shape


def test_apply_flow_rotation_tuple_matches_direct_rotation():
    burst_a = _make_burst()
    burst_b = _make_burst()  # same seed -> same data
    M = burst_a["u1"].shape[0]
    theta_h = np.full(M, 22.5)
    theta_v = np.full(M, 7.5)

    out = apply_flow_rotation(burst_a, (theta_h, theta_v))
    u_rot, v_rot, w_rot = rotate_velocity_by_theta(burst_b["u1"], burst_b["u2"], burst_b["u3"], theta_h, theta_v)

    npt.assert_allclose(out["u1"], u_rot, atol=1e-12)
    npt.assert_allclose(out["u2"], v_rot, atol=1e-12)
    npt.assert_allclose(out["u3"], w_rot, atol=1e-12)


def test_apply_flow_rotation_invalid_string_raises():
    burst = _make_burst()
    with pytest.raises(ValueError):
        apply_flow_rotation(burst, "not_a_real_mode")


def test_apply_flow_rotation_tuple_wrong_length_raises():
    burst = _make_burst(M=2)
    bad_theta_h = np.array([1.0])  # length 1, but burst has 2 rows
    bad_theta_v = np.array([2.0])
    with pytest.raises(ValueError):
        apply_flow_rotation(burst, (bad_theta_h, bad_theta_v))


def test_apply_flow_rotation_wrong_type_raises():
    burst = _make_burst()
    with pytest.raises(TypeError):
        apply_flow_rotation(burst, 42)
