import numpy as np
from typing import Tuple, Optional
from scipy.stats import circmean


def coord_transform_3_beam_nortek(
    u1,
    u2,
    u3,
    heading,
    pitch,
    roll,
    transformation_matrix,
    declination: float = 0.0,
    orientation: str = "up",
    coords_in: str = "beam",
    coords_out: str = "xyz",
):
    """
    Implementation of Nortek's coordinate transformation for 3-beam instruments.
    https://support.nortekgroup.com/hc/en-us/articles/360029820971-How-is-a-coordinate-transformation-done

    Parameters
    ----------
    transformation_matrix
    orientation
    heading
    pitch
    roll

    Returns
    -------

    """
    if coords_in == coords_out:
        return (u1, u2, u3)

    T = transformation_matrix.copy()
    if np.any(T > 1000):
        T /= 4096.0

    # Velocity array
    U = np.vstack((u1.reshape(1, -1), u2.reshape(1, -1), u3.reshape(1, -1)))

    if coords_in == "beam" and coords_out == "xyz":
        U_rot = T @ U
        u1_rot = U_rot[0, :]
        u2_rot = U_rot[1, :]
        u3_rot = U_rot[2, :]
        return (u1_rot, u2_rot, u3_rot)
    elif coords_in == "xyz" and coords_out == "beam":
        U_rot = np.linalg.inv(T) @ U
        u1_rot = U_rot[0, :]
        u2_rot = U_rot[1, :]
        u3_rot = U_rot[2, :]
        return (u1_rot, u2_rot, u3_rot)

    T_flip = T.copy()
    if orientation == "down":
        T_flip[1, :] = -T_flip[1, :]
        T_flip[2, :] = -T_flip[2, :]

    heading_plus_dec = (heading + declination) % 360
    h_rad = circmean(np.radians(heading_plus_dec - 90))
    p_rad = circmean(np.radians(pitch))
    r_rad = circmean(np.radians(roll))

    # Heading matrix
    H = np.array(
        [
            [np.cos(h_rad), np.sin(h_rad), 0],
            [-np.sin(h_rad), np.cos(h_rad), 0],
            [0, 0, 1],
        ]
    )

    # Pitch and Roll matrix
    P = np.array(
        [
            [
                np.cos(p_rad),
                -np.sin(p_rad) * np.sin(r_rad),
                -np.cos(r_rad) * np.sin(p_rad),
            ],
            [0, np.cos(r_rad), -np.sin(r_rad)],
            [
                np.sin(p_rad),
                np.sin(r_rad) * np.cos(p_rad),
                np.cos(p_rad) * np.cos(r_rad),
            ],
        ]
    )

    # XYZ to ENU matrix
    R = H @ P

    if coords_in == "beam" and coords_out == "enu":
        U_rot = R @ T_flip @ U
    elif coords_in == "enu" and coords_out == "beam":
        U_rot = np.linalg.inv(T_flip) @ np.linalg.inv(R) @ U
    elif coords_in == "enu" and coords_out == "xyz":
        U_rot = T @ np.linalg.inv(T_flip) @ np.linalg.inv(R) @ U
    elif coords_in == "xyz" and coords_out == "enu":
        U_rot = R @ T_flip @ np.linalg.inv(T) @ U
    else:
        raise ValueError("Invalid coordinate transformation.")

    # TODO: store T inverses so we only need to calculate once
    u1_rot = U_rot[0, :]
    u2_rot = U_rot[1, :]
    u3_rot = U_rot[2, :]
    return (u1_rot, u2_rot, u3_rot)


def coord_transform_4_beam_nortek(
    u1,
    u2,
    u3,
    u4,
    heading,
    pitch,
    roll,
    transformation_matrix,
    declination: float = 0.0,
    orientation: str = "up",
    coords_in: str = "beam",
    coords_out: str = "xyz",
):
    """
    Implementation of Nortek's coordinate transformation for the 4-beam Signature instrument.
    Supports all combinations of beam, xyz, and enu coordinates.

    https://support.nortekgroup.com/hc/en-us/articles/360029820971-How-is-a-coordinate-transformation-done

    Parameters
    ----------
    u1, u2, u3, u4 : array-like
        Velocity components in the input coordinate system
    heading, pitch, roll : float or array-like
        Instrument orientation in degrees
    transformation_matrix : np.ndarray
        4x4 beam-to-xyz transformation matrix from the instrument config
    declination : float
        Magnetic declination in degrees
    orientation : str
        "up" or "down"
    coords_in, coords_out : str
        One of "beam", "xyz", "enu"

    Returns
    -------
    tuple of four np.ndarray
    """
    if coords_in == coords_out:
        return (u1, u2, u3, u4)

    T = transformation_matrix.copy()
    if np.any(T > 1000):
        T /= 4096.0

    U = np.vstack((u1.reshape(1, -1), u2.reshape(1, -1), u3.reshape(1, -1), u4.reshape(1, -1)))

    # beam <-> xyz: no orientation correction needed
    if coords_in == "beam" and coords_out == "xyz":
        U_rot = T @ U
    elif coords_in == "xyz" and coords_out == "beam":
        U_rot = np.linalg.inv(T) @ U
    else:
        # Build the 4x4 xyz->enu rotation matrix per Nortek Signature reference
        heading_plus_dec = (heading + declination) % 360
        h_rad = circmean(np.radians(heading_plus_dec - 90))
        p_rad = circmean(np.radians(pitch))
        r_rad = circmean(np.radians(roll))

        H = np.array(
            [
                [np.cos(h_rad), np.sin(h_rad), 0],
                [-np.sin(h_rad), np.cos(h_rad), 0],
                [0, 0, 1],
            ]
        )
        P = np.array(
            [
                [np.cos(p_rad), -np.sin(p_rad) * np.sin(r_rad), -np.cos(r_rad) * np.sin(p_rad)],
                [0, np.cos(r_rad), -np.sin(r_rad)],
                [np.sin(p_rad), np.sin(r_rad) * np.cos(p_rad), np.cos(p_rad) * np.cos(r_rad)],
            ]
        )

        R = np.zeros((4, 4))
        R[:3, :3] = H @ P
        # Nortek Signature twoZs modifications (from Nortek reference script)
        R[0, 2] /= 2
        R[0, 3] = R[0, 2]
        R[1, 2] /= 2
        R[1, 3] = R[1, 2]
        R[3, :] = R[2, :]
        R[2, 3] = 0
        R[3, 3] = R[2, 2]
        R[3, 2] = 0

        R_inv = np.linalg.inv(R)

        # For "down" orientation, flip sign of y, z1, z2 (not x) in xyz space before
        # applying the ENU rotation. sign^2 = 1, so the same array undoes itself on inverse.
        sign = np.array([[1], [-1], [-1], [-1]]) if orientation == "down" else np.ones((4, 1))

        if coords_in == "xyz" and coords_out == "enu":
            U_rot = R @ (sign * U)
        elif coords_in == "beam" and coords_out == "enu":
            U_xyz = T @ U
            U_rot = R @ (sign * U_xyz)
        elif coords_in == "enu" and coords_out == "xyz":
            # sign^2 = 1, so applying sign again inverts the orientation correction
            U_rot = sign * (R_inv @ U)
        elif coords_in == "enu" and coords_out == "beam":
            U_xyz = sign * (R_inv @ U)
            U_rot = np.linalg.inv(T) @ U_xyz
        else:
            raise ValueError("Invalid coordinate transformation.")

    return (U_rot[0, :], U_rot[1, :], U_rot[2, :], U_rot[3, :])


def coord_transform_4_beam_rdi(
    u1,
    u2,
    u3,
    u4,
    heading,
    pitch,
    roll,
    beam_angle: float = 25.0,
    transformation_matrix: Optional[np.ndarray] = None,
    declination: float = 0.0,
    orientation: str = "up",
    coords_in: str = "beam",
    coords_out: str = "xyz",
):
    """
    Coordinate transformation for Teledyne RDI 4-beam ADCPs. Supports all
    combinations of beam, xyz, and enu coordinates.

    https://www.teledynemarine.com/en-us/support/SiteAssets/RDI/Manuals%20and%20Guides/General%20Interest/Coordinate_Transformation.pdf

    Parameters
    ----------
    u1, u2, u3, u4 : array-like
        Velocity components in the input coordinate system. For xyz inputs,
        u4 is the error velocity. For enu inputs, u4 is unused.
    heading, pitch, roll : float or array-like
        Instrument orientation in degrees
    beam_angle : float
        Beam angle from vertical in degrees (used when transformation_matrix is None)
    transformation_matrix : np.ndarray, optional
        4x4 beam-to-xyz transformation matrix. Computed from beam_angle if not provided.
    declination : float
        Magnetic declination in degrees
    orientation : str
        "up" or "down"
    coords_in, coords_out : str
        One of "beam", "xyz", "enu"

    Returns
    -------
    tuple of four np.ndarray
        For enu output: (E, N, U, error), where error is passed through from
        the xyz stage. For enu input (enu→xyz, enu→beam), the 4th return value
        is zero because the error velocity cannot be recovered from ENU.
    """
    if coords_in == coords_out:
        return (u1, u2, u3, u4)

    if transformation_matrix is None:
        beam_angle_rad = np.deg2rad(beam_angle)
        a = 1 / (2 * np.sin(beam_angle_rad))
        b = 1 / (4 * np.cos(beam_angle_rad))
        c = 1  # convex transducer head
        d = a / np.sqrt(2)
        T = np.array([[c * a, -c * a, 0, 0], [0, 0, -c * a, c * a], [b, b, b, b], [d, d, -d, -d]])
    else:
        T = transformation_matrix.copy()

    U = np.vstack((u1.reshape(1, -1), u2.reshape(1, -1), u3.reshape(1, -1), u4.reshape(1, -1)))

    # beam <-> xyz: T and its inverse (all 4 components, including error velocity)
    if coords_in == "beam" and coords_out == "xyz":
        U_rot = T @ U
        return (U_rot[0, :], U_rot[1, :], U_rot[2, :], U_rot[3, :])

    if coords_in == "xyz" and coords_out == "beam":
        U_rot = np.linalg.inv(T) @ U
        return (U_rot[0, :], U_rot[1, :], U_rot[2, :], U_rot[3, :])

    # All remaining cases involve ENU: build the 3x3 attitude matrix M
    heading_plus_dec = (heading + declination) % 360
    r_rad = circmean(np.deg2rad(roll))
    h_rad = circmean(np.deg2rad(heading_plus_dec))
    p_rad = circmean(np.deg2rad(pitch))

    # Modifications per RDI manual
    p_rad = np.arctan(np.tan(p_rad) * np.cos(r_rad))
    if orientation == "up":
        r_rad += np.pi

    ch, sh = np.cos(h_rad), np.sin(h_rad)
    cr, sr = np.cos(r_rad), np.sin(r_rad)
    cp, sp = np.cos(p_rad), np.sin(p_rad)

    M = np.array(
        [
            [(ch * cr + sh * sp * sr), (sh * cp), (ch * sr - sh * sp * cr)],
            [(-sh * cr + ch * sp * sr), (ch * cp), (-sh * sr + ch * sp * cr)],
            [(-cp * sr), (sp), (cp * cr)],
        ]
    )

    if coords_in == "xyz" and coords_out == "enu":
        # M acts on xyz (first 3 rows); error velocity (row 3) is passed through
        U_enu = M @ U[:3, :]
        return (U_enu[0, :], U_enu[1, :], U_enu[2, :], U[3, :])

    if coords_in == "beam" and coords_out == "enu":
        U_xyz = T @ U  # 4 components: x, y, z, error
        U_enu = M @ U_xyz[:3, :]
        return (U_enu[0, :], U_enu[1, :], U_enu[2, :], U_xyz[3, :])

    M_inv = np.linalg.inv(M)
    zeros = np.zeros((1, U.shape[1]))

    if coords_in == "enu" and coords_out == "xyz":
        # u4 (error velocity) cannot be recovered from ENU; returned as zero
        U_xyz = M_inv @ U[:3, :]
        return (U_xyz[0, :], U_xyz[1, :], U_xyz[2, :], zeros[0, :])

    if coords_in == "enu" and coords_out == "beam":
        # Reconstruct xyz with error=0 (unrecoverable), then invert T
        U_xyz4 = np.vstack([M_inv @ U[:3, :], zeros])
        U_beam = np.linalg.inv(T) @ U_xyz4
        return (U_beam[0, :], U_beam[1, :], U_beam[2, :], U_beam[3, :])

    raise ValueError(f"Invalid coordinate transformation: {coords_in!r} -> {coords_out!r}")


def align_with_principal_axis(u1, u2, u3) -> Tuple:
    """
    Calculates the direction of maximum variance from the u and v velocities (Thomson & Emery, 4.52b).

    Parameters
    ----------

    Returns
    -------
    theta : float
        direction of maximum variance in degrees, CCW positive from east
        assuming that u = eastward velocity, v = northward velocity
    """
    # (Co)variances
    u1_bar = np.mean(u1, axis=1, keepdims=True)
    u2_bar = np.mean(u2, axis=1, keepdims=True)
    u1_prime = u1 - u1_bar
    u2_prime = u2 - u2_bar
    u1_var = np.mean(u1_prime**2, axis=1)
    u2_var = np.mean(u2_prime**2, axis=1)
    cv = np.mean(u1_prime * u2_prime, axis=1)

    # Direction of maximum variance in xy-plane (heading)
    theta_h_radians = 0.5 * np.arctan2(2.0 * cv, (u1_var - u2_var))

    # Pitch angle
    u_rot = u1 * np.cos(theta_h_radians[:, np.newaxis]) + u2 * np.sin(theta_h_radians[:, np.newaxis])
    u_rot_bar = np.mean(u_rot, axis=1)
    u3_bar = np.mean(u3, axis=1)
    theta_v_radians = np.arctan2(u3_bar, u_rot_bar)

    out = (np.rad2deg(theta_h_radians), np.rad2deg(theta_v_radians))
    return out


def align_with_flow(u1, u2, u3) -> Tuple:
    """
    Rotates u, v, w velocities to minimize the burst-averaged v and w.

    Parameters
    ----------


    Returns
    -------
    u_rot: DataArray
        Major axis horizontal velocity

    v_rot: DataArray
        Minor axis horizontal velocity

    w_rot: DataArray
        Zero-mean vertical velocity
    """

    u_bar = np.mean(u1, axis=1)
    v_bar = np.mean(u2, axis=1)
    w_bar = np.mean(u3, axis=1)
    U = np.sqrt(u_bar**2 + v_bar**2)
    theta_h = np.arctan2(v_bar, u_bar)
    theta_v = np.arctan2(w_bar, U)
    out = (np.rad2deg(theta_h), np.rad2deg(theta_v))
    return out


def rotate_velocity_by_theta(u1, u2, u3, theta_h, theta_v):
    """
    Rotates u, v, w velocities by directions defined by theta_h and theta_v.

    Parameters
    ----------
    data : dict
        Dictionary containing "u", "v", and "w" velocity arrays with shape (M, N)
    theta_h : float or np.ndarray
        Horizontal rotation angle(s) in degrees, scalar or shape (M,)
    theta_v : float or np.ndarray
        Vertical rotation angle(s) in degrees, scalar or shape (M,)

    Returns
    -------
    data : dict
        Original data dictionary with "u", "v", and "w" velocity arrays rotated
    """
    # (M,) or scalar → (M, 1) for broadcasting against (M, N)
    th = np.deg2rad(np.atleast_1d(theta_h))[:, np.newaxis]
    tv = np.deg2rad(np.atleast_1d(theta_v))[:, np.newaxis]

    cos_h, sin_h = np.cos(th), np.sin(th)
    cos_v, sin_v = np.cos(tv), np.sin(tv)

    u_rot = u1 * cos_h * cos_v + u2 * sin_h * cos_v + u3 * sin_v
    v_rot = -u1 * sin_h + u2 * cos_h
    w_rot = -u1 * cos_h * sin_v - u2 * sin_h * sin_v + u3 * cos_v
    return u_rot, v_rot, w_rot
