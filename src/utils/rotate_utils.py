import numpy as np
from typing import Tuple, Dict


def coord_transform_3_beam(
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
    T = transformation_matrix.copy()
    if np.any(T > 1000):
        T /= 4096.0

    T_flip = T.copy()
    if orientation == "down":
        T_flip[1, :] = -T_flip[1, :]
        T_flip[2, :] = -T_flip[2, :]

    heading_plus_dec = (heading + declination) % 360
    h_rad = np.radians(heading_plus_dec - 90)
    p_rad = np.radians(pitch)
    r_rad = np.radians(roll)

    # Heading matrix
    H = np.array([[np.cos(h_rad), np.sin(h_rad), 0], [-np.sin(h_rad), np.cos(h_rad), 0], [0, 0, 1]])

    # Pitch and Roll matrix
    P = np.array(
        [
            [np.cos(p_rad), -np.sin(p_rad) * np.sin(r_rad), -np.cos(r_rad) * np.sin(p_rad)],
            [0, np.cos(r_rad), -np.sin(r_rad)],
            [np.sin(p_rad), np.sin(r_rad) * np.cos(p_rad), np.cos(p_rad) * np.cos(r_rad)],
        ]
    )

    # XYZ to ENU matrix
    R = H @ P

    # Velocity array
    U = np.vstack((u1.reshape(1, -1), u2.reshape(1, -1), u3.reshape(1, -1)))

    if coords_in == "beam" and coords_out == "enu":
        U_rot = R @ T_flip @ U
    elif coords_in == "enu" and coords_out == "beam":
        U_rot = np.linalg.inv(T_flip) @ np.linalg.inv(R) @ U
    elif coords_in == "beam" and coords_out == "xyz":
        U_rot = T @ U
    elif coords_in == "xyz" and coords_out == "beam":
        U_rot = np.linalg.inv(T) @ U
    elif coords_in == "enu" and coords_out == "xyz":
        U_rot = T @ np.linalg.inv(T_flip) @ np.linalg.inv(R) @ U
    else:
        raise ValueError("Invalid coordinate transformation.")

    # TODO: store T inverses so we only need to calculate once
    u1_rot = U_rot[0, :]
    u2_rot = U_rot[1, :]
    u3_rot = U_rot[2, :]
    return (u1_rot, u2_rot, u3_rot)

def coord_transform_4_beam(
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
    Implementation of Nortek's coordinate transformation for the 4-beam Signature instrument

    https://support.nortekgroup.com/hc/en-us/articles/360029820971-How-is-a-coordinate-transformation-done
    Parameters
    ----------
    u1
    u2
    u3
    u4
    heading
    pitch
    roll
    transformation_matrix
    declination
    coords_in
    coords_out

    Returns
    -------

    """

    T = transformation_matrix.copy()
    if np.any(T > 1000):
        T /= 4096.0

    # Velocity array
    U = np.vstack((u1.reshape(1, -1), u2.reshape(1, -1), u3.reshape(1, -1), u4.reshape(1, -1)))


    if coords_in == "beam" and coords_out == "xyz":
        U_rot = T @ U
    elif (coords_in == "beam" or coords_in == "xyz") and coords_out == "enu":
        R = np.zeros((4, 4))
        heading_plus_dec = (heading + declination) % 360
        h_rad = np.radians(heading_plus_dec - 90)
        p_rad = np.radians(pitch)
        r_rad = np.radians(roll)
        H = np.array([[np.cos(h_rad), np.sin(h_rad), 0], [-np.sin(h_rad), np.cos(h_rad), 0], [0, 0, 1]])
        P = np.array(
            [
                [np.cos(p_rad), -np.sin(p_rad) * np.sin(r_rad), -np.cos(r_rad) * np.sin(p_rad)],
                [0, np.cos(r_rad), -np.sin(r_rad)],
                [np.sin(p_rad), np.sin(r_rad) * np.cos(p_rad), np.cos(p_rad) * np.cos(r_rad)],
            ]
        )
        R[:3, :3] = H @ P

        # Trusting Nortek
        R[0, 2] /= 2
        R[0, 3] = R[0, 2]
        R[1, 2] /= 2
        R[1, 3] = R[1, 2]
        R[3, :] = R[2, :]
        R[2, 3] = 0
        R[3, 3] = R[2, 2]
        R[3, 2] = 0

        if coords_in == "beam":
            U = T @ U

        if orientation == "down":
            U *= -1

        U_rot = R @ U
    else:
        raise ValueError("Invalid coordinate transformation.")

    u1_rot = U_rot[0, :]
    u2_rot = U_rot[1, :]
    u3_rot = U_rot[2, :]
    u4_rot = U_rot[3, :]
    return (u1_rot, u2_rot, u3_rot, u4_rot)

def align_with_principal_axis(data: dict) -> Tuple:
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
    u_bar = np.mean(data["u"], axis=1, keepdims=True)
    v_bar = np.mean(data["v"], axis=1, keepdims=True)
    u_prime = data["u"] - u_bar
    v_prime = data["v"] - v_bar
    u_var = np.mean(u_prime**2, axis=1)
    v_var = np.mean(v_prime**2, axis=1)
    cv = np.mean(u_prime * v_prime, axis=1)

    # Direction of maximum variance in xy-plane (heading)
    theta_h_radians = 0.5 * np.arctan2(2.0 * cv, (u_var - v_var))

    # Pitch angle
    u_rot = data["u"] * np.cos(theta_h_radians) + data["v"] * np.sin(theta_h_radians)
    u_rot_bar = np.mean(u_rot, axis=1)
    w_bar = np.mean(data["w"], axis=1)
    theta_v_radians = np.arctan2(w_bar, u_rot_bar)

    out = (np.rad2deg(theta_h_radians), np.rad2deg(theta_v_radians))
    return out


def align_with_flow(burst_data: dict) -> Tuple:
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

    u_bar = np.mean(burst_data["u"], axis=1)
    v_bar = np.mean(burst_data["v"], axis=1)
    w_bar = np.mean(burst_data["w"], axis=1)
    U = np.sqrt(u_bar**2 + v_bar**2)
    theta_h = np.arctan2(v_bar, u_bar)
    theta_v = np.arctan2(w_bar, U)
    out = (np.rad2deg(theta_h), np.rad2deg(theta_v))
    return out


def rotate_velocity_by_theta(data: Dict[str, np.ndarray], theta_h, theta_v):
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

    u_rot = data["u"] * cos_h * cos_v + data["v"] * sin_h * cos_v + data["w"] * sin_v
    v_rot = -data["u"] * sin_h + data["v"] * cos_h
    w_rot = -data["u"] * cos_h * sin_v - data["v"] * sin_h * sin_v + data["w"] * cos_v
    data["u"] = u_rot
    data["v"] = v_rot
    data["w"] = w_rot
    return data
