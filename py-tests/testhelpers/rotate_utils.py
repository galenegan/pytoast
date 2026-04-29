import numpy as np

def nortek_3beam_T():
    # 3-beam T matrix from Nortek reference script
    T_3BEAM = np.array([[2896, 2896, 0], [-2896, 2896, 0], [-2896, -2896, 5792]], dtype=float) / 4096.0
    return T_3BEAM

def nortek_4beam_T(beam_angle: float = 25.0):
    # Synthetic 4-beam Nortek T: x/y from opposing beam pairs, z1/z2 from each pair's vertical component
    theta = np.deg2rad(beam_angle)
    a = 1 / (2 * np.sin(theta))
    b = 1 / (2 * np.cos(theta))
    T_4BEAM_NORTEK = np.array(
        [
            [a, 0.0, -a, 0.0],
            [0.0, -a, 0.0, a],
            [b, 0.0, b, 0.0],
            [0.0, b, 0.0, b],
        ]
    )
    return T_4BEAM_NORTEK

def rdi_4beam_T(beam_angle: float = 25):
    # RDI default transformation matrix
    theta = np.deg2rad(beam_angle)
    a = 1 / (2 * np.sin(theta))
    b = 1 / (4 * np.cos(theta))
    c = 1  # convex transducer head
    d = a / np.sqrt(2)
    T_4BEAM_RDI = np.array([[c * a, -c * a, 0, 0], [0, 0, -c * a, c * a], [b, b, b, b], [d, d, -d, -d]])
    return T_4BEAM_RDI

