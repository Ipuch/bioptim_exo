import biorbd
import numpy as np


def eul2quat(eul: np.ndarray) -> np.ndarray:
    rotation_matrix = biorbd.Rotation_fromEulerAngles(eul, "xyz")
    quat = biorbd.Quaternion_fromMatrix(rotation_matrix).to_array().squeeze()
    return quat


def quat2eul(quat: np.ndarray) -> np.ndarray:
    quat_biorbd = biorbd.Quaternion(quat[0], quat[1], quat[2], quat[3])
    rotation_matrix = biorbd.Quaternion.toMatrix(quat_biorbd)
    eul = biorbd.Rotation_toEulerAngles(rotation_matrix, "xyz").to_array()
    return eul
