import biorbd as biorbd_eigen
from scipy import optimize
import numpy as np


def IK_Kinova(model_path: str, q0: np.ndarray, targetd: np.ndarray, targetp: np.ndarray):
    """

    :param targetd:
    :param targetp:
    :param q0:
    :type model_path: object
    """
    m = biorbd_eigen.Model(model_path)
    bound_min = []
    bound_max = []
    for i in range(m.nbSegment()):
        seg = m.segment(i)
        for r in seg.QRanges():
            bound_min.append(r.min())
            bound_max.append(r.max())
    bounds = (bound_min, bound_max)

    def objective_function(x, *args, **kwargs):
        markers = m.markers(x)
        out1 = np.linalg.norm(markers[0].to_array() - targetd) ** 2
        out3 = np.linalg.norm(markers[-1].to_array() - targetp) ** 2
        T = m.globalJCS(x, m.nbSegment() - 1).to_array()
        out2 = T[2, 0] ** 2 + T[2, 1] ** 2 + T[0, 2] ** 2 + T[1, 2] ** 2 + (1 - T[2, 2]) ** 2

        # print(out2)
        # print(out1)
        return 10 * out1 + out2 + 10 * out3

    pos = optimize.least_squares(objective_function, args=(m, targetd, targetp), x0=q0,
                                 bounds=bounds, verbose=2, method='trf',
                                 jac='3-point', ftol=2.22e-16, gtol=2.22e-16)
    # print(pos)
    # print(f"Optimal q for the assistive arm at {target} is:\n{pos.x}\n"
    #       f"with cost function = {objective_function(pos.x)}")
    # print(m.globalJCS(q0, m.nbSegment()-1).to_array())
    # print(m.globalJCS(pos.x, m.nbSegment()-1).to_array())
    # Verification
    # q = np.tile(pos.x, (10, 1)).T
    # q = np.tile(q0, (10, 1)).T
    return pos.x


def IK_Kinova_RT(model_path: str, q0: np.ndarray, targetd: np.ndarray, targetp: np.ndarray):
    """

    :param targetd:
    :param targetp:
    :param q0:
    :type model_path: object
    """
    m = biorbd_eigen.Model(model_path)
    bound_min = []
    bound_max = []
    for i in range(m.nbSegment()):
        seg = m.segment(i)
        for r in seg.QRanges():
            bound_min.append(r.min())
            bound_max.append(r.max())
    bounds = (bound_min, bound_max)

    def objective_function(x, *args, **kwargs):
        markers = m.markers(x)
        out1 = np.linalg.norm(markers[0].to_array() - targetd) ** 2
        out3 = np.linalg.norm(markers[-1].to_array() - targetp) ** 2
        T1 = m.globalJCS(x, m.nbSegment() - 1).to_array()
        out2 = T1[2, 0] ** 2 + T1[2, 1] ** 2 + T1[0, 2] ** 2 + T1[1, 2] ** 2 + (1 - T1[2, 2]) ** 2
        T2 = m.globalJCS(x, 0).to_array()
        out4 = np.sum((T2[:3, :3] - np.eye(3)) ** 2)
        # print(out2)
        # print(out1)
        return 10 * out1 + out2 + 10 * out3 + out4

    pos = optimize.least_squares(objective_function, args=(m, targetd, targetp), x0=q0,
                                 bounds=bounds, verbose=2, method='trf',
                                 jac='3-point', ftol=2.22e-16, gtol=2.22e-16)
    # print(pos)
    # print(f"Optimal q for the assistive arm at {target} is:\n{pos.x}\n"
    #       f"with cost function = {objective_function(pos.x)}")
    # print(m.globalJCS(q0, m.nbSegment()-1).to_array())
    # print(m.globalJCS(pos.x, m.nbSegment()-1).to_array())
    # Verification
    # q = np.tile(pos.x, (10, 1)).T
    # q = np.tile(q0, (10, 1)).T
    return pos.x