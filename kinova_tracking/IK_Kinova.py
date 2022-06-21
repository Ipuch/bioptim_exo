import biorbd
from scipy import optimize
import numpy as np
import utils


def IK_Kinova(biorbd_model: biorbd.Model, markers_names: list[str], q0: np.ndarray, targetd: np.ndarray, targetp: np.ndarray):
    """
    :param markers_names:
    :param biorbd_model:
    :param targetd:
    :param targetp:
    :param q0:
    """
    def objective_function(x, *args, **kwargs):
        markers = biorbd_model.markers(x)
        out1 = np.linalg.norm(markers[0].to_array() - targetd) ** 2
        out3_1 = (markers[markers_names.index('md0')].to_array()[0] - targetp[0]) ** 2
        out3_2 = (markers[markers_names.index('md0')].to_array()[1] - targetp[1]) ** 2
        out3_3 = (markers[markers_names.index('md0')].to_array()[2] - targetp[2]) ** 2
        out3 = out3_1 + out3_2 + out3_3
        # out3 = np.linalg.norm(markers[X_names.index('md0')].to_array() - targetp) ** 2
        T = biorbd_model.globalJCS(x, biorbd_model.nbSegment() - 1).to_array()
        out2 = T[2, 0] ** 2 + T[2, 1] ** 2 + T[0, 2] ** 2 + T[1, 2] ** 2 + (1 - T[2, 2]) ** 2

        # return 10 * out1 + out2 + 10 * out3 + 1000 * (q_3e_pivot - q_3epivot_desired)
        return 10 * out1 + out2 + 10 * out3

    pos = optimize.least_squares(
        objective_function,
        args=(biorbd_model, targetd, targetp),
        x0=q0,
        bounds=utils.get_range_q(biorbd_model),
        verbose=2,
        method="trf",
        jac="3-point",
        ftol=1e-5,
        gtol=1e-5,
    )

    return pos.x


def IK_Kinova_RT(model_path: str, q0: np.ndarray, targetd: np.ndarray, targetp: np.ndarray):
    """

    :param targetd:
    :param targetp:
    :param q0:
    :type model_path: object
    """
    m = biorbd.Model(model_path)
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

    pos = optimize.least_squares(
        objective_function,
        args=(m, targetd, targetp),
        x0=q0,
        bounds=bounds,
        verbose=2,
        method="trf",
        jac="3-point",
        ftol=2.22e-16,
        gtol=2.22e-16,
    )
    # print(pos)
    # print(f"Optimal q for the assistive arm at {target} is:\n{pos.x}\n"
    #       f"with cost function = {objective_function(pos.x)}")
    # print(m.globalJCS(q0, m.nbSegment()-1).to_array())
    # print(m.globalJCS(pos.x, m.nbSegment()-1).to_array())
    # Verification
    # q = np.tile(pos.x, (10, 1)).T
    # q = np.tile(q0, (10, 1)).T
    return pos.x