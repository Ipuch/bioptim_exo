import biorbd
from scipy import optimize
import numpy as np
import utils


def IK_Kinova(
    biorbd_model: biorbd.Model,
    markers_names: list[str],
    markers: np.ndarray,
    q0: np.ndarray,
    q_ik_1: np.ndarray,
):
    """
    :param markers:
    :param markers_names:
    :param biorbd_model:
    :param table:
    :param thorax:
    :param q0:
    """

    def objective_function(
        x: np.ndarray,
        biorbd_model: biorbd.Model,
        q_ik_thorax: np.ndarray,
        table_markers: np.ndarray,
        thorax_markers: np.ndarray,
    ):
        markers_model = biorbd_model.markers(x)
        table5_xyz = (
            np.linalg.norm(markers_model[markers_names.index("Table:Table5")].to_array()[:] - table_markers[:, 0]) ** 2
        )
        table6_xy = (
            np.linalg.norm(markers_model[markers_names.index("Table:Table6")].to_array()[:2] - table_markers[:2, 1])
            ** 2
        )
        mark_list = []
        mark_out = 0
        for j in range(len(thorax_markers[0, :])):
            mark = np.linalg.norm(markers_model[j].to_array()[:] - thorax_markers[:, j]) ** 2
            mark_list.append(mark)
            mark_out += mark

        T = biorbd_model.globalJCS(x, biorbd_model.nbSegment() - 1).to_array()
        out2 = T[2, 0] ** 2 + T[2, 1] ** 2 + T[0, 2] ** 2 + T[1, 2] ** 2 + (1 - T[2, 2]) ** 2

        # Minimize the q of thorax
        out3 = 0
        for i, value in enumerate(q_ik_thorax):
            out3 += (x[i] - value) ** 2

        # minimize value of angles of piece 1 and 2
        out4 = 0
        for h in range(1, 3):
            out4 += (x[-h] - 0.0) ** 2

        return 1000 * table5_xyz + 1000 * table6_xy + out2 + mark_out + 10 * out3 + out4

    idx_markers_kinova = [
        i
        for i, value in enumerate(biorbd_model.markerNames())
        if value.to_string() == "Table:Table6" or value.to_string() == "Table:Table5"
    ]
    idx_markers_human_body = [
        i
        for i, value in enumerate(biorbd_model.markerNames())
        if not value.to_string() == "Table:Table6" or value.to_string() == "Table:Table5"
    ]

    q = np.zeros((biorbd_model.nbQ(), markers.shape[2]))
    bounds = [
        (mini, maxi) for mini, maxi in zip(utils.get_range_q(biorbd_model)[0], utils.get_range_q(biorbd_model)[1])
    ]
    for f in range(markers.shape[2]):
        x0 = q0 if f == 0 else q[:, f - 1]
        pos = optimize.minimize(
            fun=objective_function,
            args=(biorbd_model, q_ik_1[:, f], markers[:, idx_markers_kinova, f], markers[:, idx_markers_human_body, f]),
            x0=x0,
            bounds=bounds,
            method="trust-constr",
            jac="3-point",
            tol=1e-5,
        )
        q[:, f] = pos.x
        print(f"frame {f} done")

    return q
