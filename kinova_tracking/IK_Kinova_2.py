import biorbd
from scipy import optimize
import numpy as np
import utils


def IK_Kinova(
        biorbd_model: biorbd.Model,
        markers_names: list[str],
        markers_xp_data: np.ndarray,
        q0: np.ndarray,
        new_q: np.ndarray,
):
    """
    :param markers_xp_data:
    :param markers_names:
    :param biorbd_model:
    :param table:
    :param thorax:
    :param q0:
    """

    def objective_function_param(p0, biorbd_model, x, x0, markers, nb_frames):
        table5_xyz_all_frames = 0
        table6_xy_all_frames = 0
        mark_list_all_frames = []
        mark_out_all_frames = 0
        out2_all_frames = 0
        for frame in range(nb_frames):
            thorax_markers = markers[:, 0:14, frame]
            table_markers = markers[:, 14:, frame]
            markers_model = biorbd_model.markers(x0)

            table5_xyz = np.linalg.norm(
                markers_model[markers_names.index('Table:Table5')].to_array()[:] - table_markers[:, 0]) ** 2
            table5_xyz_all_frames += table5_xyz

            table6_xy = np.linalg.norm(
                markers_model[markers_names.index('Table:Table6')].to_array()[:2] - table_markers[:2, 1]) ** 2
            table6_xy_all_frames += table6_xy

            mark_list = []
            mark_out = 0
            for j in range(len(thorax_markers[0, :])):
                mark = np.linalg.norm(markers_model[j].to_array()[:] - thorax_markers[:, j]) ** 2
                mark_list.append(mark)
                mark_out += mark
            mark_out_all_frames += mark_out
            mark_list_all_frames.append(mark_list)

            T = biorbd_model.globalJCS(x0, biorbd_model.nbSegment() - 1).to_array()
            out2 = T[2, 0] ** 2 + T[2, 1] ** 2 + T[0, 2] ** 2 + T[1, 2] ** 2 + (1 - T[2, 2]) ** 2

            out2_all_frames += out2

            p_test = np.linalg.norm(
                p0 - np.array([0]*6)) ** 2

            x[:, frame][10:16] = p0
            x0 = x[:, frame]
            print("param per frame", p0)

        print("param", p0)
        return 1000*p_test + table5_xyz_all_frames + table6_xy_all_frames + mark_out_all_frames + out2_all_frames

    def objective_function(x, biorbd_model, q_ik_thorax, table_markers, thorax_markers):
        new_x = q_ik_thorax
        new_x[16:] = x
        markers_model = biorbd_model.markers(new_x)
        table5_xyz = np.linalg.norm(
            markers_model[markers_names.index('Table:Table5')].to_array()[:] - table_markers[:, 0]) ** 2
        table6_xy = np.linalg.norm(
            markers_model[markers_names.index('Table:Table6')].to_array()[:2] - table_markers[:2, 1]) ** 2
        # mark_list = []
        # mark_out = 0
        # for j in range(len(thorax_markers[0, :])):
        #     mark = np.linalg.norm(markers_model[j].to_array()[:] - thorax_markers[:, j]) ** 2
        #     mark_list.append(mark)
        #     mark_out += mark

        T = biorbd_model.globalJCS(new_x, biorbd_model.nbSegment() - 1).to_array()
        out2 = T[2, 0] ** 2 + T[2, 1] ** 2 + T[0, 2] ** 2 + T[1, 2] ** 2 + (1 - T[2, 2]) ** 2

        out3 = 0
        for i, value in enumerate(q_ik_thorax[:16]):
            out3 += (new_x[i] - value) ** 2

        out4 = 0
        for h in range(1, 3):
            out4 += (new_x[-h] - 0.) ** 2

        # return 1000 * table5_xyz + 1000 * table6_xy + out2 + mark_out + 10 * out3 + out4
        return 100000 * table5_xyz + 100000 * table6_xy + out2 + 10 * out3 + 10 * out4

    q = np.zeros((biorbd_model.nbQ(), markers_xp_data.shape[2]))
    bounds = [(mini, maxi) for mini, maxi in
              zip(utils.get_range_q(biorbd_model)[0], utils.get_range_q(biorbd_model)[1])]

    nb_frames = markers_xp_data.shape[2]

    p0 = q0[10:16]
    p0 = [0]*6
    # jspkoimaitre = True
    # while jspkoimaitre:
    # p0 = q_ik_1[10:16, :]
    param_opt = optimize.minimize(
        fun=objective_function_param,
        args=(biorbd_model, new_q, q0, markers_xp_data, nb_frames),
        x0=p0,
        bounds=bounds[10:16],
        method="trust-constr",
        jac="3-point",
        tol=1e-5,
    )
    print(param_opt.x)
    print("minimize", p0)

    new_q[10:16, :] = np.array([param_opt.x]*new_q.shape[1]).T
    q_pre = np.array((0.0, 0.2618, 0.3903, 1.7951, 0.6878, 0.3952))

    x0_2 = q_pre
    q[:16, :] = new_q[:16, :]
    for f in range(nb_frames):
        x0 = x0_2 if f == 0 else q[16:, f - 1]
        IK_i = optimize.minimize(
            fun=objective_function,
            args=(biorbd_model, new_q[:, f], markers_xp_data[:, 14:, f], markers_xp_data[:, 0:14, f]),
            # todo: automatiser slicing
            x0=x0,
            bounds=bounds[16:],
            method="trust-constr",
            jac="3-point",
            tol=1e-5,
        )
        print(f"frame {f} done")

        q[16:, f] = IK_i.x

    return q
