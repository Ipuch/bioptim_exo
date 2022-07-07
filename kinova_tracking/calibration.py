import biorbd
from scipy import optimize
import numpy as np
import utils


def objective_function_param(
    p0: np.ndarray,
    biorbd_model: biorbd.Model,
    x: np.ndarray,
    x0: np.ndarray,
    markers: np.ndarray,
    nb_frames: int,
    markers_names,
):
    """
    Objective function

    Parameters
    ----------
    p0: np.ndarray
        (6x1) Generalized coordinates between ulna and piece 7, unique for all frames
    biorbd_model: biorbd.Model
        The biorbd model
    x: np.ndarray
        Generalized coordinates for all frames all dof
    x0: np.ndarray
        Generalized coordinates for the first frame
    markers: np.ndarray
        (3 x n_markers x n_frames) marker values for all frames

    Return
    ------
    The value of the objective function
    """
    nb_dof = x.shape[0]
    n_adjust = p0.shape[0]
    n_bras = nb_dof - n_adjust - 6

    table5_xyz_all_frames = 0
    table6_xy_all_frames = 0
    mark_out_all_frames = 0
    out2_all_frames = 0
    Q = np.zeros(nb_dof)
    Q[n_bras : n_bras + n_adjust] = p0
    for frame in range(nb_frames):
        thorax_markers = markers[:, 0:14, frame]
        table_markers = markers[:, 14:, frame]

        Q[:n_bras] = x[:n_bras, frame]
        Q[n_bras + n_adjust :] = x[n_bras + n_adjust :, frame]

        markers_model = biorbd_model.markers(Q)

        table5_xyz = (
            np.linalg.norm(markers_model[markers_names.index("Table:Table5")].to_array()[:] - table_markers[:, 0]) ** 2
        )
        table5_xyz_all_frames += table5_xyz

        table6_xy = (
            np.linalg.norm(markers_model[markers_names.index("Table:Table6")].to_array()[:2] - table_markers[:2, 1])
            ** 2
        )
        table6_xy_all_frames += table6_xy

        mark_out = 0
        for j in range(len(thorax_markers[0, :])):
            mark = np.linalg.norm(markers_model[j].to_array()[:] - thorax_markers[:, j]) ** 2
            mark_out += mark
        mark_out_all_frames += mark_out

        T = biorbd_model.globalJCS(x0, biorbd_model.nbSegment() - 1).to_array()
        out2 = T[2, 0] ** 2 + T[2, 1] ** 2 + T[0, 2] ** 2 + T[1, 2] ** 2 + (1 - T[2, 2]) ** 2

        out2_all_frames += out2

        print("param per frame", p0)

    print("param", p0)
    return 1000 * table5_xyz_all_frames + 1000 * table6_xy_all_frames + mark_out_all_frames + out2_all_frames


def ik_step(
    x: np.ndarray,
    biorbd_model: biorbd.Model,
    p: np.ndarray,
    table_markers: np.ndarray,
    thorax_markers: np.ndarray,
    markers_names,
):
    """
    Objective function

    Parameters
    ----------
    x: np.ndarray
        Generalized coordinates for all dof except those between ulna and piece 7, unique for all frames
    biorbd_model: biorbd.Model
        The biorbd model
    p: np.ndarray
        Generalized coordinates between ulna and piece 7
    table_markers: np.ndarray
        (3 x n_markers_on_table x n_frames) marker values for all frames
    thorax_markers: np.ndarray
        (3 x n_markers_on_wu_model x n_frames) marker values for all frames

    Return
    ------
    The value of the objective function
    """
    x_with_p = np.zeros(22)
    x_with_p[:10] = x[:10]
    x_with_p[10:16] = p
    x_with_p[16:] = x[10:]
    markers_model = biorbd_model.markers(x_with_p)
    table5_xyz = (
        np.linalg.norm(markers_model[markers_names.index("Table:Table5")].to_array()[:] - table_markers[:, 0]) ** 2
    )
    table6_xy = (
        np.linalg.norm(markers_model[markers_names.index("Table:Table6")].to_array()[:2] - table_markers[:2, 1]) ** 2
    )
    mark_list = []
    mark_out = 0
    for j in range(len(thorax_markers[0, :])):
        mark = np.linalg.norm(markers_model[j].to_array()[:] - thorax_markers[:, j]) ** 2
        mark_list.append(mark)
        mark_out += mark

    T = biorbd_model.globalJCS(x_with_p, biorbd_model.nbSegment() - 1).to_array()
    out2 = T[2, 0] ** 2 + T[2, 1] ** 2 + T[0, 2] ** 2 + T[1, 2] ** 2 + (1 - T[2, 2]) ** 2

    # out3 = 0
    # for i, value in enumerate(x[:10]):
    #     out3 += (x_with_p[i] - value) ** 2

    # out4 = 0
    # for h in range(1, 3):
    #     out4 += (x_with_p[-h] - 0.0) ** 2

    # return 1000 * table5_xyz + 1000 * table6_xy + out2 + mark_out + 10 * out3 + out4
    return 1000 * table5_xyz + 1000 * table6_xy + out2 + 1000 * mark_out


def step_2(
    biorbd_model,
    p,
    bounds,
    nb_dof_wu_model,
    nb_parameters,
    nb_frames,
    q_first_ik,
    q_output,
    markers_xp_data,
    markers_names,
):
    # build the bounds for step 2
    bounds_without_p_1 = bounds[:nb_dof_wu_model]
    bounds_without_p_2 = bounds[nb_dof_wu_model + nb_parameters :]
    bounds_without_p = np.concatenate((bounds_without_p_1, bounds_without_p_2))

    for f in range(nb_frames):
        x0_1 = q_first_ik[:nb_dof_wu_model, 0] if f == 0 else q_output[:nb_dof_wu_model, f - 1]
        x0_2 = (
            q_first_ik[nb_dof_wu_model + nb_parameters :, 0]
            if f == 0
            else q_output[nb_dof_wu_model + nb_parameters :, f - 1]
        )
        x0 = np.concatenate((x0_1, x0_2))
        IK_i = optimize.minimize(
            fun=ik_step,
            args=(
                biorbd_model,
                p,
                markers_xp_data[:, 14:, f],  # todo: remove the raw hard coded walues
                markers_xp_data[:, 0:14, f],
                markers_names,
            ),
            x0=x0,  # x0 q sans p
            bounds=bounds_without_p,
            method="trust-constr",
            jac="3-point",
            tol=1e-5,
        )
        print(f"frame {f} done")

        q_output[:nb_dof_wu_model, f] = IK_i.x[:nb_dof_wu_model]
        q_output[nb_dof_wu_model + nb_parameters :, f] = IK_i.x[nb_dof_wu_model:]

    return q_output


def arm_support_calibration(
    biorbd_model: biorbd.Model,
    markers_names: list[str],
    markers_xp_data: np.ndarray,
    q_first_ik: np.ndarray,
    nb_dof_wu_model,
    nb_parameters,
    nb_frames,
):
    """
    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model
    markers_names: list[str]
        The list of markers names
    markers_xp_data: np.ndarray
        (3 x n_markers x n_frames) marker values for all frames
    q_first_ik: np.ndarray
         Generalized coordinates for all frames all dof
    nb_dof_wu_model: int
        The number of dof for the thorax and the arm
    nb_parameters: int
        The number of dof between the ulna and piece 7
    nb_frames: int
        The number of frames
    Return
    ------
        The optimized Generalized coordinates
    """

    q0 = q_first_ik[:, 0]

    # idx_human = [0, ..., n_dof]
    # idx_support = [n_dof + 1, ..., n_dof + parameter]
    # idx_exo =

    q_output = np.zeros((biorbd_model.nbQ(), markers_xp_data.shape[2]))
    bounds = [
        (mini, maxi) for mini, maxi in zip(utils.get_range_q(biorbd_model)[0], utils.get_range_q(biorbd_model)[1])
    ]
    # nb_frames = markers_xp_data.shape[2]
    p0 = q_first_ik[nb_dof_wu_model : nb_dof_wu_model + nb_parameters, 0]

    # step 1 - param opt
    param_opt = optimize.minimize(
        fun=objective_function_param,
        args=(biorbd_model, q_first_ik, q0, markers_xp_data, nb_frames, markers_names),
        x0=p0,
        bounds=bounds[10:16],
        method="trust-constr",
        jac="3-point",
        tol=1e-5,
    )
    print(param_opt.x)

    q_first_ik[nb_dof_wu_model : nb_dof_wu_model + nb_parameters, :] = np.array([param_opt.x] * nb_frames).T
    p = param_opt.x
    q_output[nb_dof_wu_model : nb_dof_wu_model + nb_parameters, :] = np.array([param_opt.x] * nb_frames).T

    # step 2 - ik step

    q_out = step_2(
        biorbd_model,
        p,
        bounds,
        nb_dof_wu_model,
        nb_parameters,
        nb_frames,
        q_first_ik,
        q_output,
        markers_xp_data,
        markers_names,
    )

    # return support parameters, q_output
    return q_out


# todo: futur steps, redo IK with identified postion and orientation of arm support.
