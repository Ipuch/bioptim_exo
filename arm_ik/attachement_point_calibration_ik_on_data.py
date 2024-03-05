import biorbd_casadi as biorbd
import biorbd as biorbd_eigen
from biorbd_casadi import RotoTrans
from biorbd_casadi import get_range_q, segment_index, marker_index
import bioviz
from casadi import MX, vertcat, sumsqr, nlpsol, horzcat
import numpy as np
from data.enums import Tasks, TasksKinova
from models.enums import Models
from models.merge_biomod import merge_biomod
import data.load_events as load_events
from models.biorbd_model import NewModel, KinovaModel
import tracking.load_experimental_data as load_experimental_data

from pathlib import Path
from step_2 import (
    create_frame_from_three_points,
    penalty_rotation_matrix_cas,
    penalty_position_cas,
    penalty_not_too_far_from_previous_pose_case,
)


def main():
    task = TasksKinova.DRINK
    task_files = load_events.LoadTask(task=task, model=Models.WU_INVERSE_KINEMATICS)
    model_path_upperlimb = Models.WU_INVERSE_KINEMATICS.value
    model_mx_upperlimb = biorbd.Model(model_path_upperlimb)

    data = load_experimental_data.LoadData(
        model=model_mx_upperlimb,
        c3d_file=task_files.c3d_path,
        q_file=task_files.q_file_path,
        qdot_file=task_files.qdot_file_path,
    )

    kinova_model = KinovaModel(model=Models.KINOVA_RIGHT_SLIDE_POLAR_BASE_WITH_VARIABLES)
    marker_attachment_name = "Table:Table5"
    index_marker_attachment = data.c3d_data.c3d["parameters"]["POINT"]["LABELS"]["value"].index(marker_attachment_name)
    attachment_point_location = np.mean(data.c3d_data.c3d["data"]["points"][:, index_marker_attachment, :], axis=1)[0:3]
    kinova_model.add_header(
        model_template=Models.KINOVA_RIGHT_SLIDE_POLAR_BASE_TEMPLATE,
        attachment_point_location=attachment_point_location / 1000,
        offset=np.array([0, -0.02, 0.]),
    )
    model_path_kinova = kinova_model.model_path
    model_mx_kinova = biorbd.Model(model_path_kinova)

    x_trans = MX.sym("xtrans", 1, 1)
    y_trans = MX.sym("ytrans", 1, 1)
    rt = biorbd.RotoTrans.fromEulerAngles(MX([0, 0, 0]), horzcat(x_trans, y_trans, 0.059).T, "xyz")
    model_mx_kinova.segments()[1].setLocalJCS(model_mx_kinova, rt)

    model_eigen_kinova = biorbd_eigen.Model(model_path_kinova)
    model_eigen_upperlimb = biorbd_eigen.Model(model_path_upperlimb)

    # repeat q_upper_limb for 100 frames
    q_upper_limb = data.q
    n_frames = q_upper_limb.shape[1]
    all_rt = np.zeros((4, 4, n_frames))

    q_sym = MX.sym("q", model_eigen_kinova.nbQ() * n_frames, 1)

    x_trans_up = np.array([-0.01])
    x_trans_low = np.array([-0.290 - 0.5])

    y_trans_up = np.array([0.037 + 0.5])
    y_trans_low = np.array([0.01])

    q_up = get_range_q(model_eigen_kinova)[1][:, np.newaxis].repeat(n_frames, axis=1).flatten("F")
    q_low = get_range_q(model_eigen_kinova)[0][:, np.newaxis].repeat(n_frames, axis=1).flatten("F")

    # initialize q_opt
    q_opt = np.zeros((model_eigen_kinova.nbQ() * n_frames))

    # constraints
    constraints = []
    lbg = []
    ubg = []
    objective = 0

    for i in range(n_frames):
        q_sym_i = q_sym[model_eigen_kinova.nbQ() * i : model_eigen_kinova.nbQ() * (i + 1)]
        q_upper_limb_i = q_upper_limb[:, i]

        # do the rotation matrix from markers it would be easier
        m0_idx = marker_index(model_eigen_upperlimb, "flexion_axis0")
        m1_idx = marker_index(model_eigen_upperlimb, "flexion_axis1")
        m2_idx = marker_index(model_eigen_upperlimb, "ulna_longitudinal_frame")

        markers = model_eigen_upperlimb.markers(q_upper_limb_i)
        m0 = markers[m0_idx].to_array()
        m1 = markers[m1_idx].to_array()
        m2 = markers[m2_idx].to_array()

        rt = create_frame_from_three_points(m0, m1, m2)
        all_rt[:, :, i] = rt
        rotation = rt[:3, :3]
        position = rt[:3, 3]

        objective += 10 * penalty_rotation_matrix_cas(model_mx_kinova, q_sym_i, rotation)
        objective += 10 * penalty_position_cas(model_mx_kinova, q_sym_i, position)
        if i != 0:
            objective += 1e-5 * penalty_not_too_far_from_previous_pose_case(
                model_mx_kinova,
                q_sym_i[:-3],
                q_sym[model_eigen_kinova.nbQ() * (i - 1) : model_eigen_kinova.nbQ() * i][:-3],
            )
        # penality for the z axis of the support

        # it should not go under the table
        constraints += [model_mx_kinova.globalJCS(q_sym_i, model_mx_kinova.nbSegment() - 1).trans().to_mx()[2]]
        lbg += [0]
        ubg += [np.inf]

    # Create a NLP solver
    prob = {"f": objective, "x": vertcat(q_sym, x_trans, y_trans), "g": vertcat(*constraints)}
    opts = {"ipopt": {"max_iter": 5000, "linear_solver": "mumps", "print_level": 5}}
    solver = nlpsol('solver', 'ipopt', prob, opts)

    # Solve the NLP
    q_init = np.array([0.2] * model_eigen_kinova.nbQ() * n_frames)
    x_trans_init = np.array([-0.290])
    y_trans_init = np.array([0.037])

    sol = solver(
        x0=np.concatenate((q_init, x_trans_init, y_trans_init)),
        lbx=np.concatenate((q_low, x_trans_low, y_trans_low)),
        ubx=np.concatenate((q_up, x_trans_up, y_trans_up)),
        lbg=lbg,
        ubg=ubg,
    )
    q_opt_all = sol["x"].full().flatten()
    print(sol["x"].full().flatten()[-2:])

    np.save("q_opt", q_opt_all)

    q_opt = q_opt_all[:model_eigen_kinova.nbQ() * n_frames].reshape((model_eigen_kinova.nbQ(), n_frames), order="F")

    #save q_opt and q_opt_2
    np.save("q_opt", q_opt)
    # np.save("q_opt_2", q_opt_2)

    parent_path = Path(model_path_kinova).parent
    output_path = str(parent_path / "merged.bioMod")
    merge_biomod(model_path_kinova, model_path_upperlimb, output_path)
    q_tot = np.concatenate((q_opt, q_upper_limb), axis=0)

    # Display the rotation_matrix and translation terms in a (3, 4) subplot for each frame and see the correlation
    # between the two

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 4)

    kinova_rts = np.zeros((4, 4, n_frames))
    kinova_rts2 = np.zeros((4, 4, n_frames))
    for i in range(n_frames):
        kinova_rts[:, :, i] = model_eigen_kinova.globalJCS(q_opt[:, i], model_eigen_kinova.nbSegment() - 1).to_array()

    for row in range(3):
        for col in range(4):
            axs[row, col].plot(all_rt[row, col, :], 'b-')
            axs[row, col].plot(kinova_rts[row, col, :], 'r--')
            axs[row, col].plot(kinova_rts2[row, col, :], '--', color='#FFB6C1')
            axs[row, col].set_title(f'row {row}, col {col}')

    plt.show()

    viz = bioviz.Viz(output_path, show_floor=False, show_global_ref_frame=False, show_muscles=False)
    from extra_viz import VtkFrameModel
    vtkObject = VtkFrameModel(viz.vtk_window, normalized=True)

    i = 1
    while viz.vtk_window.is_active:
        # Update the markers
        vtkObject.update_frame(rt=all_rt[:, :, i])
        viz.set_q(q_tot[:, i])
        i = (i+1) % n_frames

    viz.exec()


if __name__ == "__main__":
    main()
