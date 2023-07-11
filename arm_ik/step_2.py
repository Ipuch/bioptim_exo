import biorbd_casadi as biorbd
import biorbd as biorbd_eigen
from biorbd_casadi import RotoTrans
from biorbd_casadi import get_range_q, segment_index, marker_index
import bioviz
from casadi import MX, vertcat, sumsqr, nlpsol
import numpy as np

from models.enums import Models
from models.merge_biomod import merge_biomod

from pathlib import Path


def penalty_rotation_matrix_cas(model, x: MX, rotation_matrix_ref) -> MX:
    """
    Calculate the penalty cost for rotation matrix

    Parameters
    ----------
    x : MX
        the entire vector with q and p

    Returns
    -------
    The cost of the penalty function

    """
    rotation_matrix = model.globalJCS(x, model.nbSegment() - 1).rot().to_mx()
    # we want to compare the rotation matrix of the part 7 with the rotation matrix of the reference

    R = rotation_matrix.T @ rotation_matrix_ref

    rot_matrix_list_model = [
        (R[0,0] - 1),
        R[0,1],
        R[0,2],
        R[1,0],
        (R[1,1] - 1),
        R[1,2],
        R[2,0],
        R[2,1],
        (R[2,2] - 1),
    ]

    return sumsqr(vertcat(*rot_matrix_list_model))


def penalty_position_cas(model, x: MX, position_ref) -> MX:
    """
    Calculate the penalty cost for position

    Parameters
    ----------
    x : MX
        the entire vector with q and p

    Returns
    -------
    The cost of the penalty function

    """
    rotation_matrix = model.globalJCS(x, model.nbSegment() - 1).trans().to_mx()

    # we want to compare the position of the last part with the reference
    position_list_model = [
        rotation_matrix[0] - position_ref[0],
        rotation_matrix[1] - position_ref[1],
        rotation_matrix[2] - position_ref[2],
    ]

    return sumsqr(vertcat(*position_list_model))


def create_frame_from_three_points(p1, p2, p3) -> np.ndarray:
    """
    Create a homogenous transform from three points. middle of p1 and p2 is the origin of the frame
    p1->p2 is the y axis of the frame
    z from cross product of p1->p2 and midp1p2->p3
    x from cross product of y and z
    """
    mid_p1_p2 = (p1 + p2) / 2

    x = (p3 - mid_p1_p2) / np.linalg.norm(p3 - mid_p1_p2)
    y_temp = (p2 - p1) / np.linalg.norm(p2 - p1)
    z = np.cross(x, y_temp) / np.linalg.norm(np.cross(x, y_temp))
    y = np.cross(z, x) / np.linalg.norm(np.cross(z, x))

    x0 = np.concatenate((x, [0]))
    y0 = np.concatenate((y, [0]))
    z0 = np.concatenate((z, [0]))
    t1 = np.concatenate((mid_p1_p2, [1]))

    return np.hstack((x0.reshape(4, 1), y0.reshape(4, 1), z0.reshape(4, 1), t1.reshape(4, 1)))

def main():
    model_path_kinova = Models.KINOVA_RIGHT_SLIDE.value
    model_path_upperlimb = Models.WU_INVERSE_KINEMATICS.value

    model_mx_kinova = biorbd.Model(model_path_kinova)
    model_mx_upperlimb = biorbd.Model(model_path_upperlimb)

    model_eigen_kinova = biorbd_eigen.Model(model_path_kinova)
    model_eigen_upperlimb = biorbd_eigen.Model(model_path_upperlimb)

    # get position and orientation of the forearm
    q_upper_limb = np.array([0.0, 0.5, 0.45, np.pi / 2, 0.0, 0.0,  # floating base
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             1.8, 0.0,  # elbow and radioulnar joints
                             ])
    # repeat q_upper_limb for 100 frames
    n_frames = 2
    all_rt = np.zeros((4, 4, n_frames))

    q_upper_limb = np.tile(q_upper_limb, (n_frames, 1)).T
    q_upper_limb[-2, :] = np.linspace(1.3, 2, n_frames)
    # ulna_idx = segment_index(model_mx_upperlimb, "ulna")
    # ulna_idx = segment_index(model_mx_upperlimb, "ulna_elbow_flexion")
    # ulna_idx = segment_index(model_mx_upperlimb, "ulna_rotation2")
    ulna_idx = segment_index(model_mx_upperlimb, "ulna_reset_axis")

    q_sym = MX.sym("q", model_eigen_kinova.nbQ(), 1)
    q_up = [20] * model_eigen_kinova.nbQ()
    q_low = [-20] * model_eigen_kinova.nbQ()

    # q_up = get_range_q(model_eigen_kinova)[1]
    # q_low = get_range_q(model_eigen_kinova)[0]

    # initialize q_opt
    q_opt = np.zeros((model_eigen_kinova.nbQ(), n_frames))

    for i in range(n_frames):
        q_upper_limb_i = q_upper_limb[:, i]

        # rt = model_eigen_upperlimb.globalJCS(q_upper_limb_i, ulna_idx)
        # r_offset = biorbd_eigen.Rotation.fromEulerAngles(rot=np.array([-np.pi / 2, 0, 0]), seq="xyz").to_array()
        #
        # rotation = r_offset @ rt.rot().to_array()
        # position = rt.trans().to_array()

        # do the rotation matrix from markers it would be easier
        m0_idx = marker_index(model_eigen_upperlimb, "flexion_axis0")
        m1_idx = marker_index(model_eigen_upperlimb, "flexion_axis1")
        m2_idx = marker_index(model_eigen_upperlimb, "ulna_longitudinal_frame")   # todo: still need to fix this marker...

        markers = model_eigen_upperlimb.markers(q_upper_limb_i)
        m0 = markers[m0_idx].to_array()
        m1 = markers[m1_idx].to_array()
        m2 = markers[m2_idx].to_array()

        rt = create_frame_from_three_points(m0, m1, m2)
        all_rt[:, :, i] = rt
        rotation = rt[:3, :3]
        position = rt[:3, 3]

        if i == 0:
            q_init = [0.1] * model_eigen_kinova.nbQ()
        else:
            q_init = q_opt[:, i - 1]

        # objectives
        objective = 0
        objective += penalty_rotation_matrix_cas(model_mx_kinova, q_sym, rotation)
        objective += 100 * penalty_position_cas(model_mx_kinova, q_sym, position)
        # penality for the z axis of the support

        # constraints
        constraints = []
        lbg = []
        ubg = []

        # origin_kinova = np.array([0.0, 0.0])
        # radius_three_bar = np.linalg.norm([-0.08842, -0.02369]) + 2 * 0.120 - np.linalg.norm(origin_kinova)
        # # it should stay in the circle defined by the radius of the three bar mechanism
        # constraints += [q_sym[0] ** 2 + q_sym[1] ** 2 - radius_three_bar ** 2]
        # lbg += [-np.inf]
        # ubg += [0]
        # # it should not go under the table
        # constraints += [model_mx_kinova.globalJCS(q_sym, model_mx_kinova.nbSegment() - 1).trans().to_mx()[2]]
        # lbg += [0]
        # ubg += [np.inf]

        # Create a NLP solver
        prob = {"f": objective, "x": q_sym, "g": vertcat(*constraints)}
        opts = {"ipopt": {"max_iter": 5000, "linear_solver": "mumps"}}
        solver = nlpsol('solver', 'ipopt', prob, opts)

        # Solve the NLP
        sol = solver(
            x0=q_init,
            lbx=q_low,
            ubx=q_up,
            lbg=lbg,
            ubg=ubg,
        )
        q_opt[:, i] = sol["x"].full().flatten()
        print(q_opt[:, i])
        if i>0:
            print(q_opt[:, i] - q_opt[:, i-1])
        # # position of the end effector
        # print("position of the end effector:")
        # print(model_eigen_kinova.globalJCS(q_opt[:, i], model_eigen_kinova.nbSegment() - 1).trans().to_array())
        # print("expected position of the end effector:")
        # print(position)
        print("rotation matrix of the end effector:")
        print(model_eigen_kinova.globalJCS(q_opt[:, i], model_eigen_kinova.nbSegment() - 1).rot().to_array())

        print("expected rotation matrix of the end effector:")
        print(rt[:3, :3])

    parent_path = Path(model_path_kinova).parent
    output_path = str(parent_path / "merged.bioMod")
    merge_biomod(model_path_kinova, model_path_upperlimb, output_path)
    q_tot = np.concatenate((q_opt, q_upper_limb), axis=0)

    viz = bioviz.Viz(output_path, show_floor=False, show_global_ref_frame=False, show_muscles=False)
    from extra_viz import VtkFrameModel
    vtkObject = VtkFrameModel(viz.vtk_window, normalized=True)

    i = 1
    while viz.vtk_window.is_active:
        # Update the markers
        vtkObject.update_frame(rt=all_rt[:, :, i])
        viz.set_q(q_tot[:, i])
        # i = (i+1) % n_frames

    # viz.exec()


if __name__ == "__main__":
    main()
