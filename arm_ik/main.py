import biorbd_casadi as biorbd
import biorbd as biorbd_eigen
from biorbd_casadi import RotoTrans
from biorbd_casadi import get_range_q, segment_index
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
    rot_matrix_list_model = [
        rotation_matrix[0, 0] - rotation_matrix_ref[0, 0],
        rotation_matrix[1, 0] - rotation_matrix_ref[1, 0],
        rotation_matrix[2, 0] - rotation_matrix_ref[2, 0],
        rotation_matrix[0, 1] - rotation_matrix_ref[0, 1],
        rotation_matrix[1, 1] - rotation_matrix_ref[0, 1],
        rotation_matrix[1, 2] - rotation_matrix_ref[1, 2],
        rotation_matrix[0, 2] - rotation_matrix_ref[0, 2],
        rotation_matrix[2, 1] - rotation_matrix_ref[2, 1],
        rotation_matrix[2, 2] - rotation_matrix_ref[2, 2],
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


def ik_kinova():
    model_path = Models.KINOVA_RIGHT_SLIDE.value
    model = biorbd.Model(model_path)

    # set a position and orientation of the forearm
    position = np.array([0.0, 0.2, 0.2])
    orientation = np.array([0, 0.0, 0.1])  # zyx euler angles

    rt = RotoTrans.fromEulerAngles(rot=orientation, trans=position, seq="zyx").to_mx()

    # set the forearm radius
    radius = 0.035

    q_sym = MX.sym("q", model.nbQ(), 1)
    q_init = [0] * model.nbQ()
    q_up = get_range_q(model)[1]
    q_low = get_range_q(model)[0]

    objective = 0
    objective += penalty_rotation_matrix_cas(model, q_sym, rt)
    objective += 100 * penalty_position_cas(model, q_sym, position)

    constraints = []
    lbg = []
    ubg = []
    origin_kinova = np.array([0.0, 0.0])
    radius_three_bar = np.linalg.norm([-0.08842, -0.02369]) + 2 * 0.120 - np.linalg.norm(origin_kinova)
    # it should stay in the circle defined by the radius of the three bar mechanism
    constraints += [q_sym[0] ** 2 + q_sym[1] ** 2 - radius_three_bar ** 2]
    lbg += [-np.inf]
    ubg += [0]
    # it should not go under the table
    constraints += [model.globalJCS(q_sym, model.nbSegment() - 1).trans().to_mx()[2]]
    lbg += [0]
    ubg += [np.inf]

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
    q_opt = sol["x"].full().flatten()

    # position of the end effector
    model_eigen = biorbd_eigen.Model(model_path)
    print("position of the end effector:")
    print(model_eigen.globalJCS(q_opt, model.nbSegment() - 1).trans().to_array())
    print("expected position of the end effector:")
    print(position)
    print("rotation matrix of the end effector:")
    print(model_eigen.globalJCS(q_opt, model.nbSegment() - 1).rot().to_array())
    print("expected rotation matrix of the end effector:")
    print(biorbd_eigen.RotoTrans.fromEulerAngles(rot=orientation, trans=position, seq="zyx").rot().to_array())

    viz = bioviz.Viz(model_path, show_floor=False, show_global_ref_frame=False)
    viz.load_movement(q_opt[:, np.newaxis])
    viz.exec()


def main():
    model_path_kinova = Models.KINOVA_RIGHT_SLIDE.value
    model_path_upperlimb = Models.WU_INVERSE_KINEMATICS.value

    model_mx_kinova = biorbd.Model(model_path_kinova)
    model_mx_upperlimb = biorbd.Model(model_path_upperlimb)

    model_eigen_kinova = biorbd_eigen.Model(model_path_kinova)
    model_eigen_upperlimb = biorbd_eigen.Model(model_path_upperlimb)

    # get position and orientation of the forearm
    q_upper_limb = np.array([0.0, 0.5, 0.45, np.pi/2, 0.0, 0.0,  # floating base
                             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             1.8, 0.0,  # elbow and radioulnar joints
                             ])
    # ulna_idx = segment_index(model_mx_upperlimb, "ulna")
    # ulna_idx = segment_index(model_mx_upperlimb, "ulna_elbow_flexion")
    ulna_idx = segment_index(model_mx_upperlimb, "ulna_rotation2")
    rt = model_eigen_upperlimb.globalJCS(q_upper_limb, ulna_idx)
    r_offset = biorbd_eigen.Rotation.fromEulerAngles(rot=np.array([-np.pi/2, 0, 0]), seq="xyz").to_array()
    # rotation = rt.rot().to_array() @ r_offset
    rotation = r_offset @ rt.rot().to_array()
    position = rt.trans().to_array()

    # set the forearm radius
    radius = 0.035

    q_sym = MX.sym("q", model_eigen_kinova.nbQ(), 1)
    q_init = [0] * model_eigen_kinova.nbQ()
    q_up = get_range_q(model_eigen_kinova)[1]
    q_low = get_range_q(model_eigen_kinova)[0]

    objective = 0
    objective += penalty_rotation_matrix_cas(model_mx_kinova, q_sym, rotation)
    objective += 100 * penalty_position_cas(model_mx_kinova, q_sym, position)
    # penality for the z axis of the support

    constraints = []
    lbg = []
    ubg = []
    origin_kinova = np.array([0.0, 0.0])
    radius_three_bar = np.linalg.norm([-0.08842, -0.02369]) + 2 * 0.120 - np.linalg.norm(origin_kinova)
    # it should stay in the circle defined by the radius of the three bar mechanism
    constraints += [q_sym[0] ** 2 + q_sym[1] ** 2 - radius_three_bar ** 2]
    lbg += [-np.inf]
    ubg += [0]
    # it should not go under the table
    constraints += [model_mx_kinova.globalJCS(q_sym, model_mx_kinova.nbSegment() - 1).trans().to_mx()[2]]
    lbg += [0]
    ubg += [np.inf]

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
    q_opt = sol["x"].full().flatten()

    # position of the end effector
    print("position of the end effector:")
    print(model_eigen_kinova.globalJCS(q_opt, model_eigen_kinova.nbSegment() - 1).trans().to_array())
    print("expected position of the end effector:")
    print(position)
    print("rotation matrix of the end effector:")
    print(model_eigen_kinova.globalJCS(q_opt, model_eigen_kinova.nbSegment() - 1).rot().to_array())
    print("expected rotation matrix of the end effector:")
    print(rt.rot().to_array())


    parent_path = Path(model_path_kinova).parent
    output_path = str(parent_path / "merged.bioMod")
    merge_biomod(model_path_kinova, model_path_upperlimb, output_path)
    q_tot = np.concatenate((q_opt, q_upper_limb))

    viz = bioviz.Viz(output_path, show_floor=False, show_global_ref_frame=False)
    viz.load_movement(q_tot[:, np.newaxis])
    viz.exec()


if __name__ == "__main__":
    main()
