"""
converged, like this!
"""

import biorbd_casadi
import biorbd
import bioviz
import numpy as np
from bioptim import (
    Solver,
    PenaltyNode,
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    DynamicsList,
    Bounds,
    BoundsList,
    QAndQDotBounds,
    InitialGuess,
    InitialGuessList,
    ObjectiveFcn,
    Objective,
    ObjectiveList,
    OdeSolver,
    Node,
    ConstraintFcn,
    ConstraintList,
    DynamicsFunctions,
    NonLinearProgram,
    ConfigureProblem,
    CostType,
)
import IK_Kinova_2
import inverse_kinematics as ik
import time
import numpy as np
from ezc3d import c3d
import biorbd
from models.utils import add_header, thorax_variables
from utils import get_range_q


def prepare_ocp(
        biorbd_model_path: str = None,
        q0: np.ndarray = np.zeros((12, 1)),
        qfin: np.ndarray = np.zeros((12, 1)),
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    q0:
    qfin:
    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = (biorbd_casadi.Model(biorbd_model_path),)
    nbQ = biorbd_model[0].nbQ()

    n_shooting = (30,)
    final_time = (0.5,)

    tau_min, tau_max, tau_init = -40, 40, 0

    # dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=20, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=5, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[6, 9, 10, 11], weight=1, phase=0)

    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))

    x_bounds[0][nbQ:, 0] = 0  # 0 velocity at the beginning and the end to the phase
    x_bounds[0][nbQ:-3, -1] = 0

    u_bounds = BoundsList()
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque())

    u_bounds[0][6:, :] = 0

    x_init = InitialGuessList()
    x_init.add(q0.tolist() + [0] * nbQ)  # [0] * (nbQ+ nbQdot)

    u_init = InitialGuessList()
    u_init.add([tau_init] * nbQ)

    # Constraints
    constraints = ConstraintList()

    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="mg1", second_marker="md0")
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="mg2", second_marker="md0")

    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="grd_contact1", second_marker="Contact_mk1"
    )
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.ALL, first_marker="grd_contact2", second_marker="Contact_mk2"
    )

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=OdeSolver.RK4(),
        n_threads=8,
    )


if __name__ == "__main__":

    model_path_with_floating_base = "../models/KINOVA_merge_inverse_kinematics.bioMod"
    biorbd_model_with_floating_base = biorbd.Model(model_path_with_floating_base)

    c3d_path = "../data/F3_aisselle_01.c3d"

    c3d = c3d(c3d_path)

    points = c3d["data"]["points"]
    labels_markers = c3d["parameters"]["POINT"]["LABELS"]["value"]
    labels_markers.append('Table:Table6')

    marker_names_with_floating_base = [
        biorbd_model_with_floating_base.markerNames()[i].to_string() for i in range(biorbd_model_with_floating_base.nbMarkers())
    ]
    markers_without_kinova = np.zeros((3, len(marker_names_with_floating_base), len(points[0, 0, :])))

    for i, name in enumerate(marker_names_with_floating_base):
        if name in labels_markers:
            if name == 'Table:Table6':
                markers_without_kinova[:, i, :] = points[:3, labels_markers.index('Table:Table5'), :] / 1000
            else:
                markers_without_kinova[:, i, :] = points[:3, labels_markers.index(name), :] / 1000

    markers_without_kinova[2, marker_names_with_floating_base.index('Table:Table6'), :] = markers_without_kinova[2, marker_names_with_floating_base.index('Table:Table6'), :] + 0.1

    my_ik = ik.InverseKinematics(model_path_with_floating_base, markers_without_kinova)
    my_ik.solve("lm")

    # my_ik.animate()

    thorax_values = {
        "thoraxRT1": my_ik.q[3, :].mean(),
        "thoraxRT2": my_ik.q[4, :].mean(),
        "thoraxRT3": my_ik.q[5, :].mean(),
        "thoraxRT4": my_ik.q[0, :].mean(),
        "thoraxRT5": my_ik.q[1, :].mean(),
        "thoraxRT6": my_ik.q[2, :].mean(),
    }
    old_biomod_file = (
        "../models/KINOVA_merge_without_floating_base_template.bioMod"
    )
    new_biomod_file = (
        "../models/KINOVA_merge_without_floating_base_template_with_variables.bioMod"
    )
    add_header(old_biomod_file, new_biomod_file, thorax_values)
    model_path = new_biomod_file
    biorbd_model = biorbd.Model(model_path)

    markers_names = [value.to_string() for value in biorbd_model.markerNames()]
    markers = np.zeros((3, len(markers_names), len(points[0, 0, :])))

    for i, name in enumerate(markers_names):
        if name in labels_markers:
            if name == 'Table:Table6':
                markers[:, i, :] = points[:3, labels_markers.index('Table:Table5'), :] / 1000
            else:
                markers[:, i, :] = points[:3, labels_markers.index(name), :] / 1000

    markers[2, markers_names.index('Table:Table6'), :] = markers[2, markers_names.index('Table:Table6'), :] + 0.1
    xp_data = markers[:, :, :100]

    new_q = np.zeros((biorbd_model.nbQ(), markers.shape[2]))
    new_q[:10, :] = my_ik.q[6:16, :]
    new_q[16:, :] = my_ik.q[16:, :]

    q0 = new_q[:, 0]

    pos_init = IK_Kinova_2.IK_Kinova(biorbd_model, markers_names, xp_data, q0, new_q[:, :100])

    b = bioviz.Viz(loaded_model=biorbd_model, show_muscles=False, show_floor=False)
    b.load_experimental_markers(xp_data)
    # b.load_movement(np.array(q0, q0).T)
    b.load_movement(pos_init)

    b.exec()

    # ocp = prepare_ocp(model, pos_init, pos_fin)
    # ocp.print(to_console=False, to_graph=True)
    # # Custom plots
    # ocp.add_plot_penalty(CostType.ALL)
    #
    # # --- Solve the program --- #
    # show_options = dict(show_bounds=True)
    # solver_options = {
    #     "ipopt.tol": 1e-6,
    #     "ipopt.max_iter": 2000,
    #     "ipopt.hessian_approximation": "exact",  # "exact", "limited-memory"
    #     "ipopt.limited_memory_max_history": 50,
    #     "ipopt.linear_solver": "mumps",  # "ma57", "ma86", "mumps"
    # }
    #
    # sol = ocp.solve()
    # sol.animate()
    #
    # # --- Show results --- #
    # sol.print_cost()
    # # ocp.save(sol, "Kinova.bo")
    # sol.animate()
    # # sol.graphs()
