"""
converged, like this!
"""

import biorbd_casadi as biorbd
import biorbd as biorbd_eigen
import numpy as np
from casadi import MX, vertcat
from bioptim import (
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
import IK_Kinova


def prepare_ocp(
        biorbd_model_path: str = "KINOVA_arm_reverse.bioMod",
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

    biorbd_model = biorbd.Model(biorbd_model_path),
    nbQ = biorbd_model[0].nbQ()

    n_shooting = 30,
    final_time = 0.5,

    tau_min, tau_max, tau_init = -40, 40, 0

    # dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, with_contact=False)
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=20, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=range(10, 12), weight=1, phase=0)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=0.1, phase=0) no
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_QDDOT, weight=0.1, phase=0) no
    # objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, weight=10000, first_marker="md0",
    #  second_marker="mg2", node=Node.END) in constraint
    # objective_functions.add(ObjectiveFcn.Mayer.TRACK_STATE, key="q", weight=10,
    #                         target=q0[:3], index=range(0, 3), node=Node.START) redundant
    # objective_functions.add(ObjectiveFcn.Mayer.TRACK_STATE, key="q", weight=10,
    #                         target=qfin[:3], index=range(0, 3), node=Node.END) redundant
    # objective_functions.add(ObjectiveFcn.Lagrange.PROPORTIONAL_STATE, key="q", weight=0.1,
    #                         first_dof=10, second_dof=11, coef=1) no
    # objective_functions.add(ObjectiveFcn.Lagrange.PROPORTIONAL_STATE, key="qdot", weight=0.1,
    #                         first_dof=10, second_dof=11, coef=1) no
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, derivative=True, key="tau", weight=2, phase=0) no

    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))

    x_bounds[0][nbQ:, 0] = 0  # 0 velocity at the beginning and the end to the phase
    x_bounds[0][nbQ:-2, -1] = 0

    u_bounds = BoundsList()
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(),
                 [tau_max] * biorbd_model[0].nbGeneralizedTorque())

    u_bounds[0][6:, :] = 0

    x_init = InitialGuessList()
    x_init.add(q0.tolist() + [0] * biorbd_model[0].nbQdot())  # [0] * (nbQ+ nbQdot)

    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())

    # Constraints
    constraints = ConstraintList()

    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="mg1", second_marker="md0")
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="mg2", second_marker="md0")

    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="grd_contact1",
                    second_marker="Contact_mk1")
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.ALL, first_marker="grd_contact2",
                    second_marker="Contact_mk2")

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
    model = "KINOVA_arm_reverse.bioMod"
    q0 = np.array((0.0, 0.0, 0.0, 0.0, -0.1709, 0.0515, -0.2892, 0.6695, 0.721, 0.0, 0.0, 0.0))

    m = biorbd_eigen.Model(model)
    X = m.markers()
    targetd = X[2].to_array()  # 0 0 0 for now
    targetp_init = X[4].to_array()
    targetp_fin = X[5].to_array()

    pos_init = IK_Kinova.IK_Kinova(model, q0, targetd, targetp_init)
    pos_fin = IK_Kinova.IK_Kinova(model, pos_init, targetd, targetp_fin)

    ocp = prepare_ocp(model, pos_init, pos_fin)
    ocp.print(to_console=False, to_graph=True)
    # Custom plots
    ocp.add_plot_penalty(CostType.ALL)

    # --- Solve the program --- #
    show_options = dict(show_bounds=True)
    solver_options = {
        "ipopt.tol": 1e-6,
        "ipopt.max_iter": 2000,
        "ipopt.hessian_approximation": "exact",  # "exact", "limited-memory"
        "ipopt.limited_memory_max_history": 50,
        "ipopt.linear_solver": "mumps",  # "ma57", "ma86", "mumps"
    }

    sol = ocp.solve(show_online_optim=True, show_options=show_options, solver_options=solver_options)

    # --- Show results --- #
    sol.print()
    ocp.save(sol, "Kinova.bo")
    sol.animate()
    # sol.graphs()
