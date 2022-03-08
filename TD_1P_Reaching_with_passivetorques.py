"""
Not converging
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
    PlotType,
)
import IK_Kinova
import CustomDynamics
import spring

def prepare_ocp(
        biorbd_model_path: str = "KINOVA_arm_reverse.bioMod",
        q0: np.ndarray = np.zeros((12, 1)),
        qfin: np.ndarray = np.zeros((12, 1)),
        springs: list = [],
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    q0:
    qfin:
    springs:
    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path),
    nbQ = biorbd_model[0].nbQ()

    n_shooting = 30,
    final_time = 0.5,

    tau_min, tau_max, tau_init = -30, 30, 0

    dynamics = Dynamics(CustomDynamics.dynamic_config, with_contact=True,
                        dynamic_function=CustomDynamics.custom_dynamic,
                        springs=springs)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=2, phase=0, index=6)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=5, phase=0, index=range(0, 9))
    # objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, weight=1000, first_marker="md0",
    #                         second_marker="mg2", node=Node.END)
    # objective_functions.add(ObjectiveFcn.Mayer.TRACK_STATE, key="q", weight=10,
    #                         target=qfin[:-3], index=range(0, 9), node=Node.END)
    # objective_functions.add(ObjectiveFcn.Lagrange.PROPORTIONAL_STATE, key="q", weight=0.1,
    #                         first_dof=10, second_dof=11, coef=1)
    # objective_functions.add(ObjectiveFcn.Lagrange.PROPORTIONAL_STATE, key="qdot", weight=0.1,
    #                         first_dof=10, second_dof=11, coef=1)

    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))

    x_bounds[0][nbQ:, 0] = 0  # 0 velocity at the beginning and the end to the phase
    x_bounds[0][nbQ:-3, -1] = 0
    x_bounds[0].min[-3:, -2:] = -10 * 2 * np.pi
    x_bounds[0].max[-3:, -2:] = 10 * 2 * np.pi

    u_bounds = BoundsList()
    u_bounds.add([tau_min] * nbQ,
                 [tau_max] * nbQ)

    u_bounds[0][6:, :] = 0

    x_init = InitialGuessList()
    x_init.add(q0.tolist() + [0] * nbQ)  # [0] * (nbQ+ nbQdot)

    u_init = InitialGuessList()
    u_init.add([tau_init] * nbQ)

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

    # Define passive torque parameters
    springs = [{}] * m.nbQ()
    springParam = dict(s=-1, k1=-1, k2=10, q0=0)
    springs[10] = spring.assignParam(springParam)
    springs[11] = spring.assignParam(springParam)
    springs[8] = spring.linearSpring(6380, 0.15, 0.12506, 1.16056, 0.07844, 5*np.pi/6)

    ocp = prepare_ocp(model, pos_init, pos_fin, springs)
    ocp.print(to_console=False, to_graph=True)
    # Custom plots
    ocp.add_plot_penalty(CostType.ALL)
    NP = "PassiveTorquePlot"
    ocp.add_plot(NP, lambda t, x, u, p: CustomDynamics.plot_passive_torque(x, u, p, ocp.nlp[0], springs, [8, 10, 11]),
                 plot_type=PlotType.INTEGRATED, axes_idx=[0, 1, 2])
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
