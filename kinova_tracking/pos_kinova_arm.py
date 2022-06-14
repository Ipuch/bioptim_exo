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
import IK_Kinova
from ezc3d import c3d


def prepare_ocp(
    biorbd_model_path: str = "KINOVA_arm_reverse_left.bioMod",
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
    #todo: lock the floating base
    #todo: create kinova merged with variables
    model = "../models/KINOVA_merge.bioMod"
    # model = "../models/KINOVA_arm_reverse_right.bioMod"
    c3d_path = "../data/kinova_arm/F3_aisselle_01.c3d"

    biorbd_model = biorbd.Model(model)
    c3d = c3d(c3d_path)

    # q0 = np.array((0.0, 0.0, 0.0, 0.0, -0.1709, 0.0515, -0.2892, 0.6695, 0.721, 0.0, 0.0, 0.0))
    q0 = np.array((-0.49936691256448906,
                   0.6129779697515374,
                   0.37513329009911695,
                   -1.588636623158031,
                   3.1415926535897927,
                   -3.0956382757004506,
                   -0.1828863713271869,
                   -0.18554581848601184,
                   3.1415926535897927,
                   -3.1415926535897927,
                   3.1415926535897913,
                   0.4561403964243357,
                   0.6148957681121301,
                   -0.1053897314454207,
                   1.1118454130974944,
                   1.0199120908159471,
                   -0.3535,
                   0.3739,
                   0.4524,
                   -1.879,
                   0.1111,
                   0.2817))

    markers_names = [value.to_string() for value in biorbd_model.markerNames()]
    markers_list = biorbd_model.markers()

    points = c3d["data"]["points"]
    labels_markers = c3d["parameters"]["POINT"]["LABELS"]["value"]

    targetd = markers_list[markers_names.index('grd_contact1')].to_array()  # 0 0 0 for now
    targetp_init = markers_list[markers_names.index('mg1')].to_array()
    targetp_fin = markers_list[markers_names.index('mg2')].to_array()

    pos_init = IK_Kinova.IK_Kinova(biorbd_model, markers_names, q0, targetd, targetp_init)
    pos_fin = IK_Kinova.IK_Kinova(biorbd_model, markers_names, pos_init, targetd, targetp_fin)

    b = bioviz.Viz(loaded_model=biorbd_model, show_muscles=False, show_floor=False)
    q = [pos_init, pos_fin]
    b.load_movement(np.array((pos_init, pos_fin)).T)

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
