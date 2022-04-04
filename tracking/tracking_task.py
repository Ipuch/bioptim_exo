from load_experimental_data import LoadData, C3dData
from scipy.integrate import solve_ivp
import numpy as np
import biorbd_casadi as biorbd
from casadi import MX, vertcat
from matplotlib import pyplot as plt

from bioptim import (
    OptimalControlProgram,
    NonLinearProgram,
    BiMapping,
    DynamicsList,
    DynamicsFcn,
    DynamicsFunctions,
    ObjectiveList,
    ObjectiveFcn,
    BoundsList,
    Bounds,
    QAndQDotBounds,
    InitialGuessList,
    OdeSolver,
    Node,
    Solver,
    CostType,
)


def prepare_ocp(
    biorbd_model: biorbd.Model,
    final_time: float,
    n_shooting: int,
    markers_ref: np.ndarray,
    ode_solver: OdeSolver = OdeSolver.RK4(),
) -> OptimalControlProgram:
    """
    Prepare the ocp to solve

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The loaded biorbd model
    final_time: float
        The time at final node
    n_shooting: int
        The number of shooting points
    markers_ref: np.ndarray
        The marker to track if 'markers' is chosen in kin_data_to_track
    ode_solver: OdeSolver
        The ode solver to use

    Returns
    -------
    The OptimalControlProgram ready to solve
    """

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10)
    objective_functions.add(ObjectiveFcn.Mayer.TRACK_MARKERS, weight=100, target=markers_ref, node=Node.ALL)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model))

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))

    # Define control path constraint
    u_bounds = BoundsList()
    u_init = InitialGuessList()
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds.add(
        [tau_min] * biorbd_model.nbGeneralizedTorque(), [tau_max] * biorbd_model.nbGeneralizedTorque(),
    )
    u_init.add([tau_init] * biorbd_model.nbGeneralizedTorque())

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        ode_solver=ode_solver,
    )


def main():
    """
    Get data, then create a tracking problem, and finally solve it and plot some relevant information
    """

    # Define the problem
    c3d_path = "../data/F0_dessiner_05.c3d"
    model_path = "../models/wu_converted_definitif.bioMod"

    biorbd_model = biorbd.Model(model_path)
    data = C3dData(c3d_path)
    final_time = data.get_final_time()
    n_shooting_points = 30

    # Marker ref
    data_loaded = LoadData(biorbd_model, c3d_path)
    markers_ref = data_loaded.get_marker_ref(nb_shooting=[n_shooting_points], phase_time=[final_time])

    ocp = prepare_ocp(biorbd_model, final_time, n_shooting_points, markers_ref[0])

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    # --- Show the results --- #
    q = sol.states["q"]
    n_q = ocp.nlp[0].model.nbQ()
    n_mark = ocp.nlp[0].model.nbMarkers()
    n_frames = q.shape[1]

    markers = np.ndarray((3, n_mark, q.shape[1]))
    symbolic_states = MX.sym("x", n_q, 1)
    markers_func = biorbd.to_casadi_func("ForwardKin", biorbd_model.markers, symbolic_states)
    for i in range(n_frames):
        markers[:, :, i] = markers_func(q[:, i])

    plt.figure("Markers")
    n_steps_ode = ocp.nlp[0].ode_solver.steps + 1 if ocp.nlp[0].ode_solver.is_direct_collocation else 1
    for i in range(markers.shape[1]):
        plt.plot(np.linspace(0, 2, n_shooting_points + 1), markers_ref[:, i, :].T, "k")
        plt.plot(np.linspace(0, 2, n_shooting_points * n_steps_ode + 1), markers[:, i, :].T, "r--")
    plt.xlabel("Time")
    plt.ylabel("Markers Position")

    # --- Plot --- #
    plt.show()
    sol.print()
    sol.graphs(show_bounds=True)
    sol.animate(n_frames=100)


if __name__ == "__main__":
    main()
