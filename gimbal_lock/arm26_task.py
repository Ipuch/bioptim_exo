import biorbd_casadi as biorbd
import bioviz
from casadi import MX, Function
from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    QAndQDotBounds,
    OdeSolver,
    Dynamics,
    InitialGuess,
    Bounds,
    Solver,
    CostType,
    InterpolationType,
    Node,
)
import os
import numpy as np
from datetime import datetime
import get_markers


def eul2quat(eul: np.ndarray) -> np.ndarray:
    """
    Converts Euler angles to quaternion. It assumes a sequence angle of XYZ

    Parameters
    ----------
    eul: np.ndarray
        The 3 angles of sequence XYZ

    Returns
    -------
    The quaternion associated to the Euler angles in the format [W, X, Y, Z]
    """
    eul_sym = MX.sym("eul", 3)
    Quat = Function("Quaternion_fromEulerAngles", [eul_sym], [biorbd.Quaternion_fromXYZAngles(eul_sym).to_mx()])(eul)
    return Quat


def prepare_ocp(
        biorbd_model_path: str,
        n_shooting: int,
        target: any,
        x_init_ref: np.array,
        u_init_ref: np.array,
        ode_solver: OdeSolver = OdeSolver.RK4(),
        use_sx: bool = False,
        n_threads: int = 4,
        phase_time: float = 1,
) -> object:
    biorbd_model = biorbd.Model(biorbd_model_path)
    nb_q = biorbd_model.nbQ()

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1500)
    # objective_functions.add(
    #     ObjectiveFcn.Mayer.TRACK_STATE, key="q", node=Node.START, target=x_init_ref[:5, 0], weight=100000
    # )
    # objective_functions.add(
    #     ObjectiveFcn.Mayer.TRACK_STATE, key="q", node=Node.END, target=x_init_ref[:5, -1], weight=100000
    # )
    # objective_functions.add(
    #     ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1)
    # objective_functions.add(
    #     ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[0, 1, 2, 3, 4], weight=1000)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", node=Node.START, weight=1000)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", node=Node.END, weight=50)
    # objective_functions.add(
    #     ObjectiveFcn.Mayer.TRACK_MARKERS,
    #     target=target,
    #     weight=10000,
    #     node=Node.ALL
    #     )

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # initial guesses
    x_init = InitialGuess(x_init_ref, interpolation=InterpolationType.LINEAR)
    u_init = InitialGuess(u_init_ref)

    # Define control path constraint
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[:5, 0] = x_init_ref[:, 0][:5]
    x_bounds[:5, -1] = x_init_ref[:, 1][:5]
    x_bounds[5:, 0] = [1e-5] * biorbd_model.nbQdot()
    x_bounds[5:, -1] = [1e-5] * biorbd_model.nbQdot()
    x_bounds.min[3, 1] = 0
    x_bounds.max[3, 1] = np.pi
    x_bounds.min[4, 1] = -1
    x_bounds.max[4, 1] = 1

    n_tau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max = -20, 20
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)

    return OptimalControlProgram(
        biorbd_model_path,
        dynamics,
        n_shooting,
        phase_time=phase_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        ode_solver=ode_solver,
        n_threads=n_threads,
        use_sx=use_sx,
    )


def main():
    """
    Get data, then create a tracking problem, and finally solve it and plot some relevant information
    """

    # Define the problem
    c3d_path = "arm26.c3d"
    # todo: manger, dessiner
    n_shooting_points = 25
    nb_iteration = 10000

    model_path = "/home/lim/Documents/Stage_Thasaarah/bioptim_exo/models/arm26_quat.bioMod"
    biorbd_model = biorbd.Model(model_path)
    marker_ref = [m.to_string() for m in biorbd_model.markerNames()]


#############################################################

# import casadi as cas
#
# Quaternion = cas.MX.zeros(4)
# Quaternion[0] = 1
# Quaternion_biorbd = biorbd.Quaternion(Quaternion[0], Quaternion[1], Quaternion[2], Quaternion[3])
# Rotation_matrix = biorbd.Quaternion.toMatrix(Quaternion_biorbd)
# biorbd.Rotation_toEulerAngles(Rotation_matrix, 'xyz').to_mx()

#############################################################

    x_init_ref = np.array(
        [[0, -1.4], [0, -2], [0, 1], [0, 0.7], [0, 0], [0, 0], [0, 0],
         [0, 0]])
    x_init_quat = x = np.vstack((np.zeros((biorbd_model.nbQ(), 2)), np.ones((biorbd_model.nbQdot(), 2)) / 100000))
    for i in range(2):
        x_quat = eul2quat(x_init_ref[:3])
        x_init_quat[:3, i] = x_quat[:, i].toarray()[1:, 0]
        x_init_quat[4, i] = x_quat[:, i].toarray()[0, 0]
    x_init_quat[3] = x_init_ref[3]

    u_init = [0] * biorbd_model.nbGeneralizedTorque()

    x = np.zeros((2, 5))
    x[0] = x_init_quat[:5, 0]
    x[1] = x_init_quat[:5, 1]
    target = get_markers.marker_position(x, n_shooting_points)
    phase_time = 1.5

    # optimal control program
    my_ocp = prepare_ocp(
        biorbd_model_path=model_path,
        x_init_ref=x_init_quat,
        u_init_ref=u_init,
        target=target,
        n_shooting=n_shooting_points,
        phase_time=phase_time,
        use_sx=False,
        n_threads=4,
    )

    # my_ocp.print()

    # add figures of constraints and objectives
    my_ocp.add_plot_penalty(CostType.ALL)

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver.set_linear_solver("ma57")
    solver.set_maximum_iterations(nb_iteration)
    sol = my_ocp.solve(solver)
    sol.print_cost()
    norm = []
    for i in range(n_shooting_points):
        norm.append(
            np.sqrt(
                sol.states["q"][0, i] ** 2
                + sol.states["q"][1, i] ** 2
                + sol.states["q"][2, i] ** 2
                + sol.states["q"][4, i] ** 2
            )
        )
    print(norm)

    # --- Save --- #
    c3d_str = c3d_path.split("/")
    c3d_name = os.path.splitext(c3d_str[-1])[0]
    save_path = f"save/{c3d_name}_{datetime.now()}"
    save_path = save_path.replace(" ", "_").replace("-", "_").replace(":", "_").replace(".", "_")
    my_ocp.save(sol, save_path)

    # --- Plot --- #
    sol.graphs(show_bounds=True)
    # todo: animate first and last frame with markers
    # sol.animate(n_frames=n_shooting_points)
    # b = bioviz.Viz(model_path=model_path)
    # b.load_movement(sol.states["q"][:, 1:])
    # b.exec()


if __name__ == "__main__":
    main()
    # main("F0_tete_05.c3d")
