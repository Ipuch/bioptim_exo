import sys
import biorbd_casadi as biorbd

sys.path.append("../event")
import ezc3d
from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    QAndQDotBounds,
    OdeSolver,
    Node,
    Dynamics,
    InitialGuess,
    Bounds,
    Solver,
    CostType,
    InterpolationType,
)
import numpy as np
import os
from datetime import datetime

sys.path.append("../models")
import utils

sys.path.append("../data")
import load_events

sys.path.append("../tracking")
import load_experimental_data


def prepare_ocp(
    biorbd_model_path: str,
    c3d_file_path: str,
    n_shooting: int,
    x_init_ref: np.array,
    u_init_ref: np.array,
    target_start: any,
    target_end: any,
    # LoadEvent: object = LoadEvent(c3d_path="F0_tete_05.c3d"),
    ode_solver: OdeSolver = OdeSolver.RK4(),
    use_sx: bool = False,
    n_threads: int = 4,
) -> object:
    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1000)  # 100
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, weight=1, key="qdot")
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[0, 1], weight=10)

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = QAndQDotBounds(biorbd_model)

    x_init = InitialGuess(x_init_ref, interpolation=InterpolationType.EACH_FRAME)
    u_init = InitialGuess(u_init_ref, interpolation=InterpolationType.EACH_FRAME)

    Q = biorbd.GeneralizedCoordinates(biorbd_model)
    Qdot = biorbd.GeneralizedVelocity(biorbd_model)
    Qddot = biorbd.GeneralizedAcceleration(biorbd_model)



    # Define control path constraint
    n_tau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max = -100, 100
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    # u_bounds[1, :] = 0

    u_init = InitialGuess([0] * n_tau)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        phase_time=1,
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
    # with_floating_base = False

    c3d_path = "F0_dents_05.c3d"
    # todo: manger, aisselle, dessiner
    c3d = ezc3d.c3d(c3d_path)
    freq = c3d["parameters"]["POINT"]["RATE"]["value"][0]
    nb_frames = c3d["parameters"]["POINT"]["FRAMES"]["value"][0]
    duration = nb_frames / freq
    # n_shooting_points = int(duration * 100)
    # n_shooting_points = 100
    nb_iteration = 10000

    q_file_path = c3d_path.removesuffix(".c3d") + "_q.txt"
    qdot_file_path = c3d_path.removesuffix(".c3d") + "_qdot.txt"

    thorax_values = utils.thorax_variables(q_file_path)  # load c3d floating base pose
    new_biomod_file = "../models/wu_converted_definitif_without_floating_base_template_with_variables.bioMod"
    model_path_without_floating_base = "../models/wu_converted_definitif_without_floating_base_template.bioMod"
    utils.add_header(model_path_without_floating_base, new_biomod_file, thorax_values)

    biorbd_model = biorbd.Model(new_biomod_file)
    marker_ref = [m.to_string() for m in biorbd_model.markerNames()]

    event = load_events.LoadEvent(c3d_path=c3d_path, marker_list=marker_ref)

    target_start = event.get_markers(0)[:, :, np.newaxis]
    target_end = event.get_markers(1)[:, :, np.newaxis]

    data = load_experimental_data.LoadData(biorbd_model, c3d_path, q_file_path, qdot_file_path)
    start_frame = event.get_frame(0)
    end_frame = event.get_frame(1)

    n_shooting_points = 100
    phase_time = 1
    tau = data.tau
    q_ref, qdot_ref, tau_ref = data.get_states_ref(
        number_shooting_points=[n_shooting_points],
        phase_time=[phase_time],
        start=start_frame,
        end=end_frame,
    )
    x_init_ref = np.concatenate([q_ref[0][6:], qdot_ref[0][6:]]) # without floating base
    u_init_ref = tau_ref[0][6:]
    my_ocp = prepare_ocp(
        biorbd_model_path=new_biomod_file,
        c3d_file_path=c3d_path,
        x_init_ref=x_init_ref,
        u_init_ref=u_init_ref,
        target_start=target_start,
        target_end=target_end,
        n_shooting=n_shooting_points,
        ode_solver=OdeSolver.RK4(),
        use_sx=False,
        n_threads=4,
    )
    my_ocp.add_plot_penalty(CostType.ALL)
    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver.set_linear_solver("ma57")
    solver.set_maximum_iterations(nb_iteration)
    sol = my_ocp.solve(solver)
    sol.print_cost()
    # sol.print_cost(

    # --- Save --- #
    c3d_str = c3d_path.split("/")
    c3d_name = os.path.splitext(c3d_str[-1])[0]
    save_path = f"save/{c3d_name}_{datetime.now()}"
    save_path = save_path.replace(" ", "_").replace("-", "_").replace(":", "_").replace(".", "_")
    my_ocp.save(sol, save_path)

    # --- Plot --- #
    # sol.graphs(show_bounds=True)
    # todo: animate first and last frame with markers
    sol.animate(n_frames=100)


if __name__ == "__main__":
    main()
