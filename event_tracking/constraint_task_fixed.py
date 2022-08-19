import biorbd_casadi as biorbd
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
from data.enums import Tasks
from datetime import datetime
import models.utils as utils
from models.enums import Models
import data.load_events as load_events
import tracking.load_experimental_data as load_experimental_data



def prepare_ocp(
    biorbd_model_path: str,
    c3d_path: str,
    n_shooting: int,
    x_init_ref: np.array,
    u_init_ref: np.array,
    target: any,
    ode_solver: OdeSolver = OdeSolver.RK4(),
    use_sx: bool = False,
    n_threads: int = 4,
    phase_time: float = 1,
) -> object:
    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, derivative=True)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1000)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=50)
    # objective_functions.add(
    #     ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[0, 1, 2, 3, 4], weight=1000)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", node=Node.START, weight=50)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", node=Node.END, weight=50)
    # objective_functions.add(
    #     ObjectiveFcn.Mayer.TRACK_MARKERS,
    #     weight=10000, marker_index=[10, 12, 13, 14, 15],
    #     target=target,
    #     node=Node.ALL
    # )

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # initial guesses
    x_init = InitialGuess(
        np.concatenate([x_init_ref[5:10], x_init_ref[15:] * 0]), interpolation=InterpolationType.EACH_FRAME
    )

    u_init = InitialGuess(u_init_ref[5:10] * 0, interpolation=InterpolationType.EACH_FRAME)

    # Define control path constraint
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[:5, 0] = x_init_ref[:5, 0]
    x_bounds[:5, -1] = x_init_ref[:5, -1]
    # x_bounds[5:, 0] = [0] * biorbd_model.nbQdot()
    # x_bounds.min[8:, 0] = [-np.pi/4] * biorbd_model.nbQdot()
    # x_bounds.max[8:, 0] = [np.pi/4] * biorbd_model.nbQdot()
    x_bounds.min[5:, -1] = [-np.pi / 2] * biorbd_model.nbQdot()
    x_bounds.max[5:, -1] = [np.pi / 2] * biorbd_model.nbQdot()

    n_tau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max = -20, 20
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)

    return OptimalControlProgram(
        biorbd_model,
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


def main(task):
    """
    Get data, then create a tracking problem, and finally solve it and plot some relevant information
    """

    # Define the problem
    # c3d_path = "F0_dents_05.c3d"
    n_shooting_points = 800
    nb_iteration = 10000

    c3d_path = task.value
    data_path = c3d_path.removesuffix(c3d_path.split("/")[-1])
    file_path = data_path + Models.WU_INVERSE_KINEMATICS.name + "_" + c3d_path.split("/")[-1].removesuffix(".c3d")
    q_file_path = file_path + "_q.txt"
    qdot_file_path = file_path + "_qdot.txt"

    thorax_values = utils.thorax_variables(q_file_path)  # load c3d floating base pose
    model_template_path = Models.WU_WITHOUT_FLOATING_BASE_FIXED_TEMPLATE.value
    new_biomod_file = model_template_path.removesuffix(".bioMod") + "_with_variables.bioMod"
    utils.add_header(model_template_path, new_biomod_file, thorax_values)

    biorbd_model = biorbd.Model(new_biomod_file)
    marker_ref = [m.to_string() for m in biorbd_model.markerNames()]

    # get key events
    event = load_events.LoadEvent(c3d_path=c3d_path, marker_list=marker_ref)
    data = load_experimental_data.LoadData(biorbd_model, c3d_path, q_file_path, qdot_file_path)
    start_frame = event.get_frame(0)
    end_frame = event.get_frame(1)
    phase_time = event.get_time(1) - event.get_time(0)
    target = data.get_marker_ref(
        number_shooting_points=[n_shooting_points],
        phase_time=[phase_time],
        markers_names=["ULNA", "RADIUS", "SEML", "MET2", "MET5"],
        start=start_frame,
        end=end_frame,
    )


    # load initial guesses
    q_ref, qdot_ref, tau_ref = data.get_variables_ref(
        number_shooting_points=[n_shooting_points],
        phase_time=[phase_time],
        start=start_frame,
        end=end_frame,
    )
    x_init_ref = np.concatenate([q_ref[0][6:, :], qdot_ref[0][6:, :]])  # without floating base
    u_init_ref = tau_ref[0][6:, :]

    # optimal control program
    my_ocp = prepare_ocp(
        biorbd_model_path=new_biomod_file,
        c3d_path=c3d_path,
        x_init_ref=x_init_ref,
        u_init_ref=u_init_ref,
        target=target[0],
        n_shooting=n_shooting_points,
        use_sx=False,
        n_threads=4,
        phase_time=phase_time,
    )

    # my_ocp.print()

    # add figures of constraints and objectives
    my_ocp.add_plot_penalty(CostType.ALL)

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver.set_linear_solver("ma57")
    solver.set_maximum_iterations(nb_iteration)
    sol = my_ocp.solve(solver)
    # sol.print_cost()

    # --- Save --- #
    c3d_str = c3d_path.split("/")
    c3d_name = os.path.splitext(c3d_str[-1])[0]
    save_path = f"save/{c3d_name}_{datetime.now()}"
    save_path = save_path.replace(" ", "_").replace("-", "_").replace(":", "_").replace(".", "_")
    my_ocp.save(sol, save_path)

    # --- Plot --- #
    # sol.graphs(show_bounds=True)
    # todo: animate first and last frame with markers
    # sol.animate(n_frames=100)


if __name__ == "__main__":
    main(Tasks.TEETH)
