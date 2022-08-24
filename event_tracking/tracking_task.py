import biorbd_casadi as biorbd
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

import os
import numpy as np
from datetime import datetime
from data.enums import Tasks
import data.load_events as load_events
import models.utils as utils
from models.enums import Models
import tracking.load_experimental_data as load_experimental_data


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int,
    x_init_ref: np.array,
    target_start: any,
    target_end: any,
    ode_solver: OdeSolver = OdeSolver.RK4(),
    use_sx: bool = False,
    n_threads: int = 4,
) -> object:
    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1000)
    objective_functions.add(
        ObjectiveFcn.Mayer.TRACK_MARKERS,
        weight=1000,
        target=target_start,
        node=Node.START,
    )
    objective_functions.add(
        ObjectiveFcn.Mayer.TRACK_MARKERS,
        weight=1000,
        target=target_end,
        node=Node.END,
    )
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, weight=1, key="qdot")

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = QAndQDotBounds(biorbd_model)
    x_init = InitialGuess(x_init_ref, interpolation=InterpolationType.EACH_FRAME)

    # Define control path constraint
    n_tau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
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


def main(task):
    """
    Get data, then create a tracking problem, and finally solve it and plot some relevant information
    """

    # Define the problem
    with_floating_base = False
    nb_iteration = 1000
    n_shooting_points = 50
    phase_time = 1

    # Files used
    c3d_path = task.value
    data_path = c3d_path.removesuffix(c3d_path.split("/")[-1])
    file_path = data_path + Models.WU_INVERSE_KINEMATICS.name + "_" + c3d_path.split("/")[-1].removesuffix(".c3d")
    q_file_path = file_path + "_q.txt"
    qdot_file_path = file_path + "_qdot.txt"

    thorax_values = utils.thorax_variables(q_file_path)  # load c3d floating base pose
    model_template_path = Models.WU_WITHOUT_FLOATING_BASE_TEMPLATE.value
    new_biomod_file = model_template_path.removesuffix(".bioMod") + "with_variables.bioMod"
    utils.add_header(model_template_path, new_biomod_file, thorax_values)
    # Define event
    biorbd_model = biorbd.Model(new_biomod_file)
    marker_ref = [m.to_string() for m in biorbd_model.markerNames()]
    event = load_events.LoadEvent(c3d_path=c3d_path, marker_list=marker_ref)

    # Markers and frame at the beginning and the end of the first phase
    target_start = event.get_markers(0)[:, :, np.newaxis]
    target_end = event.get_markers(1)[:, :, np.newaxis]
    start_frame = event.get_frame(0)
    end_frame = event.get_frame(1)

    # Define initial states
    data = load_experimental_data.LoadData(biorbd_model, c3d_path, q_file_path, qdot_file_path)
    q_ref, qdot_ref = data.get_states_ref(
        number_shooting_points=[n_shooting_points],
        phase_time=[phase_time],
        start=start_frame,
        end=end_frame,
        with_floating_base=False,
    )
    x_init_ref = np.concatenate([q_ref[0], qdot_ref[0]])

    my_ocp = prepare_ocp(
        biorbd_model_path=new_biomod_file,
        x_init_ref=x_init_ref,
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

    # --- Save --- #
    c3d_str = c3d_path.split("/")
    c3d_name = os.path.splitext(c3d_str[-1])[0]
    save_path = f"save/{c3d_name}_{datetime.now()}"
    save_path = save_path.replace(" ", "_").replace("-", "_").replace(":", "_").replace(".", "_")
    my_ocp.save(sol, save_path)

    # --- Animate --- #
    sol.animate(n_frames=100)


if __name__ == "__main__":
    main(task=Tasks.TEETH)
