import biorbd_casadi as biorbd
from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    QAndQDotBounds,
    OdeSolver,
    DynamicsList,
    InitialGuessList,
    BoundsList,
    Solver,
    CostType,
    InterpolationType,
    Node,
)
import os
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from quat import eul2quat, quat2eul
from data.enums import Tasks
import data.load_events as load_events
import models.utils as utils
from models.enums import Models
import tracking.load_experimental_data as load_experimental_data


def prepare_ocp(
    biorbd_model_path: str,
    task: any,
    track_markers: bool,
    n_shooting: int,
    x_init_ref: np.array,
    u_init_ref: np.array,
    target: any,
    ode_solver: OdeSolver = OdeSolver.RK4(),
    use_sx: bool = False,
    n_threads: int = 4,
    phase_time: float = 1,
) -> object:
    biorbd_model = biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path)
    n_q = biorbd_model[0].nbQ()
    n_qdot = biorbd_model[0].nbQdot()
    n_tau = biorbd_model[0].nbGeneralizedTorque()
    tau_min, tau_max = -20, 20

    # initialise lists
    objective_functions = ObjectiveList()
    dynamics = DynamicsList()
    x_init = InitialGuessList()
    u_init = InitialGuessList()
    x_bounds = BoundsList()
    u_bounds = BoundsList()

    for i in range(2):
        # Add objective functions
        if task == Tasks.TEETH:
            print("dents!")
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=5, phase=i)
            objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1000, phase=i
            )
            objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[0, 1, 2], weight=1000, phase=i
            )
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[3, 4], weight=100, phase=i)
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=300, phase=i)
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=50, phase=i)

        elif task == Tasks.DRINK:
            print("boire!")
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=3, phase=i)
            objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1000, phase=i
            )
            objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[0, 1, 2], weight=1000, phase=i
            )
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[3, 4], weight=100, phase=i)
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=50, phase=i)
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1000, phase=i)
            if track_markers:
                print("tracking markers boire")
                objective_functions.add(
                    ObjectiveFcn.Mayer.TRACK_MARKERS,
                    weight=1000,  # weight=500,
                    target=target[i],
                    node=Node.ALL,
                    phase=i,
                )

        elif task == Tasks.HEAD:
            print("tete!")
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=5, phase=i)
            objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1000, phase=i
            )
            objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[0, 1, 2], weight=1000, phase=i
            )
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[3, 4], weight=100, phase=i)
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=50)
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1000, phase=i)
            if track_markers:
                objective_functions.add(
                    ObjectiveFcn.Mayer.TRACK_MARKERS,
                    weight=10000,
                    target=target[i],
                    node=Node.ALL,
                    phase=i,
                )

        elif task == Tasks.EAT:
            print("manger!")
            objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[0, 1, 2], weight=1000, phase=i
            )
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[3, 4], weight=100, phase=i)
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=300, phase=i)
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1.5, phase=i)
            objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1000, phase=i
            )
            objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1000, phase=i
            )
            if track_markers:
                print("tracking markers manger")
                objective_functions.add(
                    ObjectiveFcn.Mayer.TRACK_MARKERS,
                    weight=10000,
                    target=target[i],
                    node=Node.ALL,
                    phase=i,
                )

        elif task == Tasks.ARMPIT:
            print("aisselle!")
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=5, phase=i)
            objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1000, phase=i
            )
            objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[0, 1, 2, 3, 4], weight=1000, phase=i
            )
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=300, phase=i)
            objective_functions.add(
                ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", node=Node.START, derivative=True, weight=100, phase=i
            )
            if track_markers:
                print("tracking markers.")
                objective_functions.add(
                    ObjectiveFcn.Mayer.TRACK_MARKERS,
                    weight=1000,
                    marker_index=[10, 12, 13, 14, 15],
                    target=target[i],
                    node=Node.ALL,
                    phase=i,
                )

        # Dynamics
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

        # initial guesses
        x_init.add(x_init_ref[i], interpolation=InterpolationType.EACH_FRAME)
        u_init.add(u_init_ref[i], interpolation=InterpolationType.EACH_FRAME)

        # Define state bounds

        x_bounds_i = QAndQDotBounds(biorbd_model[i])
        x_bounds_i[:n_q, 0] = x_init_ref[i][:n_q, 0]
        x_bounds_i[:n_q, -1] = x_init_ref[i][:n_q, -1]
        if task == Tasks.ARMPIT:
            x_bounds_i.min[n_q:, 0] = [-1e-2] * n_qdot
            x_bounds_i.max[n_q:, 0] = [1e-2] * n_qdot
            x_bounds_i.min[n_q:, -1] = [-1e-2] * n_qdot
            x_bounds_i.max[n_q:, -1] = [1e-2] * n_qdot
        else:
            x_bounds_i.min[n_q:, 0] = [-1e-3] * n_qdot
            x_bounds_i.max[n_q:, 0] = [1e-3] * n_qdot
            x_bounds_i.min[n_q:, -1] = [-1e-3] * n_qdot
            x_bounds_i.max[n_q:, -1] = [1e-3] * n_qdot
        x_bounds_i.min[8:10, 1], x_bounds_i.min[10, 1] = x_bounds_i.min[9:11, 1], -1
        x_bounds_i.max[8:10, 1], x_bounds_i.max[10, 1] = x_bounds_i.max[9:11, 1], 1
        x_bounds.add(bounds=x_bounds_i)

        # define control bounds
        u_bounds.add([tau_min] * n_tau, [tau_max] * n_tau)

    return OptimalControlProgram(
        biorbd_model=biorbd_model,
        dynamics=dynamics,
        n_shooting=n_shooting,
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


def main(
    task: Tasks = None,
    track_markers: bool = False,
):
    """
    Get data, then create a tracking problem, and finally solve it and plot some relevant information
    """

    # Define the problem
    c3d_path = task.value
    # todo: manger, aisselle, dessiner
    n_shooting_points = 50
    nb_iteration = 10000

    data_path = c3d_path.removesuffix(c3d_path.split("/")[-1])
    file_path = data_path + Models.WU_INVERSE_KINEMATICS_XYZ.name + "_" + c3d_path.split("/")[-1].removesuffix(".c3d")
    q_file_path = file_path + "_q.txt"
    qdot_file_path = file_path + "_qdot.txt"

    thorax_values = utils.thorax_variables(q_file_path)  # load c3d floating base pose
    model_template_path = Models.WU_WITHOUT_FLOATING_BASE_QUAT_TEMPLATE.value
    new_biomod_file = Models.WU_WITHOUT_FLOATING_BASE_QUAT_VARIABLES.value
    utils.add_header(model_template_path, new_biomod_file, thorax_values)

    biorbd_model = (biorbd.Model(new_biomod_file), biorbd.Model(new_biomod_file))
    marker_ref = [m.to_string() for m in biorbd_model[0].markerNames()]

    # get key events
    event = load_events.LoadEvent(c3d_path=c3d_path, marker_list=marker_ref)
    data = load_experimental_data.LoadData(biorbd_model[0], c3d_path, q_file_path, qdot_file_path)
    x_initial = []
    u_initial = []
    phase_times = []
    targets = []
    n_shooting = []
    for i in range(2):
        if c3d_path == Tasks.EAT.value:
            start_frame = event.get_frame(i + 1)
            end_frame = event.get_frame(i + 2)
            phase_time = event.get_time(i + 2) - event.get_time(i + 1)
        else:
            start_frame = event.get_frame(i)
            end_frame = event.get_frame(i + 1)
            phase_time = event.get_time(i + 1) - event.get_time(i)
        target = data.get_marker_ref(
            number_shooting_points=[n_shooting_points],
            phase_time=[phase_time],
            start=int(start_frame),
            end=int(end_frame),
        )

        # load initial guesses
        q_ref, qdot_ref, tau_ref = data.get_variables_ref(
            number_shooting_points=[n_shooting_points],
            phase_time=[phase_time],
            start=int(start_frame),
            end=int(end_frame),
        )
        x_init_ref = np.concatenate([q_ref[0][6:, :], qdot_ref[0][6:, :]])  # without floating base
        u_init_ref = tau_ref[0][6:, :]
        n_q = biorbd_model[0].nbQ()
        n_qdot = biorbd_model[0].nbQdot()
        x_init_quat = np.vstack((np.zeros((n_q, n_shooting_points + 1)), np.ones((n_qdot, n_shooting_points + 1))))
        for i in range(n_shooting_points + 1):
            x_quat_shoulder = eul2quat(x_init_ref[5:8, i])
            x_init_quat[5:8, i] = x_quat_shoulder[1:]
            x_init_quat[10, i] = x_quat_shoulder[0]
        x_init_quat[:5] = x_init_ref[:5]
        x_init_quat[8:10] = x_init_ref[8:10]
        x_init_quat[11:, :] = x_init_ref[10:, :]
        x_initial.append(x_init_quat)
        u_initial.append(u_init_ref)
        phase_times.append(phase_time)
        targets.append(target[0])
        n_shooting.append(n_shooting_points)

    # optimal control program
    my_ocp = prepare_ocp(
        biorbd_model_path=new_biomod_file,
        task=task,
        track_markers=track_markers,
        x_init_ref=x_initial,
        u_init_ref=u_initial,
        target=targets,
        n_shooting=n_shooting,
        use_sx=False,
        n_threads=4,
        phase_time=phase_times,
    )

    # my_ocp.print()

    # add figures of constraints and objectives
    my_ocp.add_plot_penalty(CostType.ALL)

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver.set_linear_solver("ma57")
    solver.set_maximum_iterations(nb_iteration)
    sol = my_ocp.solve(solver)
    # sol.graphs()
    # sol.print_cost()

    # --- Save --- #
    c3d_str = c3d_path.split("/")
    c3d_name = os.path.splitext(c3d_str[-1])[0]
    save_path = f"save/quat/6_juillet/multi//{track_markers}/controls/{c3d_name}_{datetime.now()}"
    save_path = save_path.replace(" ", "_").replace("-", "_").replace(":", "_").replace(".", "_")
    my_ocp.save(sol, save_path)

    # --- Plot --- #
    q = np.zeros((3, n_shooting[0] + n_shooting[1]))
    Quaternion = np.zeros(4)
    for i in range(n_shooting[0] + n_shooting[1]):
        Q = np.concatenate([sol.states[0]["q"], sol.states[1]["q"]], axis=1)[:, i]
        Quaternion[0] = Q[10]
        Quaternion[1:] = Q[5:8]
        euler = quat2eul(Quaternion)
        q[:, i] = np.array(euler)
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True)
    for i in range(3):
        fig.add_trace(go.Scatter(y=q[i]), row=0 + 1, col=i + 1)
    fig.show()


if __name__ == "__main__":
    # main(Tasks.TEETH, False)
    # main(Tasks.DRINK, True)
    # main(Tasks.HEAD, False)
    main(Tasks.EAT, False)
    # main(Tasks.ARMPIT, track_markers=False)
