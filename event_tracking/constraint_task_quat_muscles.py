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
    BoundsList,
    Solver,
    CostType,
    InterpolationType,
    IntegralApproximation,
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

"""
Did not converge.
"""

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
    n_threads: int = 16,
    phase_time: float = 1,
) -> object:
    biorbd_model = biorbd.Model(biorbd_model_path)
    nb_q = biorbd_model.nbQ()

    # Add objective functions
    objective_functions = ObjectiveList()

    if task == Tasks.TEETH:
        print("dents!")
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=500)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1000)
        # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1000)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[0, 1, 2], weight=500)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[3, 4], weight=500)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=500)
        if track_markers:
            print("tracking markers")
            objective_functions.add(
                ObjectiveFcn.Mayer.TRACK_MARKERS,
                weight=150,
                target=target,
                node=Node.ALL,
            )

    elif task == Tasks.DRINK:
        print("boire!")
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=500)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1000)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", derivative=True, weight=300)
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            key="q",
            index=[0, 1, 2],
            weight=500,
            integration_rule=IntegralApproximation.TRAPEZOIDAL,
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            key="q",
            index=[3, 4],
            weight=500,
            integration_rule=IntegralApproximation.TRAPEZOIDAL,
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            key="qdot",
            weight=500,
            integration_rule=IntegralApproximation.TRAPEZOIDAL,
        )
        if track_markers:
            # does not converge without tracking markers
            print("tracking markers boire")
            objective_functions.add(
                ObjectiveFcn.Mayer.TRACK_MARKERS,
                weight=6000,  # 1000
                target=target,
                node=Node.ALL,
            )

    elif task == Tasks.HEAD:
        print("tete!")
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=500)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1000)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", derivative=True, weight=300)
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            key="q",
            index=[0, 1, 2],
            weight=500,
            integration_rule=IntegralApproximation.TRAPEZOIDAL,
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            key="q",
            index=[3, 4],
            weight=500,
            integration_rule=IntegralApproximation.TRAPEZOIDAL,
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            key="qdot",
            weight=500,
            integration_rule=IntegralApproximation.TRAPEZOIDAL,
        )
        if track_markers:
            # does not converge without tracking markers
            print("tracking markers tete")
            objective_functions.add(
                ObjectiveFcn.Mayer.TRACK_MARKERS,
                weight=6000,
                target=target,
                node=Node.ALL,
            )

    elif task == Tasks.EAT:
        print("manger!")
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=100)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1000)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1000)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[0, 1, 2], weight=500)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[3, 4], weight=50)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=600)
        objective_functions.add(ObjectiveFcn.Mayer.TRACK_MARKERS, weight=150, target=target, node=Node.ALL)

    elif task == Tasks.ARMPIT:
        print("aisselle!")
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=500)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1000)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", derivative=True, weight=300)
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            key="q",
            index=[0, 1, 2],
            weight=500,
            integration_rule=IntegralApproximation.TRAPEZOIDAL,
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            key="q",
            index=[3, 4],
            weight=500,
            integration_rule=IntegralApproximation.TRAPEZOIDAL,
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            key="qdot",
            weight=500,
            integration_rule=IntegralApproximation.TRAPEZOIDAL,
        )
        if track_markers:
            # does not converge without tracking markers
            print("tracking markers aisselle")
            objective_functions.add(
                ObjectiveFcn.Mayer.TRACK_MARKERS,
                weight=6000,
                target=target,
                node=Node.ALL,
            )

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.MUSCLE_DRIVEN, with_torque=True)

    # Define control path constraint
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[:nb_q, 0] = x_init_ref[:nb_q, 0]
    x_bounds[:nb_q, -1] = x_init_ref[:nb_q, -1]
    x_bounds.min[nb_q:, 0] = [-1e-3] * biorbd_model.nbQdot()
    x_bounds.max[nb_q:, 0] = [1e-3] * biorbd_model.nbQdot()
    x_bounds.min[nb_q:, -1] = [-1e-1] * biorbd_model.nbQdot()
    x_bounds.max[nb_q:, -1] = [1e-1] * biorbd_model.nbQdot()
    x_bounds.min[8:10, 1], x_bounds.min[10, 1] = x_bounds.min[9:11, 1], -1
    x_bounds.max[8:10, 1], x_bounds.max[10, 1] = x_bounds.max[9:11, 1], 1

    muscle_min, muscle_max, muscle_init = 0, 1, 0.05

    # initial guesses
    x_init = InitialGuess(x_init_ref, interpolation=InterpolationType.EACH_FRAME)

    tau_min, tau_max, tau_init = -20, 20, 0
    u_bounds = BoundsList()
    u_bounds.add(
        [tau_min] * biorbd_model.nbGeneralizedTorque() + [muscle_min] * biorbd_model.nbMuscleTotal(),
        [tau_max] * biorbd_model.nbGeneralizedTorque() + [muscle_max] * biorbd_model.nbMuscleTotal(),
    )

    # todo: use u_init_ref
    u_init = InitialGuess([tau_init] * biorbd_model.nbGeneralizedTorque() + [muscle_init] * biorbd_model.nbMuscles())

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

    biorbd_model = biorbd.Model(new_biomod_file)
    marker_ref = [m.to_string() for m in biorbd_model.markerNames()]

    # get key events
    event = load_events.LoadEvent(c3d_path=c3d_path, marker_list=marker_ref)
    data = load_experimental_data.LoadData(biorbd_model, c3d_path, q_file_path, qdot_file_path)
    if c3d_path == Tasks.EAT.value:
        start_frame = event.get_frame(1)
        end_frame = event.get_frame(2)
        phase_time = event.get_time(2) - event.get_time(1)
    else:
        start_frame = event.get_frame(0)
        end_frame = event.get_frame(1)
        phase_time = event.get_time(1) - event.get_time(0)
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
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    x_init_quat = np.vstack((np.zeros((nb_q, n_shooting_points + 1)), np.ones((nb_qdot, n_shooting_points + 1))))
    for i in range(n_shooting_points + 1):
        x_quat_shoulder = eul2quat(x_init_ref[5:8, i])
        x_init_quat[5:8, i] = x_quat_shoulder[1:]
        x_init_quat[10, i] = x_quat_shoulder[0]
    x_init_quat[:5] = x_init_ref[:5]
    x_init_quat[8:10] = x_init_ref[8:10]
    x_init_quat[11:, :] = x_init_ref[10:, :]

    # optimal control program
    my_ocp = prepare_ocp(
        biorbd_model_path=new_biomod_file,
        task=task,
        track_markers=track_markers,
        x_init_ref=x_init_quat,
        u_init_ref=u_init_ref,
        target=target[0],
        n_shooting=n_shooting_points,
        use_sx=False,
        n_threads=3,
        phase_time=phase_time,
    )

    # add figures of constraints and objectives
    my_ocp.add_plot_penalty(CostType.ALL)

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver.set_linear_solver("ma57")
    solver.set_maximum_iterations(nb_iteration)
    sol = my_ocp.solve(solver)
    sol.graphs()
    sol.print_cost()

    # --- Save --- #
    c3d_str = c3d_path.split("/")
    c3d_name = os.path.splitext(c3d_str[-1])[0]
    save_path = f"save/quat/6_juillet//muscles/{track_markers}/{c3d_name}_{datetime.now()}"
    save_path = save_path.replace(" ", "_").replace("-", "_").replace(":", "_").replace(".", "_")
    my_ocp.save(sol, save_path)

    # --- Plot --- #
    q = np.zeros((3, n_shooting_points))
    Quaternion = np.zeros(4)
    for i in range(n_shooting_points):
        Q = sol.states["q"][:, i]
        Quaternion[0] = Q[10]
        Quaternion[1:] = Q[5:8]
        euler = quat2eul(Quaternion)
        q[:, i] = np.array(euler)
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True)
    j = 0
    for i in range(3):
        ii = i - j * 4
        fig.add_trace(go.Scatter(y=q[i]), row=j + 1, col=ii + 1)
    fig.show()


if __name__ == "__main__":
    # main(task=Tasks.EAT, track_markers=False)
    main(task=Tasks.TEETH, track_markers=False)
    # main(task=Tasks.DRINK, track_markers=False)
    # main(task=Tasks.HEAD, track_markers=False)
    # main(task=Tasks.ARMPIT, track_markers=False)
