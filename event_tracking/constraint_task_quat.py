import biorbd_casadi as biorbd
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
import numpy as np
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime
import models.utils as utils
import data.load_events as load_events
import tracking.load_experimental_data as load_experimental_data


# import sys
# sys.path.append("../models")
# sys.path.append("../data")
# sys.path.append("../tracking")


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
    nb_q = biorbd_model.nbQ()

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1000)
    # objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="qdot", weight=50, node=Node.START)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[0, 1, 2, 3, 4], weight=100)
    # objective_functions.add(

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # initial guesses
    x_init = InitialGuess([1e-5] * (nb_q + nb_qdot))
    u_init = InitialGuess([0] * biorbd_model.nbGeneralizedTorque())
    # x_qdot = np.ones((10, n_shooting+1)) * 1e-5
    # x_init_ref[11:, :] = x_qdot
    x_init = InitialGuess(x_init_ref, interpolation=InterpolationType.EACH_FRAME)
    u_init = InitialGuess(u_init_ref, interpolation=InterpolationType.EACH_FRAME)
    names = [i.to_string() for i in biorbd_model.nameDof()]
    fig = make_subplots(rows=5, cols=4, subplot_titles=names, shared_yaxes=True)
    j = 0
    for i in range(12):
        ii = i - j * 4
        fig.add_trace(go.Scatter(y=x_init_ref[i]), row=j + 1, col=ii + 1)
        if ii == 3:
            j += 1

    # fig.show()

    # Define control path constraint
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[:nb_q, 0] = x_init_ref[:nb_q, 0]
    x_bounds[:nb_q, -1] = x_init_ref[:nb_q, -1]
    # x_bounds.min[10:, 0] = [-np.pi / 4] * biorbd_model.nbQdot()
    # x_bounds.max[10:, 0] = [np.pi / 4] * biorbd_model.nbQdot()
    # x_bounds.min[10:, -1] = [-np.pi / 2] * biorbd_model.nbQdot()
    # x_bounds.max[10:, -1] = [np.pi / 2] * biorbd_model.nbQdot()
    x_bounds.min[8:10, 1], x_bounds.min[10, 1] = x_bounds.min[9:11, 1],  -1
    x_bounds.max[8:10, 1], x_bounds.max[10, 1] = x_bounds.max[9:11, 1], 1

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


def main(c3d_path: str):
    """
    Get data, then create a tracking problem, and finally solve it and plot some relevant information
    """

    # Define the problem
    # c3d_path = "F0_dents_05.c3d"
    # todo: manger, aisselle, dessiner
    n_shooting_points = 50
    nb_iteration = 10000

    q_file_path = c3d_path.removesuffix(".c3d") + "_q.txt"
    qdot_file_path = c3d_path.removesuffix(".c3d") + "_qdot.txt"

    thorax_values = utils.thorax_variables(q_file_path)  # load c3d floating base pose
    new_biomod_file = "../models/wu_converted_definitif_without_floating_base_template_quat_with_variables.bioMod"
    model_path_without_floating_base = "../models/wu_converted_definitif_without_floating_base_template_quat.bioMod"
    utils.add_header(model_path_without_floating_base, new_biomod_file, thorax_values)

    # biomod_path = "/home/lim/Documents/Stage_Thasaarah/bioptim_exo/models/wu_converted_definitif_without_modif_quat.bioMod"
    biorbd_model = biorbd.Model(new_biomod_file)
    marker_ref = [m.to_string() for m in biorbd_model.markerNames()]

    # get key events
    event = load_events.LoadEvent(c3d_path=c3d_path, marker_list=marker_ref)
    data = load_experimental_data.LoadData(biorbd_model, c3d_path, q_file_path, qdot_file_path)
    # target_start = event.get_markers(0)[:, :, np.newaxis]
    # target_end = event.get_markers(1)[:, :, np.newaxis]
    start_frame = event.get_frame(0)
    end_frame = event.get_frame(1)
    phase_time = event.get_time(1) - event.get_time(0)
    phase_time = 0.3
    target = data.get_marker_ref(
        number_shooting_points=[n_shooting_points],
        phase_time=[phase_time],
        markers_names=["ULNA", "RADIUS", "SEML", "MET2", "MET5"],
        start=int(start_frame),
        end=int(end_frame),
    )
    # target = data.dispatch_data(target_frame, nb_shooting=[n_shooting_points], phase_time=[phase_time],
    # start=start_frame, end=end_frame)

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
        x_init_quat[5:8, i] = x_quat_shoulder[1:].toarray().squeeze()
        x_init_quat[10, i] = x_quat_shoulder[0].toarray().squeeze()
    x_init_quat[:5] = x_init_ref[:5]
    x_init_quat[8:10] = x_init_ref[8:10]
    x_init_quat[11:, :] = x_init_ref[10:, :]

    # optimal control program
    my_ocp = prepare_ocp(
        biorbd_model_path=new_biomod_file,
        c3d_path=c3d_path,
        x_init_ref=x_init_quat,
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
    # sol.graphs()
    # sol.print_cost()

    # --- Save --- #
    c3d_str = c3d_path.split("/")
    c3d_name = os.path.splitext(c3d_str[-1])[0]
    save_path = f"save/quat/{c3d_name}_{datetime.now()}"
    save_path = save_path.replace(" ", "_").replace("-", "_").replace(":", "_").replace(".", "_")
    my_ocp.save(sol, save_path)

    # --- Plot --- #
    # sol.graphs(show_bounds=True)
    # todo: animate first and last frame with markers
    # sol.animate(n_frames=100)


if __name__ == "__main__":
    main("../data/xyz humerus rotations/F0_aisselle_05.c3d")
    main("../data/xyz humerus rotations/F0_boire_05.c3d")
    main("../data/xyz humerus rotations/F0_aisselle_05.c3d")
    main("../data/xyz humerus rotations/F0_tete_05.c3d")
