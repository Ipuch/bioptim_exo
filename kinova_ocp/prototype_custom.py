import biorbd_casadi as biorbd
from biorbd import get_range_q
from bioptim import (
    OptimalControlProgram,
    ObjectiveList,
    ObjectiveFcn,
    OdeSolver,
    DynamicsList,
    InitialGuessList,
    BoundsList,
    Solver,
    CostType,
    InterpolationType,
    Node,
    BiorbdModel,
    MultiBiorbdModel,
    PhaseDynamics,
    ConstraintList,
)
from custom_dynamics import custom_configure, torque_driven
import numpy as np
from data.enums import Tasks, TasksKinova
import data.load_events as load_events
from models.enums import Models
from models.biorbd_model import NewModel, KinovaModel
import tracking.load_experimental_data as load_experimental_data
from custom_ik_objectives import superimpose_markers, superimpose_matrix


# Todo: try to get a nice follow of the kinova arm with the forearm of the upper limb.

def prepare_ocp(
    upper_limb_biorbd_model_path: str,
    kinova_model_path: str,
    n_shooting: int,
    x_init_ref: np.array,
    u_init_ref: np.array,
    target: any,
    ode_solver: OdeSolver = OdeSolver.RK4(),
    use_sx: bool = False,
    n_threads: int = 16,
    phase_time: float = 1,
) -> OptimalControlProgram:

    biorbd_model = MultiBiorbdModel(
        bio_model=(upper_limb_biorbd_model_path,),
        extra_bio_models=(kinova_model_path,),
    )

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=3)
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", index=range(5), weight=100, derivative=True
    )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", index=range(5, 10), weight=1000, derivative=True
    )
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=5000)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", index=[0, 1, 2, 3, 4], weight=10000)
    objective_functions.add(
        ObjectiveFcn.Mayer.TRACK_MARKERS, weight=1000, marker_index=[10, 12, 13, 14, 15], target=target, node=Node.ALL,
    )

    # inverse kinematics objectives
    # todo: add inverse kinematics objectives
    # regularization
    # objective_functions.add(
    #     ObjectiveFcn.Mayer.MINIMIZE_CONTROL,
    #     key="q_k",
    #     weight=1e-3,
    #     target=np.repeat(np.array([0, 0, 0, 0, -2, 0, 0])[:, np.newaxis], axis=1, repeats=n_shooting),
    #     node=Node.ALL_SHOOTING,
    #     quadratic=True
    # )
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="q_k", weight=100, index=1)
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="q_k", weight=100, derivative=True)
    objective_functions.add(
        superimpose_markers,
        custom_type=ObjectiveFcn.Mayer,
        weight=1000,
        first_model_marker=["flexion_axis0", "flexion_axis1"],
        second_model_marker="md0",
        node=Node.ALL,
        quadratic=True,
    )
    objective_functions.add(
        superimpose_matrix,
        custom_type=ObjectiveFcn.Mayer,
        weight=1000,
        track_segment=1,
        first_model_marker="flexion_axis0",
        second_model_marker="flexion_axis1",
        third_model_marker="ulna_longitudinal_frame",
        node=Node.ALL,
        quadratic=True,
    )

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(custom_configure, dynamic_function=torque_driven, expand_dynamics=True, phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE)

    # initial guesses
    x_init = InitialGuessList()
    x_init.add("q", x_init_ref[:biorbd_model.nb_q, :], interpolation=InterpolationType.EACH_FRAME)
    x_init.add("qdot", x_init_ref[biorbd_model.nb_q: biorbd_model.nb_q + biorbd_model.nb_qdot, :],
               interpolation=InterpolationType.EACH_FRAME)

    u_init = InitialGuessList()
    u_init.add("tau", u_init_ref, interpolation=InterpolationType.EACH_FRAME)
    u_init.add(
        "q_k",
        [0] * biorbd_model.extra_models[0].nb_q,
        interpolation=InterpolationType.CONSTANT,
    )

    # Define control path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = biorbd_model.bounds_from_ranges("q")
    x_bounds["q"][:, 0] = x_init_ref[:10, 0]
    x_bounds["q"][:, -1] = x_init_ref[:10, -1]

    x_bounds["qdot"] = biorbd_model.bounds_from_ranges("qdot")
    x_bounds["qdot"].min[:, 0] = [-1e-3] * biorbd_model.nb_qdot
    x_bounds["qdot"].max[:, 0] = [1e-3] * biorbd_model.nb_qdot
    x_bounds["qdot"].min[:, -1] = [-1e-1] * biorbd_model.nb_qdot
    x_bounds["qdot"].max[:, -1] = [1e-1] * biorbd_model.nb_qdot

    n_tau = biorbd_model.nb_tau
    tau_min, tau_max = -100, 100
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * n_tau, [tau_max] * n_tau
    u_bounds["tau"].min[:2, :], u_bounds["tau"].max[:2, :] = -30, 30
    u_bounds["tau"].min[2:5, :], u_bounds["tau"].max[2:5, :] = -50, 50
    u_bounds["tau"].min[5:8, :], u_bounds["tau"].max[5:8, :] = -70, 70
    u_bounds["tau"].min[8:, :], u_bounds["tau"].max[8:, :] = -40, 40

    kinova_q_ranges = biorbd_model.extra_models[0].ranges_from_model("q")
    u_bounds["q_k"] = [b.min() for b in kinova_q_ranges], [b.max() for b in kinova_q_ranges]

    # # translation limited by design of the kinova arm
    # u_bounds["q_k"].max[1, :] = np.linalg.norm([-0.08842, -0.02369]) + 2 * 0.120
    # u_bounds["q_k"].min[1, :] = np.linalg.norm([-0.08842, -0.02369]) + 2 * 0.120

    # should try to guide the polar coordinates...
    # u_bounds["q_k"].max[0, :] = 1
    # u_bounds["q_k"].min[0, :] = -1

    u_bounds["q_k"].max[1, :] = 1
    u_bounds["q_k"].min[1, :] = -1

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


def main(task: Tasks = None):
    """
    Get data, then create a tracking problem, and finally solve it and plot some relevant information
    """

    # Define the problem
    n_shooting_points = 40
    nb_iteration = 3000

    task_files = load_events.LoadTask(task=task, model=Models.WU_INVERSE_KINEMATICS_XYZ_OFFSET)

    model = NewModel(model=Models.WU_WITHOUT_FLOATING_BASE_OFFSET_VARIABLES)
    model.add_header(
        model_template=Models.WU_WITHOUT_FLOATING_BASE_OFFSET_TEMPLATE,
        q_file_path=task_files.q_file_path,
    )

    biorbd_model = biorbd.Model(model.model_path)
    bioptim_model = BiorbdModel(model.model_path)

    # get key events
    event = load_events.LoadEvent(task=task, marker_list=bioptim_model.marker_names)
    data = load_experimental_data.LoadData(
        model=biorbd_model,
        c3d_file=task_files.c3d_path,
        q_file=task_files.q_file_path,
        qdot_file=task_files.qdot_file_path,
    )

    target = data.get_marker_ref(
        number_shooting_points=[n_shooting_points],
        phase_time=[event.phase_time() + np.random.rand(1)[0] * 0.001],
        markers_names=["ULNA", "RADIUS", "SEML", "MET2", "MET5"],
        start=event.start_frame(),
        end=event.end_frame(),
    )

    # load initial guesses
    q_ref, qdot_ref, tau_ref = data.get_variables_ref(
        number_shooting_points=[n_shooting_points],
        phase_time=[event.phase_time() + np.random.rand(1)[0] * 0.001],
        start=event.start_frame(),
        end=event.end_frame(),
    )
    x_init_ref = np.concatenate([q_ref[0][6:, :], qdot_ref[0][6:, :]])  # without floating base
    u_init_ref = tau_ref[0][6:, :]

    kinova_model = KinovaModel(model=Models.KINOVA_RIGHT_SLIDE_POLAR_BASE_WITH_VARIABLES)
    marker_attachment_name = "Table:Table5"
    index_marker_attachment = data.c3d_data.c3d["parameters"]["POINT"]["LABELS"]["value"].index(marker_attachment_name)
    attachment_point_location = np.mean(data.c3d_data.c3d["data"]["points"][:, index_marker_attachment, :], axis=1)[0:3]
    kinova_model.add_header(
        model_template=Models.KINOVA_RIGHT_SLIDE_POLAR_BASE_TEMPLATE,
        attachment_point_location=attachment_point_location / 1000,  # convert in meters
        offset=np.array([0, 0., 0.]),
                           )

    # optimal control program
    my_ocp = prepare_ocp(
        upper_limb_biorbd_model_path=model.model_path,
        kinova_model_path=Models.KINOVA_RIGHT_SLIDE_POLAR_BASE.value,
        x_init_ref=x_init_ref,
        u_init_ref=u_init_ref,
        target=target[0],
        n_shooting=n_shooting_points,
        use_sx=False,
        n_threads=4,
        phase_time=1.05,  # to handle numerical error
    )

    # add figures of constraints and objectives
    my_ocp.add_plot_penalty(CostType.ALL)

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver.set_linear_solver("mumps")
    # solver.set_maximum_iterations(nb_iteration)
    solver.set_maximum_iterations(0)
    sol = my_ocp.solve(solver)
    # sol.print_cost()

    # --- Plot --- #
    # sol.graphs(show_bounds=True)

    # n_frames_animate = 100

    from extra_viz import custom_animate
    custom_animate(
        model_kinova_path=Models.KINOVA_RIGHT_SLIDE_POLAR_BASE_WITH_VARIABLES.value,
        model_path_upperlimb=model.model_path,
        q_k=sol.controls["q_k"],
        q_upper_limb=sol.states["q"],
    )


if __name__ == "__main__":
    main(TasksKinova.DRINK)

    # main(Tasks.DRINK)
    # main(Tasks.TEETH)
    # main(Tasks.DRINK)
    # main(Tasks.HEAD)
    # main(Tasks.ARMPIT)
    # main(Tasks.EAT)
