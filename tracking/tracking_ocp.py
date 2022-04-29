from load_experimental_data import LoadData, C3dData
import numpy as np
import biorbd_casadi as biorbd
from matplotlib import pyplot as plt
import os

from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    InterpolationType,
    BoundsList,
    QAndQDotBounds,
    InitialGuessList,
    OdeSolver,
    Node,
)


class TrackingOcp:
    """
     The class for dealing tracking problem

     Attributes
     ----------
     with_floating_base: bool
         True if there is the 6 dof of thorax
     ode_solver: ode_solver = OdeSolver.RK4()
         Which type of OdeSolver to use
     model_path: str
         Path to BioMod file
     biorbd_model: biorbd.Model
         The loaded biorbd model
     q_file: str
         Path to q file
     qdot_file: str
         Path to qdot file
     data: C3dData
         The date extract from c3D file
     final_time: float
         The time at final node
     n_shooting_points: int
         The number of shooting point
     nb_iteration: int
         The Number of iteration for the OCP
     data_loaded: LoadData
         The date from c3d, q and qdot files
     q_ref: list
         List of the array of joint trajectories.
         Those trajectories were computed using Kalman filter
         They are used as initial guess
     qdot_ref: list
         List of the array of joint velocities.
         Those velocities were computed using Kalman filter
         They are used as initial guess
     markers_ref: List
         The list of Markers position at each shooting points
     ocp: OptimalControlProgram

    Methods
    -------
    prepare_ocp(self) -> OptimalControlProgram:
        Prepare the ocp to solve
    plot_c3d_markers_and_model_markers(self, c3d_path):
        Plot the positions of each Markers before and after interpolation
    """

    def __init__(
            self,
            with_floating_base: bool,
            c3d_path: str,
            n_shooting_points: int,
            nb_iteration: int,
            ode_solver: OdeSolver = OdeSolver.RK4(),
            model_path: str = None,
            n_threads: int = 6,
            final_time: float = None,
            markers_tracked: list = ["all"],
    ):
        self.with_floating_base = with_floating_base
        if model_path is None:
            model_path_without_floating_base = "../models/wu_converted_definitif_without_floating_base.bioMod"
            model_path_with_floating_base = "../models/wu_converted_definitif.bioMod"
            self.model_path = model_path_with_floating_base if self.with_floating_base \
                                                            else model_path_without_floating_base
        else:
            self.model_path = model_path

        self.n_shooting_points = n_shooting_points
        self.nb_iteration = nb_iteration
        self.ode_solver = ode_solver
        self.n_threads = n_threads
        self.biorbd_model = biorbd.Model(self.model_path)

        self.q_file = os.path.splitext(c3d_path)[0] + "_q.txt"
        self.qdot_file = os.path.splitext(c3d_path)[0] + "_qdot.txt"

        self.c3d_data = C3dData(c3d_path, self.biorbd_model)

        self.final_time = self.c3d_data.get_final_time() if final_time is None else final_time
        self.data_loaded = LoadData(self.biorbd_model, c3d_path, self.q_file, self.qdot_file)

        # self.q_ref, self.qdot_ref, self.markers_ref = self.data_loaded.get_experimental_data(
        #     [self.n_shooting_points], [self.final_time], with_floating_base=with_floating_base
        # )
        self.q_ref, self.qdot_ref = self.data_loaded.get_states_ref(
            [self.n_shooting_points], [self.final_time], with_floating_base=with_floating_base
        )
        self.markers_tracked = markers_tracked
        self.markers_ref = self.data_loaded.get_marker_ref([self.n_shooting_points],
                                                           [self.final_time],
                                                           self.markers_tracked)
        self.ocp = None
        self.prepare_ocp()

    def prepare_ocp(self):
        nb_q = self.biorbd_model.nbQ()
        nb_qdot = self.biorbd_model.nbQdot()

        # Add objective functions
        objective_functions = ObjectiveList()
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=10)
        objective_functions.add(
            ObjectiveFcn.Mayer.TRACK_MARKERS, weight=100000, target=self.markers_ref[0], node=Node.ALL
        )
        if self.with_floating_base:
            objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_STATE, weight=10, key="qdot", index=[0, 1, 2, 3, 4, 5]
            )
        else:
            objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, weight=10, key="qdot")

        # Dynamics
        dynamics = DynamicsList()
        dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

        # Path constraint
        x_bounds = BoundsList()
        x_bounds.add(bounds=QAndQDotBounds(self.biorbd_model))

        # Initial guess
        init_x = np.zeros((nb_q + nb_qdot, self.n_shooting_points + 1))
        init_x[:nb_q, :] = self.q_ref[0]
        init_x[nb_q: nb_q + nb_qdot, :] = self.qdot_ref[0]

        x_init = InitialGuessList()
        x_init.add(init_x, interpolation=InterpolationType.EACH_FRAME)

        # Define control path constraint
        u_bounds = BoundsList()
        u_init = InitialGuessList()
        tau_min, tau_max, tau_init = -100, 100, 0
        u_bounds.add(
            [tau_min] * self.biorbd_model.nbGeneralizedTorque(),
            [tau_max] * self.biorbd_model.nbGeneralizedTorque(),
        )
        u_init.add([tau_init] * self.biorbd_model.nbGeneralizedTorque())

        # ------------- #
        self.ocp = OptimalControlProgram(
            biorbd_model=self.biorbd_model,
            dynamics=dynamics,
            n_shooting=self.n_shooting_points,
            phase_time=self.final_time,
            x_init=x_init,
            u_init=u_init,
            x_bounds=x_bounds,
            u_bounds=u_bounds,
            objective_functions=objective_functions,
            ode_solver=self.ode_solver,
            n_threads=self.n_threads,
        )

    def plot_c3d_markers_and_model_markers(self, c3d_path: str):
        """
        plot c3d markers and model markers on the same plot

        Parameters:
        c3d_path: str
            The Path to c3d file
        """
        xp_data = LoadData(self.biorbd_model, c3d_path, self.q_file, self.qdot_file)
        xp_data.c3d_data.trajectories
        tf = self.final_time
        list_dir = ["X", "Y", "Z"]
        for j in range(0, 16):
            for i, direction in enumerate(list_dir):
                plt.figure(j)
                a = 311 + i
                plt.subplot(a)
                plt.plot(
                    [
                        i / len(xp_data.c3d_data.trajectories[0][0]) * tf
                        for i in range(len(xp_data.c3d_data.trajectories[0][0]))
                    ],
                    xp_data.c3d_data.trajectories[i, j, :],
                    label=f"{xp_data.c3d_data.marker_names[j]} {direction} c3d",
                    marker="o",
                )
                plt.plot(
                    [i / (self.n_shooting_points + 1) * tf for i in range(self.n_shooting_points + 1)],
                    self.markers_ref[0][i, j, :],
                    label=f"{xp_data.c3d_data.marker_names[j]} {direction} interpolate",
                    marker="o",
                )
                plt.legend()
        plt.show()
        plt.legend()
        plt.show()
