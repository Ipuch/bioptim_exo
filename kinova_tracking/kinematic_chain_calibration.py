import random
import time
from enum import Enum
from typing import Union

import biorbd as biorbd_eigen
import biorbd_casadi as biorbd
import matplotlib.pyplot as plt
import numpy as np
from casadi import MX, Function, vertcat, nlpsol, if_else, sumsqr
from casadi import fabs

from utils import get_range_q


class ObjectivesFunctions(Enum):
    ALL_OBJECTIVES = "all objectives"
    ALL_OBJECTIVES_WITHOUT_FINAL_ROTMAT = "all objectives without final rotmat"


class KinematicChainCalibration:
    """

    Attributes
    ---------
    biord_model : biorbd.Model
        The biorbd Model
    markers_model : list[str]
        Name of each marker
    marker : np.ndarray
        matrix of zeros [3 x Nb markers , x nb frame]
    closed_loop_markers : list[str]
        Name of markers associated to the table
    tracked_markers : list[str]
        Name of associated to the model
    parameter_dofs : list[str]
        name dof for which parameters are constant on each frame
    kinematic_dofs : list
        name dof which parameters aren't constant on each frame
    weights_param :np.ndarray
        weight associated with cost functions
    x_ik_initial_guess : array
        initialize q
    nb_frames_ik_step : int
        number frame for Inverse Kinematics steps
    nb_frames_param_step : int
        number of frame for parameters Identification steps
    randomize_param_step_frames : bool
        randomly choose the frames among the trial sent
    use_analytical_jacobians : bool
        Use analytical jacobians instead of numerical ones
    segment_id_with_vertical_z : int
        the segment of the Kinova arm which is fit with the table

    Examples
    ---------
    kcc = KinematicChainCalibration()
    kcc.solve()
    kkc.results()
    """

    def __init__(
            self,
            biorbd_model: biorbd.Model,
            markers_model: list[str],
            markers: np.array,  # [3 x nb_markers_model x nb_frames]
            closed_loop_markers: list[str],
            tracked_markers: list[str],
            parameter_dofs: list[str],
            kinematic_dofs: list[str],
            weights_param: Union[list[float], np.ndarray],
            weights_ik: Union[list[float], np.ndarray],
            x_ik_initial_guess: np.ndarray,
            objectives_functions: ObjectivesFunctions = None,  # [n_dof x n_frames]
            nb_frames_ik_step: int = None,
            nb_frames_param_step: int = None,
            randomize_param_step_frames: bool = True,
            use_analytical_jacobians: bool = True,
            segment_id_with_vertical_z: int = None,
            param_solver: str = "leastsquare",
            ik_solver: str = "leastsquare",
            method: str = "1step",
            same_variables: dict = None,
    ):

        self.nb_markers = None
        self.biorbd_model = biorbd.Model(biorbd_model.path().absolutePath().to_string())
        self.biorbd_model_eigen = biorbd_eigen.Model(biorbd_model.path().absolutePath().to_string())
        self.model_dofs = [dof.to_string() for dof in biorbd_model.nameDof()]

        self.nb_markers = self.biorbd_model.nbMarkers()
        self.nb_frames = markers.shape[2]

        # check if markers_model are in model
        # otherwise raise error
        for marker in markers_model:
            if marker not in [i.to_string() for i in biorbd_model.markerNames()]:
                raise ValueError(f"The following marker is not in markers_model:{marker}")
            else:
                self.markers_model = markers_model

        # check if markers model and makers have the same size
        # otherwise raise
        if markers.shape == (3, len(markers_model), nb_frames_ik_step):
            self.markers = markers
        else:
            raise ValueError(
                f"markers and markers model must have same shape, markers shape is {markers.shape()},"
                f" and markers_model shape is {markers_model.shape()}"
            )
        self.closed_loop_markers = closed_loop_markers
        self.tracked_markers = tracked_markers
        self.parameter_dofs = parameter_dofs
        self.kinematic_dofs = kinematic_dofs
        self.same_variables = [same_variables["the main dof"], same_variables["the old dof"]]
        self.index_same_variables = [same_variables["index main dof"], same_variables["index old dof"]]

        # find the indexes of closed loop markers and tracked markers
        self.table_markers_idx = [self.markers_model.index(i) for i in self.markers_model if "Table" in i]
        self.model_markers_idx = [self.tracked_markers.index(i) for i in self.tracked_markers]

        # nb markers
        self.nb_markers_table = self.table_markers_idx.__len__()
        self.nb_markers_model = self.model_markers_idx.__len__()

        # find the indexes of parameters and kinematic dofs in the model
        self.q_parameter_index = [self.model_dofs.index(dof) for dof in self.parameter_dofs]
        self.q_kinematic_index = [self.model_dofs.index(dof) for dof in self.kinematic_dofs]

        self.nb_parameters_dofs = len(parameter_dofs)
        self.nb_kinematic_dofs = len(kinematic_dofs)
        self.nb_total_dofs = self.nb_kinematic_dofs + self.nb_parameters_dofs

        # self.method
        self.method = method

        # self.objectives_function
        self.param_solver = param_solver
        self.ik_solver = ik_solver

        # check if q_ik_initial_guess has the right size
        self.x_ik_initial_guess = x_ik_initial_guess
        self.q_ik_initial_guess = self.x_ik_initial_guess[self.q_kinematic_index, :]
        self.p_ik_initial_guess = self.x_ik_initial_guess[self.q_parameter_index, 0]
        self.nb_frames_ik_step = nb_frames_ik_step
        self.nb_frames_param_step = nb_frames_param_step
        self.randomize_param_step_frames = randomize_param_step_frames
        self.use_analytical_jacobians = use_analytical_jacobians
        self.bounds = get_range_q(self.biorbd_model)
        self.bounds_q_list = [list(self.bounds[0][self.q_kinematic_index]),
                              list(self.bounds[1][self.q_kinematic_index])]
        self.bounds_p_list = [list(self.bounds[0][self.q_parameter_index]),
                              list(self.bounds[1][self.q_parameter_index])]
        self.bound_constraint = np.zeros(6)

        self.list_frames_param_step = self.frame_selector(self.nb_frames_param_step, self.nb_frames_ik_step)

        # number of weights has to be checked
        # raise Error if not the right number
        self.weights_param = weights_param
        self.weights_ik = weights_ik

        self.time_param = []
        self.time_ik = []

        self.list_sol = []
        self.q = np.zeros((self.biorbd_model.nbQ(), self.nb_frames_ik_step))
        self.segment_id_with_vertical_z = segment_id_with_vertical_z
        self.output = dict()

        # symbolic variables
        self.q_sym = MX.sym("q_sym", self.nb_kinematic_dofs)
        self.q_sym_prev = MX.sym("q_sym_prev", self.nb_kinematic_dofs)
        self.q_sym_global_vec = MX.sym("q_sym_global", self.nb_kinematic_dofs * self.nb_frames)

        # don't work yet
        # self.q_sym_global_vec = MX.sym("q_sym_global", self.nb_kinematic_dofs * self.nb_frames)
        # for x in range(self.index_same_variables[0], self.nb_frames*self.nb_kinematic_dofs, self.index_same_variables[0]):
        #     self.q_sym_global_vec[x+1] = self.q_sym_global_vec[x]

        self.q_sym_global = self.q_sym_global_vec.reshape((self.nb_kinematic_dofs, self.nb_frames))

        self.p_sym = MX.sym("p_sym", self.nb_parameters_dofs)

        # symbolic x with q qnd p variables
        self.x_sym = MX.zeros(self.nb_total_dofs)
        self.x_sym_prev = MX.zeros(self.nb_total_dofs)
        self.x_sym = self.build_x(self.q_sym, self.p_sym)
        self.x_sym_prev = self.build_x(self.q_sym_prev, self.p_sym)

        # symbolic xp data
        self.m_model_sym = MX.sym("markers_model", self.nb_markers_model * 3)
        self.m_table_sym = MX.sym("markers_table", 6)  # xyz and xyz

        self.frame = 0

        # initialize x which is a combination of q and p
        self.x = MX.zeros(self.nb_total_dofs)

    def build_x(self, q, p) -> MX:
        """
        This function built a x ( ie q and p ) for one frame
        Returns
        -------
        The MX of x
        """

        x = MX.zeros(self.nb_total_dofs)
        x[self.q_kinematic_index] = q
        x[self.q_parameter_index] = MX(p) if isinstance(p, np.ndarray) else p
        return x

    def build_x_all_frames(self):
        """
        This function built the final x ( ie q and p ) for all the frame, used in the animation
        Returns
        -------
        The array of solutions
        """
        x_all_frames = np.zeros((self.nb_total_dofs, self.nb_frames))
        for i in range(self.nb_frames):
            x_all_frames[self.q_kinematic_index, i] = self.q_all_frame[:, i]
            x_all_frames[self.q_parameter_index, i] = self.parameters
        return x_all_frames

    def _dispatch_x_all(self, x: MX) -> tuple:
        """
        This function separate the q value and p value for all the frame
        Parameters
        ----------
        x: MX
            the MX of the solutions which is a 1D vector

        Returns
        -------
        The array of q value for each frame and the 1D vector of parameters
        """
        p = x[-self.nb_parameters_dofs:]
        q = x[:self.nb_kinematic_dofs * self.nb_frames]
        q = q.reshape((self.nb_kinematic_dofs, self.nb_frames), order="F")
        return q, p

    @staticmethod
    def penalty_table(table_markers_mx: MX, table_markers_xp: np.ndarray) -> MX:
        """
        Calculate the penalty cost for table's markers

        Parameters
        ----------
        table_markers_mx : MX
            The position of each marker of the informatic model

        table_markers_xp : MX
            The position of the markers associated with the table, coming from experiment

        Returns
        -------
        MX
        The cost of the penalty function

        """

        return sumsqr(table_markers_mx - table_markers_xp)

    @staticmethod
    def penalty_open_loop_marker(model_markers_mx: MX, model_markers_xp: np.ndarray) -> MX:
        """
        Calculate the penalty cost for wu's markers

        Parameters
        ----------
        model_markers_mx : MX
            The position of each marker of the informatic model

        model_markers_xp : MX
            The position of the markers associated with wu model , coming from experiment

        Returns
        -------
        The cost of the penalty function

        """

        return sumsqr(model_markers_mx - model_markers_xp)

    def penalty_rotation_matrix_cas(self, x: MX) -> MX:
        """
        Calculate the penalty cost for rotation matrix

        Parameters
        ----------
        x : MX
            the entire vector with q and p

        Returns
        -------
        The cost of the penalty function

        """
        rotation_matrix = self.biorbd_model.globalJCS(x, self.biorbd_model.nbSegment() - 1).rot().to_mx()
        rot_matrix_list_model = [
            rotation_matrix[2, 0],
            rotation_matrix[2, 1],
            rotation_matrix[0, 2],
            rotation_matrix[1, 2],
            (rotation_matrix[2, 2] - 1),
        ]

        return sumsqr(vertcat(*rot_matrix_list_model))

    def penalty_q_continuity(self, q_sym: MX, q_init: np.ndarray) -> MX:
        """

        Parameters
        ----------
        q_sym : MX
            the unknown value of q
        q_init : np.ndarray
            value of q coming from either the q_ik_initial_guess for the first frame or the previous solutions

        Returns
        -------
        The cost of the penalty function

        """

        return sumsqr(q_sym - q_init)

    def penalty_theta(self, x: MX) -> MX:
        """
        Calculate the penalty cost for theta angle

        Parameters
        ----------
        x : MX
            the entire vector with q and p

        Returns
        -------
        The cost of the penalty function

        """

        theta_part1_3 = x[-2] + x[-1]
        theta_part1_3_lim = 7 * np.pi / 10

        return if_else(
            theta_part1_3 > theta_part1_3_lim,  # cond
            (theta_part1_3 - theta_part1_3_lim) ** 2,  # if true
            0  # if false
        )

    def objective_param(self):
        """
        Calculate the objective function used to determine parameters, build differently from objective_ik

        Parameters
        ----------

        Returns
        -------
        the value of the objective function for the set of parameters
        """
        # get the position of the markers for the info model
        all_markers_model = self.biorbd_model.markers(self.x_sym)

        # built the objective function by adding penalties one by one

        # first build the table markers symbolics based on q and p
        table_markers_model_sym = [all_markers_model[i] for i in self.table_markers_idx]
        # table_markers_model_sym = vertcat(table_markers_model_sym[0].to_mx(), table_markers_model_sym[0].to_mx()[:2])
        table_markers_model_sym = vertcat(table_markers_model_sym[0].to_mx(), table_markers_model_sym[0].to_mx())
        # second send this to penalty with symbolic experimental array
        obj_closed_loop = self.penalty_table(table_markers_model_sym, self.m_table_sym)

        # first build the other markers symbolics based on q and p
        model_markers_model_sym = vertcat(*[all_markers_model[i].to_mx() for i in self.model_markers_idx])
        # second send this to penalty with symbolic experimental array
        obj_open_loop = self.penalty_open_loop_marker(model_markers_model_sym, self.m_model_sym)

        obj_rotation = self.penalty_rotation_matrix_cas(self.x_sym)

        obj_pivot = self.penalty_theta(self.x_sym)

        output = obj_open_loop * self.weights_param[0] \
                 + obj_rotation * self.weights_param[1] \
                 + obj_pivot * self.weights_param[2] \
 \
        return Function("f", [self.q_sym, self.p_sym, self.m_model_sym, self.m_table_sym], [output],
                        ["q_sym", "p_sym", "markers_model", "markers_table"], ["obj_function"])

    def parameters_optimization(self,
                                q_init_all: np.ndarray,
                                p_init: np.ndarray,
                                ):
        """
        This method return the value of optimal parameters

        Parameters
        ----------
        q_init_all:  np.ndarray
            the MX which contains the solutions found during the initialization
        p_init: np.ndarray
            the value of parameters used at the starting point

        Returns
        -------
        the value of optimal parameters
        """

        obj_func = self.objective_param()
        objective = 0
        start_param = time.time()

        for f in self.list_frames_param_step:
            objective += obj_func(q_init_all[:, f],
                                  self.p_sym,
                                  self.markers[:, self.model_markers_idx, f].flatten("F"),
                                  self.markers[:, self.table_markers_idx, f].flatten("F")[:],
                                  )

        # Create a NLP solver
        prob = {"f": objective, "x": self.p_sym}
        opts = {"ipopt": {"max_iter": 5000, "linear_solver": "ma57"}}
        solver = nlpsol('solver', 'ipopt', prob, opts)  # no constraint yet

        # Solve the NLP
        sol = solver(
            x0=p_init,
            lbx=self.bounds_p_list[0],
            ubx=self.bounds_p_list[1],
        )
        param_opt = sol["x"].full().flatten()

        end_param = time.time()
        self.time_param.append(end_param - start_param)
        return param_opt

    def objective_ik(self) -> Function:
        """
        Calculate the objective function used to determine generalised coordinates, build differently from objective_param

        Returns
        -------
        The Casadi objective function for the ik
        """

        # get the position of the markers for the info model
        all_markers_model = self.biorbd_model.markers(self.x_sym)

        # built the objective function by adding penalties one by one
        # first build the table markers symbolics based on q and p
        table_markers_model_sym = [all_markers_model[i] for i in self.table_markers_idx]
        # table_markers_model_sym = vertcat(table_markers_model_sym[0].to_mx(), table_markers_model_sym[0].to_mx()[:2])
        table_markers_model_sym = vertcat(table_markers_model_sym[0].to_mx(), table_markers_model_sym[0].to_mx())
        # second send this to penalty with symbolic experimental array
        obj_closed_loop = self.penalty_table(table_markers_model_sym, self.m_table_sym)

        # first build the other markers symbolics based on q and p
        model_markers_model_sym = vertcat(*[all_markers_model[i].to_mx() for i in self.model_markers_idx])
        # second send this to penalty with symbolic experimental array
        obj_open_loop = self.penalty_open_loop_marker(model_markers_model_sym, self.m_model_sym)

        obj_rotation = self.penalty_rotation_matrix_cas(self.x_sym)

        obj_q_continuity = self.penalty_q_continuity(self.q_sym,
                                                     self.q_ik_initial_guess[self.q_kinematic_index, self.frame])

        obj_pivot = self.penalty_theta(self.x_sym)

        output = obj_open_loop * self.weights_ik[0] \
                 + obj_rotation * self.weights_ik[1] \
                 + obj_pivot * self.weights_ik[2] \
                 + obj_q_continuity * self.weights_ik[3]

        return Function("f", [self.q_sym, self.p_sym, self.m_model_sym, self.m_table_sym], [output],
                        ["q_sym", "p_sym", "markers_model", "markers_table"], ["obj_function"])

    def inverse_kinematics(self,
                           q_init: np.ndarray,
                           p_init: np.ndarray,
                           ):
        """
        This method complete the index of kinematic value for all frames with the optimal values found
        Parameters
        ----------
        q_init: np.ndarray
            array of value use as initial guess, each column represents one frame, expected to be changed by the end of this step
        p_init: np.ndarray
            array of value use as initial guess, NOT expected to change by the end of this step

        Returns
        -------
        the MX which represent the total x vector for each frame with optimal generalized coordinates and the epsilon
        value btwm each marker
        """
        q_output = np.zeros((self.nb_kinematic_dofs, self.nb_frames_ik_step))
        x_output = np.zeros((self.nb_kinematic_dofs + self.nb_parameters_dofs, self.nb_frames_ik_step))

        obj_func = self.objective_ik()
        start_ik = time.time()

        # enter the frame loop
        for f in range(self.nb_frames_ik_step):
            x_output[self.q_parameter_index, f] = p_init
            self.frame = f

            objective = obj_func(self.q_sym,
                                 p_init,
                                 self.markers[:, self.model_markers_idx, f].flatten("F"),
                                 self.markers[:, self.table_markers_idx, f].flatten("F")[:],

                                 )

            # constraint_func = self.build_constraint_2(q_sym=self.q_sym, p_sym=p_init, f=f)
            constraint_func = self.build_constraint_1()

            # Create a NLP solver
            prob = {"f": objective,
                    "x": self.q_sym,
                    "g": constraint_func(q_sym=self.q_sym, p_sym=p_init)
                    }

            # can add "hessian_approximation":  "limited-memory" ( or "exact") in opts
            if f == 0:
                opts = {"ipopt": {"max_iter": 5000, "linear_solver": "ma57"}}
                solver = nlpsol('solver', 'ipopt', prob, opts)

            # can add "hessian_approximation":  "limited-memory" ( or "exact") in opts
            else:
                opts = {"ipopt": {"max_iter": 5000, "linear_solver": "ma57"}}
                solver = nlpsol('solver', 'ipopt', prob, opts)

            # Solve the NLP
            sol = solver(
                x0=q_init[:, f] if f == 0 else q_output[:, f - 1],
                lbx=self.bounds_q_list[0],
                ubx=self.bounds_q_list[1],
                lbg=self.bound_constraint,
                ubg=self.bound_constraint,
            )

            if solver.stats()['success'] == False:
                print("#########################################################")
                print("#########################################################")
                print("#########################################################")
                print("#########################################################")
                print("--------------   IT DID NOT CONVERGE   ------------------")
                print("#########################################################")
                print("#########################################################")
                print("#########################################################")
                print("#########################################################")

            q_output[:, f] = sol["x"].toarray().squeeze()
            x_output[self.q_kinematic_index, f] = sol["x"].toarray().squeeze()

            markers_model = self.biorbd_model_eigen.markers(x_output[:, f])
            markers_to_compare = self.markers[:, :, f]
            espilon_markers = 0

            # sum of squared norm of difference of markers
            c = 0
            for j in range(self.nb_markers):
                mark = np.linalg.norm(markers_model[j].to_array() - markers_to_compare[:, j]) ** 2
                espilon_markers += mark
                c += 1

        end_ik = time.time()
        self.time_ik.append(end_ik - start_ik)
        return q_output, espilon_markers

    def objective_ik_1step(self):
        """
        This function determine the value of the objective function used in _solve_1step method by adding penalties

        Returns
        -------
        The value of the objective function
        """

        # get the position of the markers for the info model
        all_markers_model = self.biorbd_model.markers(self.x_sym)
        # all_markers_model = self.biorbd_model.markers(self.x_sym2)

        # first build the other markers symbolics based on q and p
        model_markers_model_sym = vertcat(*[all_markers_model[i].to_mx() for i in self.model_markers_idx])
        # second send this to penalty with symbolic experimental array
        obj_open_loop = self.penalty_open_loop_marker(model_markers_model_sym, self.m_model_sym)

        # obj_rotation = self.penalty_rotation_matrix_cas(self.x_sym2)
        obj_rotation = self.penalty_rotation_matrix_cas(self.x_sym)

        # obj_pivot = self.penalty_theta(self.x_sym2)
        obj_pivot = self.penalty_theta(self.x_sym)

        obj_q_continuity = self.penalty_q_continuity(self.q_sym, self.q_sym_prev)
        # if self.frame == 0:
        #     output = obj_open_loop * self.weights_ik[0] \
        #              + obj_rotation * self.weights_ik[1] \
        #              + obj_pivot * self.weights_ik[2] \
        #
        # else:
        #     output = obj_open_loop * self.weights_ik[0] \
        #              + obj_rotation * self.weights_ik[1] \
        #              + obj_pivot * self.weights_ik[2] \
        #              + obj_q_continuity * self.weights_ik[3]

        output = obj_open_loop * self.weights_ik[0] \
                 + obj_rotation * self.weights_ik[1] \
                 + obj_pivot * self.weights_ik[2] \
                 + obj_q_continuity * self.weights_ik[3]

        return Function("f", [self.q_sym, self.p_sym, self.q_sym_prev, self.m_model_sym], [output],
                        ["q_sym", "p_sym", "q_sym_prev", "markers_model"], ["obj_function"])

    def build_constraint_1(self):
        """
        This function build the constraint for closed loop markers ie the Table

        Returns
        -------
        a MX with the distance btwm each marker associated w/ the closed loop ie the Table
        """

        table_markers1_model = self.biorbd_model.markers(self.x_sym)[self.table_markers_idx[0]].to_mx()
        table_markers2_model = self.biorbd_model.markers(self.x_sym)[self.table_markers_idx[1]].to_mx()
        table_markers_table = vertcat(table_markers1_model, table_markers2_model)
        table_markers_xp = self.markers[:, self.table_markers_idx, self.frame].flatten("F")
        diff = table_markers_table - table_markers_xp

        return Function("g", [self.q_sym, self.p_sym], [diff],
                        ["q_sym", "p_sym"], ["constraint_func"])

    # old method but it's work

    # def build_constraint_2(self, q_sym, p_sym, f):
    #     """
    #     This function build the constraint for closed loop markers ie the Table
    #
    #     Parameters
    #     ----------
    #     q_sym : MX
    #         q symbolic vector
    #     p_sym : MX
    #         p symbolic vector
    #     f : int
    #         the number of the frame
    #
    #     Returns
    #     -------
    #      a MX with the distance btwm each marker associated w/ the closed loop ie the Table
    #     """
    #     x_sym = MX.zeros(22)
    #     x_sym[self.q_kinematic_index] = q_sym
    #     x_sym[self.q_parameter_index] = p_sym
    #
    #
    #     table_markers1_model = self.biorbd_model.markers(x_sym)[self.table_markers_idx[0]].to_mx()
    #     table_markers2_model = self.biorbd_model.markers(x_sym)[self.table_markers_idx[1]].to_mx()
    #     table_markers_table = vertcat(table_markers1_model, table_markers2_model)
    #     table_markers_xp = self.markers[:, self.table_markers_idx, f].flatten("F")
    #     diff = table_markers_table - table_markers_xp    # MX (6x1)
    #
    #     return diff

    def solve(
            self,
            threshold,
            method,
    ):
        """
        This function returns the solutions using the method chosen by the user

        Parameters
        ----------
        threshold : int
            the threshold for the delta epsilon
        method : str
            the method used to find the optimised generalized coordinates:
            - "1step": the global solutions ( for all the frame ) is found without loop, parameters and generalized
            coordinates values are determined together
            - "2step": the global solutions is build frame after frame where the parameters value (the same for each frame)
            are firstly found and generalized coordinates after that.

        Return
        ------
            The optimized Generalized coordinates and parameters
        """
        if method == "1step":
            return self._solve_1step()
        elif method == "2step":
            return self._solve_2step(threshold)
        else:
            raise NotImplementedError("This is not implemented, please use 1step or 2step")

    def _solve_1step(self):
        """
        This function find the entire solutions with only 1 step ie without a while loop unlike 2step

        Returns
        -------
        q_all_frame, value of generalised coordinates for all frames,
        param_opt, the value of the parameters,
        x_all_frames, the entire solutions for all frames
        """
        print(" | You choose 1_step |")
        start_ik = time.time()
        x_init = np.concatenate((self.q_ik_initial_guess.flatten("F"), self.p_ik_initial_guess))

        obj_func = self.objective_ik_1step()
        constraint_func = self.build_constraint_1()
        objective = 0
        constraint_list = []

        for f in range(self.nb_frames):
            self.frame = f
            objective += obj_func(
                self.q_sym_global[:, f],
                self.p_sym,
                self.q_sym_global[:, f - 1] if f != 0 else self.q_ik_initial_guess[:, 0],
                self.markers[:, self.model_markers_idx, f].flatten("F"),
            )

            constraint_list.append(constraint_func(q_sym=self.q_sym_global[:, f], p_sym=self.p_sym)["constraint_func"])

        constraint_func = vertcat(*constraint_list)

        # Create a NLP solver
        prob = {"f": objective,
                "x": vertcat(self.q_sym_global_vec, self.p_sym),
                "g": constraint_func}

        opts = {"ipopt": {"max_iter": 5000, "linear_solver": "ma57"}}
        solver = nlpsol('solver', 'ipopt', prob, opts)

        # Solve the NLP
        sol = solver(
            x0=x_init,
            lbx=np.concatenate((self.bounds_q_list[0].__mul__(self.nb_frames), self.bounds_p_list[0])),
            ubx=np.concatenate((self.bounds_q_list[1].__mul__(self.nb_frames), self.bounds_p_list[1])),
            lbg=self.bound_constraint.repeat(self.nb_frames),
            ubg=self.bound_constraint.repeat(self.nb_frames),
        )

        end_ik = time.time()
        x_output = sol["x"].toarray().squeeze()
        q_all_frame, param_opt = self._dispatch_x_all(x_output)
        self.q_all_frame = q_all_frame
        self.parameters = param_opt
        x_all_frames = self.build_x_all_frames()
        self.x_all_frames = x_all_frames
        # x_output = np.delete(x_output, [-6, -5, -4, -3, -2, -1])
        # # q_all_frame = x_output.reshape(self.nb_kinematic_dofs, self.nb_frames)
        # q_all_frame = np.zeros((self.nb_kinematic_dofs, self.nb_frames))
        # x_all_frames = np.zeros((self.nb_total_dofs, self.nb_frames))
        # for i in range(self.nb_frames):
        #     q_all_frame[:, i] = x_output[self.nb_kinematic_dofs * i : self.nb_kinematic_dofs * (i+1)]
        #     x_all_frames[self.q_kinematic_index, i] = q_all_frame[:, i]
        #     x_all_frames[self.q_parameter_index, i] = param_opt

        # print("x_all_frames = ", x_all_frames)
        # q_all_frame = x_all_frames[self.q_kinematic_index, :]
        # param_opt = x_all_frames[self.q_parameter_index, 2]

        self.time_ik.append(end_ik - start_ik)
        return q_all_frame, param_opt, x_all_frames

    def _solve_2step(
            self,
            threshold: int = 5e-3,
    ):
        """
        This function returns optimised generalized coordinates and the epsilon difference

        Parameters
        ----------
        threshold : int
            the threshold for the delta epsilon

        Return
        ------
            The optimized Generalized coordinates and parameters
        """
        print(" | You choose 2_step |")
        # get the q_bounds of the model for all dofs
        bounds = [
            (mini, maxi) for mini, maxi in zip(get_range_q(self.biorbd_model)[0], get_range_q(self.biorbd_model)[1])
        ]

        # find kinematic dof with initial guess at zeros
        idx_zeros = np.where(np.sum(self.q_ik_initial_guess, axis=1) == 0)[0]
        kinematic_idx_zeros = [idx for idx in idx_zeros if idx in self.q_kinematic_index]

        # initialize q_ik with in the half-way between q_bounds
        bounds_kinematic_idx_zeros = [b for i, b in enumerate(bounds) if i in kinematic_idx_zeros]
        kinova_q0 = np.array([(b[0] + b[1]) / 2 for b in bounds_kinematic_idx_zeros])

        # initialized q trajectories for each frames for dofs without a priori knowledge of the q (kinova arm here)
        self.q_ik_initial_guess[kinematic_idx_zeros, :] = np.repeat(
            kinova_q0[:, np.newaxis], self.nb_frames_ik_step, axis=1
        )

        # initialized q qnd p for the whole algorithm.

        p_init_global = np.zeros(self.nb_parameters_dofs)
        q_init_global = self.q_ik_initial_guess[self.q_kinematic_index, :]

        print(" #######  Initialisation beginning  ########")

        # First IK step - INITIALIZATION

        q_all_frames = self.inverse_kinematics(
            q_init=q_init_global,
            p_init=p_init_global,
        )[0]

        print(" #######  Initialisation ending ########")

        p_init = p_init_global

        iteration = 0
        epsilon_markers_n = 10  # arbitrary set
        epsilon_markers_n_minus_1 = 0
        delta_epsilon_markers = epsilon_markers_n - epsilon_markers_n_minus_1

        print("#####   Starting the while loop   #####")

        while fabs(delta_epsilon_markers) > threshold:
            epsilon_markers_n_minus_1 = epsilon_markers_n

            param_opt = self.parameters_optimization(
                q_init_all=q_all_frames,
                p_init=p_init,
            )
            print(" param opt =", param_opt)

            # step 2 - ik step
            q_all_frames, epsilon_markers_n = self.inverse_kinematics(
                q_init=q_all_frames,
                p_init=param_opt,
            )
            delta_epsilon_markers = epsilon_markers_n - epsilon_markers_n_minus_1
            print("delta_epsilon_markers:", delta_epsilon_markers)
            iteration += 1
            print("iteration:", iteration)

        print("#####   Leaving the while loop   #####")

        self.parameters = param_opt
        self.q = q_all_frames
        x_all_frames = np.zeros((self.nb_total_dofs, self.nb_frames))
        for f in range(self.nb_frames):
            x_all_frames[self.q_kinematic_index, f] = q_all_frames[:, f]  # can't broadcast
            x_all_frames[self.q_parameter_index, f] = param_opt[:]
        self.x_all_frames = x_all_frames

        return q_all_frames, param_opt, x_all_frames

    def frame_selector(self, frames_needed: int, frames: int):
        """
        Give a list of frames for calibration

        Parameters
        ----------
        frames_needed: int
            The number of random frames you need
        frames: int
            The total number of frames

        Returns
        -------
        list_frames: list[int]
            The list of frames use for calibration
        """
        list_frames = random.sample(range(frames), frames_needed)  # if not all else [i for i in range(frames)]

        list_frames.sort()

        return list_frames

    def solution(self):

        """
         This function returns a dictionnary which contains the global RMS and the RMS for each axes

         Parameters
         ----------

         Return
         ------
         the dictionnary with RMS
         """

        residuals_xyz_model = np.zeros((3, self.nb_markers_model, self.nb_frames))
        residuals_xyz_table = np.zeros((3, self.nb_markers_table, self.nb_frames))

        # for each frame
        for f in range(self.nb_frames):

            xi = self.x_all_frames[:, f]
            # get the marker's coordinates of the frame coming from xp
            markers_model = self.biorbd_model_eigen.markers(xi)

            # create a vector corresponding to model coordinates
            vect_pos_markers = np.zeros(3 * len(markers_model))
            for m, value in enumerate(markers_model):
                vect_pos_markers[m * 3: (m + 1) * 3] = value.to_array()

            # get coordinates for model and xp markers of the thorax , without the table
            marker_mod = vect_pos_markers[:(3 * self.nb_markers_model)]
            marker_xp = self.markers[:, self.model_markers_idx, f].flatten("F")

            marker_mod = np.asarray(marker_mod)
            marker_xp = np.asarray(marker_xp)

            # get coordinates of the table's markers coming from xp and model

            table_mod = vect_pos_markers[(3 * self.nb_markers_model):]
            table_xp = self.markers[:, self.table_markers_idx, f].flatten("F")

            table_mod = np.asarray(table_mod)
            table_xp = np.asarray(table_xp)

            array_residual_model = (marker_mod - marker_xp)
            array_residual_table = (table_mod - table_xp)
            residuals_xyz_model[:, :, f] = array_residual_model.reshape(3, self.nb_markers_model, order='F')
            residuals_xyz_table[:, :, f] = array_residual_table.reshape(3, self.nb_markers_table, order='F')

        residuals_norm_model = np.linalg.norm(residuals_xyz_model, axis=0)
        rmse_tot_model = np.sqrt(np.square(residuals_norm_model).mean(axis=0))
        rmse_x_model = np.sqrt(np.square(residuals_xyz_model[0, :, :]).mean(axis=0))
        rmse_y_model = np.sqrt(np.square(residuals_xyz_model[1, :, :]).mean(axis=0))
        rmse_z_model = np.sqrt(np.square(residuals_xyz_model[2, :, :]).mean(axis=0))

        residuals_norm_table = np.linalg.norm(residuals_xyz_table, axis=0)
        rmse_tot_table = np.sqrt(np.square(residuals_norm_table).mean(axis=0))
        rmse_x_table = np.sqrt(np.square(residuals_xyz_table[0, :, :]).mean(axis=0))
        rmse_y_table = np.sqrt(np.square(residuals_xyz_table[1, :, :]).mean(axis=0))
        rmse_z_table = np.sqrt(np.square(residuals_xyz_table[2, :, :]).mean(axis=0))

        self.output = dict(
            rmse_x=rmse_x_model,
            rmse_y=rmse_y_model,
            rmse_z=rmse_z_model,
            rmse_tot=rmse_tot_model,
            rmse_x_table=rmse_x_table,
            rmse_y_table=rmse_y_table,
            rmse_z_table=rmse_z_table,
            rmse_tot_table=rmse_tot_table,

        )

        return self.output

    def plot_graph_rmse(self):
        """

        Returns
        -------
        This function plot the graph of the total RMSE as well as RMSE for x, y and z direction for open loop markers
        """

        dict_rmse = self.output.values()
        nb_frames = len(dict_rmse.mapping["rmse_x"])

        plt.grid(True)
        plt.title("Armpit")
        plt.plot([p for p in range(nb_frames)], dict_rmse.mapping["rmse_x"], "b", label="RMS_x")
        plt.plot([p for p in range(nb_frames)], dict_rmse.mapping["rmse_y"], "y", label="RMS_y")
        plt.plot([p for p in range(nb_frames)], dict_rmse.mapping["rmse_z"], "g", label="RMS_z")
        plt.plot([p for p in range(nb_frames)], dict_rmse.mapping["rmse_tot"], "r", label="RMS_tot")
        plt.xlabel('Frame')
        plt.ylabel('Valeurs (m)')
        plt.legend()
        plt.show()

    def plot_graph_rmse_table(self):
        """

        Returns
        -------
        This function plot the graph of the total RMSE as well as RMSE for x, y and z direction for closed loop markers
        """
        dict_rmse = self.output.values()
        nb_frames = len(dict_rmse.mapping["rmse_x_table"])

        plt.grid(True)
        plt.title("Armpit")
        plt.plot([p for p in range(nb_frames)], dict_rmse.mapping["rmse_x_table"], "b", label="RMS_x_table")
        plt.plot([p for p in range(nb_frames)], dict_rmse.mapping["rmse_y_table"], "y", label="RMS_y_table")
        plt.plot([p for p in range(nb_frames)], dict_rmse.mapping["rmse_z_table"], "g", label="RMS_z_table")
        plt.plot([p for p in range(nb_frames)], dict_rmse.mapping["rmse_tot_table"], "r", label="RMS_tot_table")
        plt.xlabel('Frame')
        plt.ylabel('Valeurs (m)')
        plt.legend()
        plt.show()

    def plot_rotation_matrix_penalty(self):
        """

        Returns
        -------
        This function plot the graph of the value in the matrix rotation for the q found
        """
        rotation_value = []
        for i in range(self.nb_frames):
            rotation_matrix = self.biorbd_model_eigen.globalJCS(self.x_all_frames[:, i],
                                                                self.biorbd_model.nbSegment() - 1).rot().to_array()
            rot_matrix_list_model = [
                rotation_matrix[2, 0],
                rotation_matrix[2, 1],
                rotation_matrix[0, 2],
                rotation_matrix[1, 2],
                (rotation_matrix[2, 2] - 1),
            ]
            rotation_value.append(rot_matrix_list_model)
        plt.figure("rotation_value")

        Rot_20_list = [rotation_value[g][0] for g in range(self.nb_frames)]
        Rot_21_list = [rotation_value[g][1] for g in range(self.nb_frames)]
        Rot_02_list = [rotation_value[g][2] for g in range(self.nb_frames)]
        Rot_12_list = [rotation_value[g][3] for g in range(self.nb_frames)]
        Rot_22_list = [rotation_value[g][4] for g in range(self.nb_frames)]

        plt.scatter([j for j in range(self.nb_frames)], Rot_20_list, marker="x", color="b", label="Rot_20")
        plt.scatter([j for j in range(self.nb_frames)], Rot_21_list, marker="o", color="g", label="Rot_21")
        plt.scatter([j for j in range(self.nb_frames)], Rot_02_list, marker="x", color="y", label="Rot_02")
        plt.scatter([j for j in range(self.nb_frames)], Rot_12_list, marker="o", color="m", label="Rot_12")
        plt.scatter([j for j in range(self.nb_frames)], Rot_22_list, marker="x", color="r", label="Rot_22")

        plt.xlabel(" frame")
        plt.ylabel("value in the rotation matrix")
        plt.legend()
        plt.show()

    def plot_pivot(self):
        pivot_value_list = []
        for f in range(self.nb_frames):
            if self.x_all_frames[-2, f] + self.x_all_frames[-1, f] > 7 * np.pi / 10:
                pivot_value_list.append(self.x_all_frames[-2, f] + self.x_all_frames[-1, f])
            else:
                pivot_value_list.append(0)
        index_not_zero = []
        for h in pivot_value_list:
            if h != 0:
                index_not_zero.append(pivot_value_list.index(h))

        fig, ax = plt.subplots()
        ax.bar([k for k in range(self.nb_frames)], pivot_value_list)
        plt.plot([k for k in range(self.nb_frames)], [(7 * np.pi / 10) for i in range(self.nb_frames)], color="g")
        ax.set_ylabel("value")
        ax.set_xlabel("frame")
        ax.set_title("plot_pivot value")
        plt.show()

        print("index where plot_pivot value is not 0 =", index_not_zero)

    def plot_param_value(self):
        bound_param = self.bounds_p_list
        param_value = self.parameters
        for i in range(self.nb_parameters_dofs):
            if param_value[i] == bound_param[0][i] or param_value[i] == bound_param[1][0]:
                print("parameters number %r reach a bound value " % i)
        plt.figure("param value")
        plt.plot([k for k in range(self.nb_parameters_dofs)],
                 [bound_param[0][u] for u in range(self.nb_parameters_dofs)], label="lower bound")
        plt.plot([k for k in range(self.nb_parameters_dofs)],
                 [bound_param[1][u] for u in range(self.nb_parameters_dofs)], label="upper bound")
        plt.plot([k for k in range(self.nb_parameters_dofs)], param_value, label="parameters values")
        plt.xlabel(" number of parameter")
        plt.ylabel("value of parameter")
        plt.legend()
        plt.show()
        print("parameters values = ", param_value)
