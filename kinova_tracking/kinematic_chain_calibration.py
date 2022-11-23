from typing import Union, Tuple
from enum import Enum
import random
import time

import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
from casadi import MX, Function, vertcat, nlpsol, if_else, norm_2, sumsqr, fabs, qpsol

from casadi import MX, Function, vertcat, nlpsol, if_else, norm_2, sumsqr

import biorbd_casadi as biorbd
import biorbd as biorbd_eigen

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
            markers: np.array,  # [3 x nb_markers x nb_frames]
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
            # same_variables: dict = None,
    #       # coupling = dict("name_of_the_main_dof":"name_of_the_other_dof")
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
        self. x_ik_initial_guess = x_ik_initial_guess
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
        # nb marker table * 3 dim - 1 because we don't use value on z for Table:Table6

        weight_open_loop_ls = [self.weights_param[1]] * (
                len([i for i in self.tracked_markers if i not in self.closed_loop_markers]) * 3
        )
        weight_open_loop_ipopt = [self.weights_ik[1]] * (
                len([i for i in self.tracked_markers if i not in self.closed_loop_markers]) * 3
        )
        # This is for all markers except those for table

        weight_rot_matrix_ls = [self.weights_param[4]] * 5  # len(rot_matrix_list_xp)
        weight_rot_matrix_ipopt = [self.weights_ik[4]] * 5  # len(rot_matrix_list_xp)

        weight_theta_13_ls = [self.weights_param[2]]
        weight_theta_13_ipopt = [self.weights_ik[2]]

        weight_continuity_ls = [self.weights_param[3]] * (self.q_ik_initial_guess.shape[0] - len(self.parameter_dofs))
        weight_continuity_ipopt = [self.weights_ik[3]] * (self.q_ik_initial_guess.shape[0] - len(self.parameter_dofs))
        # We need the nb of dofs but without parameters

        self.weight_list_param = weight_closed_loop_ls + weight_open_loop_ls + weight_continuity_ls + weight_theta_13_ls + weight_rot_matrix_ls
        self.weight_list_ik = weight_closed_loop_ipopt + weight_open_loop_ipopt + weight_continuity_ipopt + weight_theta_13_ipopt + weight_rot_matrix_ipopt

        self.list_sol = []
        self.q = np.zeros((self.biorbd_model.nbQ(), self.nb_frames_ik_step))
        #self.parameters = np.zeros(self.nb_parameters_dofs)
        self.segment_id_with_vertical_z = segment_id_with_vertical_z
        self.output = dict()

        # symbolic variables
        self.q_sym = MX.sym("q_sym", self.nb_kinematic_dofs)
        self.q_sym_prev = MX.sym("q_sym_prev", self.nb_kinematic_dofs)
        self.q_sym_global_vec = MX.sym("q_sym_global", self.nb_kinematic_dofs * self.nb_frames)
        self.q_sym_global = self.q_sym_global_vec.reshape((self.nb_kinematic_dofs, self.nb_frames))

        self.p_sym = MX.sym("p_sym", self.nb_parameters_dofs)
        self.q_frames_sym = MX.sym("q_sym", (self.nb_kinematic_dofs, self.nb_frames_param_step))

        # synbolic x with q qnd p variables
        self.x_sym = MX.zeros(self.nb_total_dofs)
        self.x_sym_prev = MX.zeros(self.nb_total_dofs)
        self.x_sym = self.build_x(self.q_sym, self.p_sym)
        self.x_sym_prev = self.build_x(self.q_sym_prev, self.p_sym)
        # self.x_sym[self.q_kinematic_index] = self.q_sym
        # self.x_sym[self.q_parameter_index] = self.p_sym

        self.x_sym2 = MX.sym("x_sym2", self.nb_total_dofs) # todo: without it plz
        # self.x_sym
        #self.q_sym_global = MX.sym("q_sym_global", self.nb_total_dofs * self.nb_frames) #
        # self.q_sym_global = MX.sym("q_sym_global", self.nb_kinematic_dofs * self.nb_frames + self.nb_parameters_dofs)


        # symbolic xp data
        self.m_model_sym = MX.sym("markers_model", self.nb_markers_model * 3)
        # self.m_table_sym = MX.sym("markers_table", 5)  # xyz and xy
        self.m_table_sym = MX.sym("markers_table", 6)  # xyz and xyz

        # symbolic frame
        #self.f_sym = MX.sym("f_sym" , 1)
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
        The array of solution
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
            the MX of the solution which is a 1D vector

        Returns
        -------
        The array of q value for each frame and the 1D vector of parameters
        """
        p = x[-self.nb_parameters_dofs:]
        q = x[:self.nb_kinematic_dofs * self.nb_frames]
        q = q.reshape((self.nb_kinematic_dofs, self.nb_frames), order="F")
        return q, p

    # Function = ("markers_residuals", [self.q_sym, self.p_sym], [])
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
                     self.table_markers_idx[0] * 3: self.table_markers_idx[0] * 3 + 3][:]
        table_xp = table_markers_xp[:, 0].tolist()

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
        for j, name in enumerate(self.markers_model[:self.model_markers_idx[-1]+1]):

            pos_model_mx = model_markers_mx[self.markers_model.index(name) * 3: self.markers_model.index(name) * 3 + 3]
            pos_xp = model_markers_xp[:, self.markers_model.index(name)].tolist()

            for i in range(len(pos_xp)):
                diff += (pos_model_mx[i] - pos_xp[i]) ** 2

        return diff

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
        diff = MX.zeros(1)
        rotation_matrix = self.biorbd_model.globalJCS(x, self.biorbd_model.nbSegment() - 1).rot().to_mx()
        rot_matrix_list_model = [
            rotation_matrix[2, 0],
            rotation_matrix[2, 1],
            rotation_matrix[0, 2],
            rotation_matrix[1, 2],
            (rotation_matrix[2, 2] - 1),
        ]

        return sumsqr(vertcat(*rot_matrix_list_model))
        for i in rot_matrix_list_model:
            diff += i**2

    def penalty_q_continuity(self, q_sym: MX, q_init: np.ndarray) -> MX:
        """

        Parameters
        ----------
        q_sym : MX
            the unknown value of q
        q_init : np.ndarray
            value of q coming from either the q_ik_initial_guess for the first frame or the previous solution

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
            (theta_part1_3-theta_part1_3_lim) ** 2,  # if true
            0  # if false
        )

    # def objective_param(self, x_step_2: MX, q_sym: MX, p_sym: MX):
    def objective_param(self):
        """
        Calculate the objective function used to determine parameters, build differently from objective_ik

        Parameters
        ----------

            the MX which contains the solution found during the initialization
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

        #obl_q_continuity = self.penalty_q_continuity(q_sym, )
        obj_pivot = self.penalty_theta(self.x_sym)

        # output = obj_closed_loop * self.weights_param[0]\
        #          + obj_open_loop * self.weights_param[1]\
        #          + obj_rotation * self.weights_param[2]\
        #          + obj_pivot * self.weights_param[3]\

        output = obj_open_loop * self.weights_param[0]\
                 + obj_rotation * self.weights_param[1]\
                 + obj_pivot * self.weights_param[2]\

        return Function("f", [self.q_sym, self.p_sym, self.m_model_sym, self.m_table_sym], [output],
                        ["q_sym", "p_sym", "markers_model", "markers_table"], ["obj_function"])



        # be filled in the loop
        # s = MX.zeros(1)
        #
        # for f in self.list_frames_param_step:
        #     table_markers_xp = self.markers[:, self.table_markers_idx, f]
        #     model_markers_xp = self.markers[:, self.model_markers_idx, f]
        #
        #     q_sym[:] = x_step_2[self.q_kinematic_index, f]
        #     self.x[self.q_kinematic_index] = q_sym
        #     self.x[self.q_parameter_index] = p_sym
        #
        #     markers_model = self.biorbd_model.markers(self.x)
        #
        #     vect_pos_markers = MX.zeros(3 * len(markers_model))
        #
        #     for m, value in enumerate(markers_model):
        #         vect_pos_markers[m * 3: (m + 1) * 3] = value.to_mx()
        #
        #     obj_closedloop = self.penalty_table(vect_pos_markers, table_markers_xp)
        #     obj_openloop = self.penalty_open_loop_marker(vect_pos_markers, model_markers_xp)
        #     obj_rotation = self.penalty_rotation_matrix_cas(self.x)
        #     obj_theta = self.penalty_theta(self.x)
        #     output_f = obj_closedloop + obj_openloop + obj_rotation + obj_theta
        #     s += output_f
        #
        # obj = Function("f", [p_sym], [s])
        # return obj
    def parameters_optimization(self,
                                q_init_all: np.ndarray,
                                p_init: np.ndarray,
                                ):
        """
        This method return the value of optimal parameters

        Parameters
        ----------
        q_init_all:  np.ndarray
            the MX which contains the solution found during the initialization
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

        #obj_func = self.objective_param()
        # F(q_sym_all_frames, p_sym, markers_model_sym, table_sym)
        #objective = obj_func(self.p_sym)

        # Create a NLP solver
        prob = {"f": objective, "x": self.p_sym}
        opts = {"ipopt":{"max_iter": 5000, "linear_solver": "ma57"}}
        solver = nlpsol('solver', 'ipopt', prob, opts)  # no constraint yet

        # Solve the NLP
        sol = solver(
            x0=p_init,
            lbx=self.bounds_p_list[0],
            ubx=self.bounds_p_list[1],
        )
        param_opt = sol["x"].full().flatten()

        end_param = time.time()
        self.time_param.append(end_param-start_param)
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

        obj_q_continuity = self.penalty_q_continuity(self.q_sym, self.q_ik_initial_guess[self.q_kinematic_index, self.frame])

        obj_pivot = self.penalty_theta(self.x_sym)

        # output = obj_closed_loop * self.weights_ik[0]\
        #             + obj_open_loop * self.weights_ik[1]\
        #             + obj_rotation * self.weights_ik[2]\
        #             + obj_pivot * self.weights_ik[3]\
        output = obj_open_loop * self.weights_ik[0]\
                + obj_rotation * self.weights_ik[1]\
                + obj_pivot * self.weights_ik[2]\
                + obj_q_continuity * self.weights_ik[3]

        return Function("f", [self.q_sym, self.p_sym, self.m_model_sym, self.m_table_sym], [output], ["q_sym", "p_sym", "markers_model", "markers_table"], ["obj_function"])

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
        #x_output[self.q_parameter_index, :] = p_init  # broadcasting p_init
        # for f in range(self.nb_frames):
        #     x_output[self.q_parameter_index, f] = p_init

        obj_func = self.objective_ik()
        start_ik = time.time()
        #constraint_func = self.build_constraint()

        # enter the frame loop
        for f in range(self.nb_frames_ik_step):
            x_output[self.q_parameter_index, f] = p_init
            self.frame = f


            objective = obj_func(self.q_sym,
                                 p_init,
                                 self.markers[:, self.model_markers_idx, f].flatten("F"),
                                 self.markers[:, self.table_markers_idx, f].flatten("F")[:],

                                 )

            #                              p_init,
            #                              self.markers[:, self.table_markers_idx, f].flatten("F")[:-1])
            constraint_func = self.build_constraint_2(q_sym=self.q_sym, p_sym=p_init, f=f)

            # Create a NLP solver
            prob = {"f": objective, "x": self.q_sym, "g": constraint_func}

            # can add "hessian_approximation":  "limited-memory" ( or "exact") in opts
            # val = 1e-3
            # tol = val
            # compl_inf_tol = val
            # acceptable_tol = val
            # acceptable_compl_inf_tol = val
            #hessian_approximation: str = "limited-memory"  # "exact", "limited-memory"
            # if f == 0:
            #     opts = {"ipopt": {"max_iter": 5000, "linear_solver": "ma57"}}
            # else:
            #     opts = {"ipopt": {"max_iter": 5000, "linear_solver": "ma57",
            #                       "hessian_approximation": hessian_approximation}}
                              # "tol": tol,"compl_inf_tol": compl_inf_tol,
                              #"acceptable_tol": acceptable_tol, "acceptable_compl_inf_tol": acceptable_compl_inf_tol}}
            #opts = dict(qpsol="qpoases")
            #solver = nlpsol('solver', 'ipopt', prob, opts)
            if f == 0:
                opts = {"ipopt": {"max_iter": 5000, "linear_solver": "ma57"}}
                solver = nlpsol('solver', 'ipopt', prob, opts)

            # can add "hessian_approximation":  "limited-memory" ( or "exact") in opts
            else:
                opts = {"ipopt": {"max_iter": 5000, "linear_solver": "ma57"}}
                solver = nlpsol('solver', 'ipopt', prob, opts)



                # opts = dict(qpsol="qpoases",
                #             error_on_fail=False,
                #             )
                # solver = nlpsol('solver', 'sqpmethod', prob, opts)



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
                mark = np.linalg.norm(markers_model[j].to_array()- markers_to_compare[:, j]) ** 2
                espilon_markers += mark
                c += 1

        end_ik = time.time()
        self.time_ik.append(end_ik-start_ik)
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

        # x_sym global is MX (22x nb frames) x_ik_initial guess is array (22x nb frames)
        # obj_q_continuity = self.penalty_q_continuity(self.q_sym_global[:, self.frame],
        #                                              self.q_ik_initial_guess[:, self.frame] if self.frame == 0 else self.q_sym_global[:, self.frame-1])

        obj_q_continuity = self.penalty_q_continuity(self.q_sym, self.q_sym_prev)

        output = obj_open_loop * self.weights_ik[0] \
                 + obj_rotation * self.weights_ik[1] \
                 + obj_pivot * self.weights_ik[2] \
                 + obj_q_continuity * self.weights_ik[3]

        return Function("f", [self.q_sym, self.p_sym, self.q_sym_prev, self.m_model_sym], [output],
        #                        ["x_sym", "markers_model"], ["obj_function"])
        return Function("f", [self.q_sym, self.p_sym, self.q_sym_prev, self.m_model_sym], [output],   #block here x(0) isn't symbolic
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

    def build_constraint_2(self, q_sym, p_sym, f):
        """
        This function build the constraint for closed loop markers ie the Table

        Parameters
        ----------
        q_sym : MX
            q symbolic vector
        p_sym : MX
            p symbolic vector
        f : int
            the number of the frame

        Returns
        -------
         a MX with the distance btwm each marker associated w/ the closed loop ie the Table
        """
        # todo call build_x in the parent function.
        x_sym = MX.zeros(22)
        x_sym[self.q_kinematic_index] = q_sym
        x_sym[self.q_parameter_index] = p_sym
        table_markers1_model = self.biorbd_model.markers(x_sym)[self.table_markers_idx[0]].to_mx()
        table_markers2_model = self.biorbd_model.markers(x_sym)[self.table_markers_idx[1]].to_mx()
        table_markers_table = vertcat(table_markers1_model, table_markers2_model)
        table_markers_xp = self.markers[:, self.table_markers_idx, f].flatten("F")
        diff = table_markers_table - table_markers_xp    # MX (6x1)

        return diff


    def solve(
            self,
            threshold,
            method,
    ):
        """
        This function returns the solution using the method chosen by the user

        Parameters
        ----------
        threshold : int
            the threshold for the delta epsilon
        method : str
            the method used to find the optimised generalized coordinates
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
        This function find the entire solution with only 1 step ie without a while loop unlike 2step

        Returns
        -------
        q_all_frame, value of generalised coordinates for all frames,
        param_opt, the value of the parameters,
        x_all_frames, the entire solution for all frames
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
            # x = self.build_x(q=self.q_sym_global[self.nb_kinematic_dofs * f: self.nb_kinematic_dofs * (f + 1)],
            #                  p=self.q_sym_global[-6:])
            # x = self.build_x(q=self.q_sym_global[:, f],
            #                  p=self.p_sym)
            # objective += obj_func(x,
            #                       self.markers[:, self.model_markers_idx, f].flatten("F"),
            #                       )

            objective += obj_func(
                self.q_sym_global[:, f],
                self.p_sym,
                self.q_sym_global[:, f - 1] if f != 0 else self.q_ik_initial_guess[:, 0],
                self.markers[:, self.model_markers_idx, f].flatten("F"),
            )

            # We built the constraint function using build_constraint1 which returns a 6 by 1 vector, we vertcat each of
            # them as it's the only way to be understood by the solver

            # constraint_list.append(self.build_constraint_1(x_sym=self.x_sym, f=f))
            constraint_list.append(constraint_func(q_sym=self.q_sym_global[:, f], p_sym=self.p_sym)["constraint_func"])

            #     constraint_func1 = self.build_constraint_1(x_sym=self.q_sym_global[22 * f: 22 * (f+1)], f=f)
            # else:
            #     constraint_func2 = self.build_constraint_1(x_sym=self.q_sym_global[22 * f: 22 * (f+1)], f=f)
            # if f != 0:
            #     constraint_func = vertcat(constraint_func1, constraint_func2)
            #     constraint_func1 = constraint_func
        constraint_func = vertcat(*constraint_list)

        # a = vertcat(self.q_sym_global[:, 0], self.q_sym_global[:, 1])
        # for i in range(2, self.nb_frames, 1):
        #     a = vertcat(a, self.q_sym_global[:, i])
        # x = vertcat(a, self.p_sym)
        # a = MX.sym("a", 96)
        # for i in range(6):
        #     for g in range(16):
        #         a[16 * i + g, ] = self.q_sym_global[g, i]

        # Create a NLP solver
        prob = {"f": objective,
                # "x": vertcat(self.q_sym_global.reshape((self.nb_kinematic_dofs * self.nb_frames, 1)), self.p_sym),
                "x": vertcat(self.q_sym_global_vec, self.p_sym),
                # "x": a,
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
        # get the bounds of the model for all dofs
        bounds = [
            (mini, maxi) for mini, maxi in zip(get_range_q(self.biorbd_model)[0], get_range_q(self.biorbd_model)[1])
        ]

        # find kinematic dof with initial guess at zeros
        idx_zeros = np.where(np.sum(self.q_ik_initial_guess, axis=1) == 0)[0]
        kinematic_idx_zeros = [idx for idx in idx_zeros if idx in self.q_kinematic_index]

        # initialize q_ik with in the half-way between bounds
        bounds_kinematic_idx_zeros = [b for i, b in enumerate(bounds) if i in kinematic_idx_zeros]
        kinova_q0 = np.array([(b[0] + b[1]) / 2 for b in bounds_kinematic_idx_zeros])

        # initialized q trajectories for each frames for dofs without a priori knowledge of the q (kinova arm here)
        self.q_ik_initial_guess[kinematic_idx_zeros, :] = np.repeat(
            kinova_q0[:, np.newaxis], self.nb_frames_ik_step, axis=1
        )

        # initialized q qnd p for the whole algorithm.

        p_init_global = np.zeros( self.nb_parameters_dofs)
        q_init_global = self.q_ik_initial_guess[self.q_kinematic_index, :]

        print(" #######  Initialisation beginning  ########")


        # First IK step - INITIALIZATION

        q_all_frames = self.inverse_kinematics(
            q_init=q_init_global,
            p_init=p_init_global,
        )[0]

        print(" #######  Initialisation ending ########")

        #self.x_ik_initial_guess[self.q_kinematic_index, :] = q_all_frames
        p_init = p_init_global
        #q0 = self.q_ik_initial_guess[:, 0]

        #x_output = np.zeros((self.biorbd_model.nbQ(), self.markers.shape[2]))

        # #bounds = [
        #     (mini, maxi) for mini, maxi in zip(get_range_q(self.biorbd_model)[0], get_range_q(self.biorbd_model)[1])
        # ]

        #self.bounds_param = [[bounds[k][0] for k in self.q_parameter_index], [bounds[l][1] for l in self.q_parameter_index]]

        p_init = x_step_2[self.q_parameter_index, 0]

        iteration = 0
        epsilon_markers_n = 10  # arbitrary set
        epsilon_markers_n_minus_1 = 0
        delta_epsilon_markers = epsilon_markers_n - epsilon_markers_n_minus_1

        print("#####   Starting the while loop   #####")


        while fabs(delta_epsilon_markers) > threshold:
            q_first_ik_not_all_frames = q_step_2[:, self.list_frames_param_step]

            #print("threshold", threshold, "delta", abs(delta_epsilon_markers))
            #markers_xp_data_not_all_frames = self.markers[:, :, self.list_frames_param_step]

            print("threshold", threshold, "delta", abs(delta_epsilon_markers))

            epsilon_markers_n_minus_1 = epsilon_markers_n
            # step 1 - param opt

            param_opt = self.parameters_optimization(
                q_init_all=q_all_frames,
                    fun=self.objective_function_param,
                    args=(q_first_ik_not_all_frames, q0, markers_xp_data_not_all_frames,self.weights_param),
                    x0=p,
                    bounds=bounds[10:16],
                    method="trust-constr",
                    jac="3-point",
                    tol=1e-5,
                )

            elif self.param_solver == "ipopt":

                # # the value of the diff between xp and model markers for the table must reach 0
                # constraint = ()
                # for i in range(5):
                #     l=[]
                #     j1=[]
                #     j=np.zeros((len(self.list_frames_param_step),6))
                #     c=0
                #     for f, frame in enumerate(self.list_frames_param_step):
                #         x0 = self.q_ik_initial_guess[self.q_kinematic_index, 0] if f == 0 else q_output[
                #             self.q_kinematic_index, f - 1]
            #
            #
            #
            #     jac_scalar = lambda x : self.param_gradient(p,q_first_ik_not_all_frames,q0,markers_xp_data_not_all_frames)
            #
            #     frame_constraint=self.list_frames_param_step[0]
            #     #x0 = q_step_2[self.q_kinematic_index, 0] if frame_constraint == 0 else x_output[
            #         #self.q_kinematic_index, frame_constraint - 1]
            #     x0 = q_step_2[self.q_kinematic_index, 0] if frame_constraint == 0 else x_output[
            #         self.q_kinematic_index, frame_constraint - 1]
            #     q_init=q_step_2[:,0] if frame_constraint == 0 else x_output[:,frame_constraint - 1]
            #     constraint = self.build_constraint_parameters(f=frame_constraint,q_init=q_init,x=x0)
            #
            #     param_opt = minimize_ipopt(
            #         fun=self.objective_function_param,
            #         x0=p,
            #         args=(q_first_ik_not_all_frames, q0, markers_xp_data_not_all_frames,self.weights_ik),
            #         bounds=self.bounds_param,
            #         #jac=self.param_gradient,
            #         #constraints=constraint,
            #         tol=1e-4,
            #         options={'max_iter': 3000},
            #     )
            #
            # frame_constraint = self.list_frames_param_step[0]

                jac_scalar = lambda x : self.param_gradient(p,q_first_ik_not_all_frames,q0,markers_xp_data_not_all_frames)

            param_opt = self.step_1(
                x_step_2=x_step_2,
                p_init=p_init,
                x0 = q_step_2[self.q_kinematic_index, 0] if frame_constraint == 0 else q_output[
                    self.q_kinematic_index, frame_constraint - 1]
                q_init=q_step_2[:,0] if frame_constraint == 0 else q_output[:,frame_constraint - 1]
                constraint = self.build_constraint_parameters(f=frame_constraint,q_init=q_init,x=x0)

                param_opt = minimize_ipopt(
                    fun=self.objective_function_param,
                    x0=p,
                    args=(q_first_ik_not_all_frames, q0, markers_xp_data_not_all_frames,self.weights_ik),
                    bounds=self.bounds_param,
                    #jac=self.param_gradient,
                    #constraints=constraint,
                    tol=1e-4,
                    options={'max_iter': 3000},
                )
            #self.x_ik_initial_guess[self.q_parameter_index, :] = param_opt
            print(" param opt =", param_opt)

            print(" param opt =" , param_opt)

            #self.q_ik_initial_guess[self.q_parameter_index, :] = np.array([param_opt.x] * self.nb_frames_ik_step).T
            #self.q_ik_initial_guess[self.q_parameter_index, : ] = param_opt
            #self.x_output[self.q_parameter_index, :] = param_opt.x
            #x_output[self.q_parameter_index, :] = np.array([param_opt.x] * x_output.shape[1]).T

            # step 2 - ik step
            q_all_frames, epsilon_markers_n = self.inverse_kinematics(
                q_init=q_all_frames,
                p_init=param_opt,
            )

            #self.x_ik_initial_guess[self.q_kinematic_index, :] = q_all_frames
            jacobians_used.append(jacobian_x)
            # data_frame = self.solution()  # dict
            # for valeur in data_frame():
            # print(valeur)

            delta_epsilon_markers = epsilon_markers_n - epsilon_markers_n_minus_1
            print("delta_epsilon_markers:", delta_epsilon_markers)
            print("epsilon_markers_n:", epsilon_markers_n)
            print("epsilon_markers_n_minus_1:", epsilon_markers_n_minus_1)
            iteration += 1
            print("iteration:", iteration)
            #self.x_ik_initial_guess = self.x_output


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

    def penalty_table_markers(self, vect_pos_markers: np.ndarray, table_markers: np.ndarray):
        """
        The penalty function which put the pivot joint vertical

        Parameters
        ----------
        vect_pos_markers: np.ndarray
            The generalized coordinates from the model
        table_markers: np.ndarray
            The markers position from experimental data

        Return
        ------
        The value of the penalty function
        """
        table5_xyz = vect_pos_markers[
                     self.table_markers_idx[0] * 3: self.table_markers_idx[0] * 3 + 3
                     ][:]
        table_xp = table_markers[:, 0].tolist()
        table6_xy = vect_pos_markers[
                    self.table_markers_idx[1] * 3: self.table_markers_idx[1] * 3 + 3
                    ][:2]
        table_xp += table_markers[:2, 1].tolist()
        table = table5_xyz.tolist() + table6_xy.tolist()

        return table, table_xp

    def theta_pivot_penalty(self, q: np.ndarray):
        """
        Penalty function, prevent part 1 and 3 to cross

        Parameters
        ----------
        q: np.ndarray
            Generalized coordinates for all dof, unique for all frames

        Return
        ------
        The value of the penalty function
        """
        # todo : remove the hard coded, put index as an argument of this method
        theta_part1_3 = q[-2] + q[-1]
        theta_part1_3_lim = 7 * np.pi / 10

        if theta_part1_3 > theta_part1_3_lim:
            diff_model_pivot = [theta_part1_3]
            diff_xp_pivot = [theta_part1_3_lim]
        else:
            theta_cost = 0
            diff_model_pivot = [theta_cost]
            diff_xp_pivot = [theta_cost]

        return diff_model_pivot, diff_xp_pivot

    def penalty_open_loop_markers(self, model_markers_values: np.ndarray, open_loop_markers: np.ndarray):
        """
        The penalty function which minimize the difference between the open loop markers position from experimental data
        and from the model

        Parameters
        ----------
        model_markers_values: np.ndarray
            The markers location from the model [nb_markers x 3, 1]
        open_loop_markers: np.ndarray
            The open loop markers position form experimental data

        Return
        ------
        The value of the penalty function
        """
        list_model = []
        list_xp = []
        for j, name in enumerate(self.markers_model):
            if name != self.markers_model[self.table_markers_idx[0]] and name != self.markers_model[
                self.table_markers_idx[1]]:
                mark = model_markers_values[
                       self.markers_model.index(name) * 3: self.markers_model.index(name) * 3 + 3
                       ].tolist()
                open_loop = open_loop_markers[:, self.markers_model.index(name)].tolist()
                list_model += mark
                list_xp += open_loop

        return list_model, list_xp

    def penalty_rotation_matrix(self, x_with_p: np.ndarray):
        """
        The penalty function which force the model to stay horizontal

        Parameters
        ----------
        x_with_p: np.ndarray
            Generalized coordinates for all dof, unique for all frames

        Return
        ------
        The value of the penalty function
        """
        rotation_matrix = self.biorbd_model.globalJCS(x_with_p, self.biorbd_model.nbSegment() - 1).rot().to_array()

        rot_matrix_list_model = [
            rotation_matrix[2, 0],
            rotation_matrix[2, 1],
            rotation_matrix[0, 2],
            rotation_matrix[1, 2],
            (rotation_matrix[2, 2] - 1),
        ]
        rot_matrix_list_xp = [0] * len(rot_matrix_list_model)
        return rot_matrix_list_model, rot_matrix_list_xp

    def penalty_q_open_loop(self, x, q_init):
        """
        Minimize the q of open_loop

        Parameters
        ----------
        x: np.ndarray
            Generalized coordinates for all dof except those between ulna and piece 7, unique for all frames
        q_init: np.ndarray
            The initial values of generalized coordinates for the actual frame

        Return
        ------
        The value of the penalty function
        """
        #
        q_continuity_diff_model = []
        q_continuity_diff_xp = []
        for i, value in enumerate(x):
            q_continuity_diff_xp += [q_init[i]]
            q_continuity_diff_model += [value]

        return q_continuity_diff_model, q_continuity_diff_xp

    def objective_function_param(self, p0: np.ndarray, x: np.ndarray, x0: np.ndarray, markers_calibration: np.ndarray, weight:list):
        """
        Objective function,use in the Inverse Kinematic

        Parameters
        ----------
        p0: np.ndarray
            (6x1) Generalized coordinates between ulna and piece 7, unique for all frames
        x: np.ndarray
            Generalized coordinates for all frames all dof
        x0: np.ndarray
            Generalized coordinates for the first frame
        markers_calibration: np.ndarray
            (3 x n_markers x n_frames) marker values for calibration frames

        Return
        ------
        The value of the objective function
        """
        index_table_markers = [i for i, value in enumerate(self.markers_model) if "Table" in value]
        index_wu_markers = [i for i, value in enumerate(self.markers_model) if "Table" not in value]

        # be filled in the loop
        table5_xyz_all_frames = 0
        table6_xy_all_frames = 0
        mark_out_all_frames = 0
        rotation_matrix_all_frames = 0

        Q = np.zeros(x.shape[0])

        Q[self.q_parameter_index] = p0

        for f, frame in enumerate(self.list_frames_param_step):
            thorax_markers = markers_calibration[:, index_wu_markers[0]: index_wu_markers[-1] + 1, f]
            table_markers = markers_calibration[:, index_wu_markers[-1] + 1:, f]

            Q[self.q_kinematic_index] = x[self.q_kinematic_index, f]

            markers_model = self.biorbd_model.markers(Q)

            vect_pos_markers = np.zeros(3 * len(markers_model))

            for m, value in enumerate(markers_model):
                vect_pos_markers[m * 3: (m + 1) * 3] = value.to_array()

            table_model, table_xp = self.penalty_table_markers(vect_pos_markers, table_markers)

            table5_xyz = np.linalg.norm(np.array(table_model[:3]) - np.array(table_xp[:3])) ** 2
            table5_xyz_all_frames += table5_xyz

            table6_xy = np.linalg.norm(np.array(table_model[3:]) - np.array(table_xp[3:])) ** 2
            table6_xy_all_frames += table6_xy

            thorax_list_model, thorax_list_xp = self.penalty_open_loop_markers(vect_pos_markers, thorax_markers)

            mark_out = 0
            for j in range(len(thorax_markers[0, :])):
                mark = np.linalg.norm(np.array(thorax_list_model[j: j + 3]) - np.array(thorax_list_xp[j: j + 3])) ** 2
                mark_out += mark
            mark_out_all_frames += mark_out

            rot_matrix_list_model, rot_matrix_list_xp = self.penalty_rotation_matrix(Q)[0], \
                                                        self.penalty_rotation_matrix(Q)[1]

            rotation_matrix = 0
            for i in rot_matrix_list_model:
                rotation_matrix += i ** 2

            rotation_matrix_all_frames += rotation_matrix

            q_continuity_diff_model, q_continuity_diff_xp = self.penalty_q_open_loop(Q, x0)
            # Minimize the q of open loop
            q_continuity = np.sum((np.array(q_continuity_diff_model) - np.array(q_continuity_diff_xp)) ** 2)

            pivot_diff_model, pivot_diff_xp = self.theta_pivot_penalty(Q)
            pivot = (pivot_diff_model[0] - pivot_diff_xp[0]) ** 2

            x0 = Q

        return (
                    weight[0] * (table5_xyz_all_frames + table6_xy_all_frames)
                + weight[1] * mark_out_all_frames
                + weight[2] * pivot
                + weight[3] * q_continuity
                + weight[4] * rotation_matrix_all_frames

        )

    def objective_ik_list(
        self,
        x: np.ndarray,
        p: np.ndarray,
        table_markers: np.ndarray,
        thorax_markers: np.ndarray,
        q_init: np.ndarray,
    ):
        """
        This function

        Parameters
        ----------
        x: np.ndarray
            Generalized coordinates for all dof except those between ulna and piece 7, unique for all frames
        p: np.ndarray
            Generalized coordinates between ulna and piece 7
        table_markers: np.ndarray
            The markers position of the table from experimental data [3x2]
        thorax_markers: np.ndarray
            The others markers position from experimental data [3x14]
        q_init: np.ndarray
            The initial values of generalized coordinates for the actual frame

        Return
        ------
        The value of the objective function
        """
        if p is not None:
            new_x = np.zeros(self.biorbd_model.nbQ())  # we add p to x because the optimization is on p so we can't
            # give all x to minimize
            new_x[self.q_kinematic_index] = x
            new_x[self.q_parameter_index] = p
        else:
            new_x = x

        markers_model = self.biorbd_model.markers(new_x)

        vect_pos_markers = np.zeros(3 * len(markers_model))

        for m, value in enumerate(markers_model):
            vect_pos_markers[m * 3: (m + 1) * 3] = value.to_array()

        # Put the pivot joint vertical
        table_model, table_xp = self.penalty_table_markers(vect_pos_markers, table_markers)

        # Minimize difference between open loop markers from model and from experimental data
        thorax_list_model, thorax_list_xp = self.penalty_open_loop_markers(vect_pos_markers, thorax_markers)

        # Force the model horizontality
        rot_matrix_list_model, rot_matrix_list_xp = self.penalty_rotation_matrix(new_x)

        # Minimize the q of open loop
        q_continuity_diff_model, q_continuity_diff_xp = self.penalty_q_open_loop(x, q_init)

        # # Force part 1 and 3 to not cross
        pivot_diff_model, pivot_diff_xp = self.theta_pivot_penalty(new_x)

        # We add our vector to the main lists
        # diff_model = table_model + thorax_list_model + q_continuity_diff_model + pivot_diff_model + rot_matrix_list_model
        # diff_xp = table_xp + thorax_list_xp + q_continuity_diff_xp + pivot_diff_xp + rot_matrix_list_xp
        diff_model = table_model + thorax_list_model + q_continuity_diff_model + pivot_diff_model + rot_matrix_list_model
        diff_xp = table_xp + thorax_list_xp + q_continuity_diff_xp + pivot_diff_xp + rot_matrix_list_xp

        # We converted our list into array in order to be used by least_square
        diff_tab_model = np.array(diff_model)
        diff_tab_xp = np.array(diff_xp)

        # We created the difference vector
        diff = diff_tab_model - diff_tab_xp
        diff * self.weight_list_ik

        if self.ik_solver == "leastsquare":
            return diff * self.weight_list_param
        if self.ik_solver == "ipopt":
            return diff * self.weight_list_ik

    def objective_ik_scalar(self, x, p, table_markers, thorax_markers, q_init):
        objective_ik_list = self.objective_ik_list(x, p, table_markers, thorax_markers, q_init)
        return 0.5 * np.sum(np.array(objective_ik_list) ** 2, axis=0)

    def step_2_old(
        self,
        p: np.ndarray = None,
        bounds: np.ndarray = None,
        q_output: np.ndarray = None,
    ):

        """

        Determine the generalized coordinates with an IK

        Parameters
        ----------
        p :np.ndarray
            parameters values
        bounds : np.ndarray
            Lower and upper bounds on independent variables
         q_output : np.ndarray
            array of zeros


        Return
        ------
        espilon_markers :int
            sum of squared norm of difference
        q_output : np.ndarray
            generalized coordinates at the end of step 2

        """

        index_table_markers = [i for i, value in enumerate(self.markers_model) if "Table" in value]
        index_wu_markers = [i for i, value in enumerate(self.markers_model) if "Table" not in value]

        self.index_table_markers = index_table_markers
        self.index_wu_markers = index_wu_markers

        # build the bounds for step 2
        bounds_without_p_min = bounds[0][self.q_kinematic_index]
        bounds_without_p_max = bounds[1][self.q_kinematic_index]

        bounds_without_p = (bounds_without_p_min, bounds_without_p_max)
        bounds_without_p_list = [list(bounds_without_p_min), list(bounds_without_p_max)]

        gain = []
        for f in range(self.nb_frames_ik_step):

            x0 = self.q_ik_initial_guess[self.q_kinematic_index, 0] if f == 0 else q_output[
                self.q_kinematic_index, f - 1]


            start = time.time()

            if self.ik_solver == "leastsquare":

                if self.use_analytical_jacobians:
                    jac = lambda x, p, index_table_markers, index_wu_markers, x0: self.ik_jacobian(x, self.biorbd_model,
                                                                                                   self.weights_param)
                else:
                    jac = "3-point"

                IK_i = optimize.least_squares(
                    fun=self.objective_ik_list,
                    args=(
                        p,
                        self.markers[:, index_table_markers, f],
                        self.markers[:, index_wu_markers, f],
                        x0,
                    ),
                    x0=x0,  # x0 q without p
                    bounds=bounds_without_p,
                    method="trf",
                    jac=jac,
                    xtol=1e-5,
                )

                q_output[self.q_kinematic_index, f] = IK_i.x

                jacobian = IK_i.jac

            elif self.ik_solver == "ipopt":

                obj_fun = lambda x: self.objective_ik_scalar(x, p, self.markers[:, index_table_markers, f],
                                                             self.markers[:, index_wu_markers, f], x0)

                jac_scalar = lambda x: self.ik_gradient(x, p,self.markers[:, index_table_markers, f],self.markers[:, index_wu_markers, f], x0, self.biorbd_model, self.weights_ik)
                constraint = self.build_constraint(f, x0, p)
                ipopt_i = minimize_ipopt(
                    fun=obj_fun,
                    x0=x0,
                    #jac=jac_scalar,
                    constraints=constraint,
                    bounds=bounds_without_p_list,
                    tol=1e-4,
                    options={'max_iter': 5000, "print_level": 4},
                )


                q_output[self.q_kinematic_index, f] = ipopt_i.x
                jacobian = jac_scalar(ipopt_i.x)

                # print(ipopt_i)

                if ipopt_i["success"] == False:
                    raise RuntimeError("This optimization failed")

            else:
                raise ValueError("This solver is not implemented, please use 'ipopt' or 'leastsquare'.")

            # todo: it seems be all the markers
            markers_model = self.biorbd_model.markers(q_output[:, f])
            markers_to_compare = self.markers[:, :, f]
            espilon_markers = 0

            # sum of squared norm of difference
            for j in range(index_table_markers[0]):
                mark = np.linalg.norm(markers_model[j].to_array()[:] - markers_to_compare[:, j]) ** 2
                espilon_markers += mark

        end = time.time()
        gain.append(["time spend for the IK =", end - start, "use_analytical_jacobian=", self.use_analytical_jacobians])
        print("step 2 done")
        print(gain)

        return q_output, espilon_markers, gain[0][1], jacobian


    def build_constraint(self, f, q_init, p):
        constraint = ()
        index_table_markers = self.table_markers_idx
        index_wu_markers = self.model_markers_idx

        # 1
        constraint_fun = lambda x: self.objective_ik_list(x, p, self.markers[:, index_table_markers, f],
                                                          self.markers[:, index_wu_markers, f], q_init)[0]


        jac_table = lambda x: jacobians.marker_jacobian_table(x, self.biorbd_model, self.table_markers_idx,
                                                              self.q_parameter_index)[0, :] * self.weights_ik[0]

        constraint += ({"fun": constraint_fun, "jac": jac_table, "type": "eq"},)

        # 2
        constraint_fun = lambda x: self.objective_ik_list(x, p, self.markers[:, index_table_markers, f],
                                                          self.markers[:, index_wu_markers, f], q_init)[1]

        jac_table = lambda x: jacobians.marker_jacobian_table(x, self.biorbd_model, self.table_markers_idx,
                                                              self.q_parameter_index)[1, :] * self.weights_ik[0]

        constraint += ({"fun": constraint_fun, "jac": jac_table, "type": "eq"},)

        # 3
        constraint_fun = lambda x: self.objective_ik_list(x, p, self.markers[:, index_table_markers, f],
                                                          self.markers[:, index_wu_markers, f], q_init)[2]

        jac_table = lambda x: jacobians.marker_jacobian_table(x, self.biorbd_model, self.table_markers_idx,
                                                              self.q_parameter_index)[2, :] * self.weights_ik[0]

        constraint += ({"fun": constraint_fun, "jac": jac_table, "type": "eq"},)

        # 4
        constraint_fun = lambda x: self.objective_ik_list(x, p, self.markers[:, index_table_markers, f],
                                                          self.markers[:, index_wu_markers, f], q_init)[3]

        jac_table = lambda x: jacobians.marker_jacobian_table(x, self.biorbd_model, self.table_markers_idx,
                                                              self.q_parameter_index)[3, :] * self.weights_ik[0]

        constraint += ({"fun": constraint_fun, "jac": jac_table, "type": "eq"},)

        # 5
        constraint_fun = lambda x: self.objective_ik_list(x, p, self.markers[:, index_table_markers, f],
                                                          self.markers[:, index_wu_markers, f], q_init)[4]

        jac_table = lambda x: jacobians.marker_jacobian_table(x, self.biorbd_model, self.table_markers_idx,
                                                              self.q_parameter_index)[4, :] * self.weights_ik[0]

        constraint += ({"fun": constraint_fun, "jac": jac_table, "type": "eq"},)

        return constraint

    def build_constraint_parameters(self, f, q_init, x):
        constraint = ()
        index_table_markers = self.table_markers_idx
        index_wu_markers = self.model_markers_idx

        # 1
        constraint_fun = lambda p: self.objective_ik_list(x, p, self.markers[:, index_table_markers, f],
                                                          self.markers[:, index_wu_markers, f], q_init)[0]

        jac_table = lambda p: jacobians.marker_jacobian_table(x, self.biorbd_model, self.table_markers_idx,
                                                              self.q_parameter_index)[0, :] * self.weights_ik[0]

        constraint += ({"fun": constraint_fun, "jac": jac_table, "type": "eq"},)

        # 2
        constraint_fun = lambda p: self.objective_ik_list(x, p, self.markers[:, index_table_markers, f],
                                                          self.markers[:, index_wu_markers, f], q_init)[1]

        jac_table = lambda p: jacobians.marker_jacobian_table(x, self.biorbd_model, self.table_markers_idx,
                                                              self.q_parameter_index)[1, :] * self.weights_ik[0]

        constraint += ({"fun": constraint_fun, "jac": jac_table, "type": "eq"},)

        # 3
        constraint_fun = lambda p: self.objective_ik_list(x, p, self.markers[:, index_table_markers, f],
                                                          self.markers[:, index_wu_markers, f], q_init)[2]

        jac_table = lambda p: jacobians.marker_jacobian_table(x, self.biorbd_model, self.table_markers_idx,
                                                              self.q_parameter_index)[2, :] * self.weights_ik[0]

        constraint += ({"fun": constraint_fun, "jac": jac_table, "type": "eq"},)

        # 4
        constraint_fun = lambda p: self.objective_ik_list(x, p, self.markers[:, index_table_markers, f],
                                                          self.markers[:, index_wu_markers, f], q_init)[3]

        jac_table = lambda p: jacobians.marker_jacobian_table(x, self.biorbd_model, self.table_markers_idx,
                                                              self.q_parameter_index)[3, :] * self.weights_ik[0]

        constraint += ({"fun": constraint_fun, "jac": jac_table, "type": "eq"},)

        # 5
        constraint_fun = lambda p: self.objective_ik_list(x, p, self.markers[:, index_table_markers, f],
                                                          self.markers[:, index_wu_markers, f], q_init)[4]

        jac_table = lambda p: jacobians.marker_jacobian_table(x, self.biorbd_model, self.table_markers_idx,
                                                              self.q_parameter_index)[4, :] * self.weights_ik[0]

        constraint += ({"fun": constraint_fun, "jac": jac_table, "type": "eq"},)

        return constraint

    def ik_jacobian(self, x, biorbd_model, weights):
        """
        This function return the entire Jacobian of the system for the inverse kinematics step

             Parameters
             ----------
             x: np.ndarray
                 Generalized coordinates WITHOUT parameters values
             biorbd_model: biorbd.Models
                 the model used
            weights : list[int]
                list of the weight associated for each Jacobians

             Return
             ------
             the Jacobian of the entire system
             """

        table = jacobians.marker_jacobian_table(x, biorbd_model, self.table_markers_idx, self.q_parameter_index)

        # Minimize difference between thorax markers from model and from experimental data
        model = jacobians.marker_jacobian_model(x, biorbd_model, self.model_markers_idx, self.q_parameter_index)

        # Force z-axis of final segment to be vertical
        # rot_matrix_list_model  = kcc.penalty_rotation_matrix( x_with_p )
        # rot_matrix_list_xp = kcc.penalty_rotation_matrix(x_with_p)

        rotation = jacobians.rotation_matrix_jacobian(x, biorbd_model, self.segment_id_with_vertical_z,
                                                      self.q_parameter_index)

        # Minimize the q of thorax
        continuity = jacobians.jacobian_q_continuity(x, self.q_parameter_index)

        # Force part 1 and 3 to not cross
        pivot = jacobians.marker_jacobian_theta(x, self.q_parameter_index)

        # concatenate all Jacobians
        # size [16  x 69 ]
        jacobian = np.concatenate(
            (table * weights[0],
             model * weights[1],
             continuity * weights[3],
             pivot * weights[2],
             rotation * weights[4],
             ),
            axis=0
        )

        return jacobian

    def ik_parameters_jacobian(self, p, biorbd_model, weights):

        table = jacobians.jacobian_table_parameters(p, biorbd_model, self.table_markers_idx, self.q_kinematic_index)
        model = jacobians.markers_jacobian_model_parameters(p, biorbd_model, self.model_markers_idx,
                                                            self.q_kinematic_index)
        rotation = jacobians.rotation_matrix_parameter_jacobian(p, biorbd_model, self.segment_id_with_vertical_z,
                                                                self.q_kinematic_index)
        continuity = jacobians.jacobian_q_continuity_parameters()
        pivot = jacobians.marker_jacobian_theta_parameters()

        # concatenate all Jacobians
        # size [6  x 69 ]
        jacobian = np.concatenate(
            (table * weights[0],
             model * weights[1],
             continuity * weights[3],
             pivot * weights[2],
             rotation * weights[4],
             ),
            axis=0
        )

        return jacobian

    # def param_gradient(self, p,p0,x,x0,markers_calibration q_first_ik_not_all_frames, q0, markers_xp_data_not_all_frames,weight):
    #     jac=self.ik_parameters_jacobian(
    #         p, self.biorbd_model, self.weights_ipopt).repeat(self.nb_frames_param_step, axis=0)
    #     for k in range(np.shape(jac)[1]):
    #         for i in range(np.shape(jac)[0]):
    #             jac[i][k]*self.objective_function_param( p0=, x=, x0=, markers_calibration=, weight=)
    #     return self.ik_parameters_jacobian(
    #         p, self.biorbd_model, self.weights_ipopt).repeat(self.nb_frames_param_step, axis=0).sum(axis=0).tolist()

    def ik_gradient(self, x,p, table_markers,thorax_markers,q_init, biorbd_model, weights):
        jac=self.ik_jacobian(x,biorbd_model,weights)
        shape=np.shape(jac)
        for k in range(shape[1]):
            for i in range(shape[0]):
                jac[i][k]*self.objective_ik_list(x,p,table_markers,thorax_markers,q_init)[i]
        return jac.sum(axis=0).tolist()
        #return self.ik_jacobian(x, biorbd_model, weights).sum(axis=0).tolist()

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
            #mi = self.markers[:, :, f]
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

            # artificially add 0 for table6 z-axis used to reshape
            #table_mod.append(0)
            #table_xp.append(0)

            table_mod = np.asarray(table_mod)
            table_xp = np.asarray(table_xp)

            # residuals = np.zeros(((len(markers_model) - 2),1))  remove the 2 Table's markers
            residuals_model = np.zeros(((len(markers_model) - 2), 1))
            residuals_table = np.zeros((self.nb_markers_table, 1))

            # determinate the residual² btwm the coordinates of each marker
            # x_y_z
            # array_residual = (marker_mod - marker_xp)
            # residuals_xyz[:, :, f] = array_residual.reshape(3, self.nb_markers_model, order='F')

            array_residual_model = (marker_mod - marker_xp)
            array_residual_table = (table_mod - table_xp)
            residuals_xyz_model[:, :, f] = array_residual_model.reshape(3, self.nb_markers_model, order='F')
            residuals_xyz_table[:, :, f] = array_residual_table.reshape(3, self.nb_markers_table, order='F')

        # residuals_norm = np.linalg.norm(residuals_xyz, axis=0)
        # rmse_tot = np.sqrt(np.square(residuals_norm).mean(axis=0))
        # rmse_x = np.sqrt(np.square(residuals_xyz[0, :, :]).mean(axis=0))
        # rmse_y = np.sqrt(np.square(residuals_xyz[1, :, :]).mean(axis=0))
        # rmse_z = np.sqrt(np.square(residuals_xyz[2, :, :]).mean(axis=0))

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
            #gain_time=self.gain

        )

        return self.output

    # def graphs(self):
    #
    #     self.plot_graph_rmse()
    #     self.plot_graph_rmse_table()
    #     self.plot_rotation_matrix_penalty()
    #     self.plot_pivot_penalty()
    #
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

            # rotation_value.append(rot_matrix_list_model)
            rotation_matrix = self.biorbd_model_eigen.globalJCS(self.x_all_frames[:, i], self.biorbd_model.nbSegment() - 1).rot().to_array()
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

    def pivot(self):
        q_out = self.q
        #pivot_value = []
        # for i in range(self.nb_frames):
        #     pivot_diff_model = self.x_all_frames[-2:]
        #     pivot_value.append(pivot_diff_model)

        pivot_value_list = []
        for f in range(self.nb_frames):
            if self.x_all_frames[-2, f] + self.x_all_frames[-1, f] > 7 * np.pi /10 :
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
        ax.set_title("pivot value")
        plt.show()

        print("index where pivot value is not 0 =", index_not_zero)

    def plot_param_value(self):
        bound_param = self.bounds_p_list
        param_value = self.parameters
        for i in range(self.nb_parameters_dofs):
            if param_value[i] == bound_param[0][i] or param_value[i] == bound_param[1][0]:
                print("parameters number %r reach a bound value " %i )
        plt.figure("param value")
        plt.plot([k for k in range(self.nb_parameters_dofs)],[bound_param[0][u] for u in range(self.nb_parameters_dofs)],label="lower bound")
        plt.plot([k for k in range(self.nb_parameters_dofs)],[bound_param[1][u] for u in range(self.nb_parameters_dofs)],label="upper bound")
        plt.plot([k for k in range(self.nb_parameters_dofs)],param_value,label="parameters values")
        plt.xlabel(" number of parameter")
        plt.ylabel("value of parameter")
        plt.legend()
        plt.show()
        print("parameters values = ", param_value)


