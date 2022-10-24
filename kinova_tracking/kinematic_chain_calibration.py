from typing import Union
from enum import Enum
import random
import time

import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np

from cyipopt import minimize_ipopt
import biorbd

from scipy import optimize
from utils import get_range_q
import random

import jacobians
import time

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
        Name of each markers
    markers : np.ndarray
        matrix of zeros [3 x Nb markers , x nb frame]
    closed_loop_markers : list[str]
        Name of markers associated to the table
    tracked_markers : list[str]
        Name of associated to the model
    parameter_dofs : list[str]
        name dof for which parameters are constant on each frame
    kinematic_dofs : list
        name dof which parameters aren't constant on each frame
    weights :np.ndarray
        weight associated with cost functions
    q_ik_initial_guess : array
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
        weights: Union[list[float], np.ndarray],
        q_ik_initial_guess: np.ndarray,
        objectives_functions: ObjectivesFunctions = None,  # [n_dof x n_frames]
        nb_frames_ik_step: int = None,
        nb_frames_param_step: int = None,
        randomize_param_step_frames: bool = True,
        use_analytical_jacobians : bool = True,
        segment_id_with_vertical_z: int = None,
            param_solver: str = "leastsquare",
            ik_solver: str = "leastsquare",
    ):

        self.nb_markers = None
        self.biorbd_model = biorbd_model
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
        self.table_markers_idx = [self.markers_model.index(i) for i in self.markers_model if  "Table"  in i]
        self.model_markers_idx = [self.tracked_markers.index(i) for i in self.tracked_markers]

        # nb markers
        self.nb_markers_table = self.table_markers_idx.__len__()
        self.nb_markers_model = self.model_markers_idx.__len__()

        # find the indexes of parameters and kinematic dofs in the model
        self.q_parameter_index = [self.model_dofs.index(dof) for dof in self.parameter_dofs]
        self.q_kinematic_index = [self.model_dofs.index(dof) for dof in self.kinematic_dofs]

        self.nb_parameters_dofs = len(parameter_dofs)
        self.nb_kinematic_dofs = len(kinematic_dofs)

        # self.objectives_function
        self.param_solver=param_solver
        self.ik_solver = ik_solver

        # check if q_ik_initial_guess has the right size
        self.q_ik_initial_guess = q_ik_initial_guess
        self.nb_frames_ik_step = nb_frames_ik_step
        self.nb_frames_param_step = nb_frames_param_step
        self.randomize_param_step_frames = randomize_param_step_frames
        self.use_analytical_jacobians = use_analytical_jacobians

        self.list_frames_param_step = self.frame_selector(self.nb_frames_param_step, self.nb_frames_ik_step)

        # number of weights has to be checked
        # raise Error if not the right number
        self.weights = weights

        weight_closed_loop = [self.weights[0]] * (len(self.closed_loop_markers) * 3 - 1)
        # nb marker table * 3 dim - 1 because we don't use value on z for Table:Table6

        weight_open_loop = [self.weights[1]] * (
            len([i for i in self.tracked_markers if i not in self.closed_loop_markers]) * 3
        )
        # This is for all markers except those for table

        weight_rot_matrix = [self.weights[4]] * 5  #len(rot_matrix_list_xp)
        weight_theta_13 = [self.weights[2]]
        weight_continuity = [self.weights[3]] * (self.q_ik_initial_guess.shape[0] - len(self.parameter_dofs))
        # We need the nb of dofs but without parameters

        #self.weight_list = weight_closed_loop + weight_open_loop + weight_continuity + weight_theta_13 + weight_rot_matrix
        self.weight_list = weight_closed_loop + weight_open_loop + weight_continuity + weight_theta_13 + weight_rot_matrix


        self.list_sol = []
        self.q = np.zeros((self.biorbd_model.nbQ(), self.nb_frames_ik_step))
        self.parameters = np.zeros(self.nb_parameters_dofs)
        self.segment_id_with_vertical_z = segment_id_with_vertical_z
        self.output = dict()

    # if nb_frames_ik_step> markers.shape[2]:
    # raise error
    # self.nb_frame_ik_step = markers.shape[2] if nb_frame_ik_step is None else nb_frames_ik_step

    def solve(
        self,
        threshold: int = 5e-5,
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

        # prepare the size of the output of q
        q_output = np.zeros((self.biorbd_model.nbQ(), self.nb_frames_ik_step))

        # get the bounds of the model for all dofs
        bounds = [
            (mini, maxi) for mini, maxi in zip(get_range_q(self.biorbd_model)[0], get_range_q(self.biorbd_model)[1])
        ]

        # find kinematic dof with initial guess at zeros
        idx_zeros = np.where(np.sum(self.q_ik_initial_guess, axis=1) == 0)[0]
        kinematic_idx_zeros = [idx for idx in idx_zeros if idx in self.q_kinematic_index]

        # inititialize q_ik with in the half-way between bounds
        bounds_kinematic_idx_zeros = [b for i, b in enumerate(bounds) if i in kinematic_idx_zeros]
        kinova_q0 = np.array([(b[0] + b[1]) / 2 for b in bounds_kinematic_idx_zeros])

        # initialized q trajectories for each frames for dofs without a priori knowledge of the q (kinova arm here)
        self.q_ik_initial_guess[kinematic_idx_zeros, :] = np.repeat(
            kinova_q0[:, np.newaxis], self.nb_frames_ik_step, axis=1
        )

        # initialized parameters values
        p = np.zeros(self.nb_parameters_dofs)

        print("Initialisation")
        jacobians_used=[]
        gain_list=[]
        # First IK step - INITIALIZATION
        q_step_2, epsilon, gain, jacobian_ini  = self.step_2(
            p=p,
            bounds=get_range_q(self.biorbd_model),
            q_output=q_output,
        )

        gain_list.append(gain)
        jacobians_used.append(jacobian_ini)
        q0 = self.q_ik_initial_guess[:, 0]

        q_output = np.zeros((self.biorbd_model.nbQ(), self.markers.shape[2]))

        bounds = [
            (mini, maxi) for mini, maxi in zip(get_range_q(self.biorbd_model)[0], get_range_q(self.biorbd_model)[1])
        ]

        p = q_step_2[self.q_parameter_index, 0]

        iteration = 0
        epsilon_markers_n = 10
        epsilon_markers_n_minus_1 = 0
        delta_epsilon_markers = epsilon_markers_n - epsilon_markers_n_minus_1

        while abs(delta_epsilon_markers) > threshold:
            q_first_ik_not_all_frames = q_step_2[:, self.list_frames_param_step]

            markers_xp_data_not_all_frames = self.markers[:, :, self.list_frames_param_step]

            print("threshold", threshold, "delta", abs(delta_epsilon_markers))

            epsilon_markers_n_minus_1 = epsilon_markers_n
            # step 1 - param opt
            param_opt = optimize.minimize(
                fun=self.objective_function_param,
                args=(q_first_ik_not_all_frames, q0, markers_xp_data_not_all_frames),
                x0=p,
                bounds=bounds[10:16],
                method="trust-constr",
                jac="3-point",
                tol=1e-5,
            )
            print(param_opt.x)

            self.q_ik_initial_guess[self.q_parameter_index, :] = np.array([param_opt.x] * self.nb_frames_ik_step).T
            p = param_opt.x
            q_output[self.q_parameter_index, :] = np.array([param_opt.x] * q_output.shape[1]).T

            # step 2 - ik step
            q_out, epsilon_markers_n, gain2, jacobian_x = self.step_2(
                p= np.zeros((1,6)),
                bounds=get_range_q(self.biorbd_model),
                q_output=q_output,
            )

            gain_list.append(gain2)
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

            self.q_ik_initial_guess = q_output

        self.gain = gain_list
        self.parameters = p
        self.q = q_out



        return q_out, p, jacobians_used

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
            self.table_markers_idx[0] * 3 : self.table_markers_idx[0] * 3 + 3
        ][:]
        table_xp = table_markers[:, 0].tolist()
        table6_xy = vect_pos_markers[
            self.table_markers_idx[1] * 3 : self.table_markers_idx[1] * 3 + 3
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
            if name != self.markers_model[self.table_markers_idx[0]] and name != self.markers_model[self.table_markers_idx[1]]:
                #ie name is different from "Table:Table5" and "Table:Table6"
                mark = model_markers_values[
                    self.markers_model.index(name) * 3 : self.markers_model.index(name) * 3 + 3
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
            The initial values of generalized coordinates fo the actual frame

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

    def objective_function_param(self, p0: np.ndarray, x: np.ndarray, x0: np.ndarray, markers_calibration: np.ndarray):
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
            thorax_markers = markers_calibration[:, index_wu_markers[0] : index_wu_markers[-1] + 1, f]
            table_markers = markers_calibration[:, index_wu_markers[-1] + 1 :, f]

            Q[self.q_kinematic_index] = x[self.q_kinematic_index, f]

            markers_model = self.biorbd_model.markers(Q)

            vect_pos_markers = np.zeros(3 * len(markers_model))

            for m, value in enumerate(markers_model):
                vect_pos_markers[m * 3 : (m + 1) * 3] = value.to_array()

            table_model, table_xp = self.penalty_table_markers(vect_pos_markers, table_markers)

            table5_xyz = np.linalg.norm(np.array(table_model[:3]) - np.array(table_xp[:3])) ** 2
            table5_xyz_all_frames += table5_xyz

            table6_xy = np.linalg.norm(np.array(table_model[3:]) - np.array(table_xp[3:])) ** 2
            table6_xy_all_frames += table6_xy

            thorax_list_model, thorax_list_xp = self.penalty_open_loop_markers(vect_pos_markers, thorax_markers)

            mark_out = 0
            for j in range(len(thorax_markers[0, :])):
                mark = np.linalg.norm(np.array(thorax_list_model[j : j + 3]) - np.array(thorax_list_xp[j : j + 3])) ** 2
                mark_out += mark
            mark_out_all_frames += mark_out

            rot_matrix_list_model, rot_matrix_list_xp = self.penalty_rotation_matrix(Q)[0],self.penalty_rotation_matrix(Q)[1]

            rotation_matrix = 0
            for i in rot_matrix_list_model:
                rotation_matrix += i**2

            rotation_matrix_all_frames += rotation_matrix

            q_continuity_diff_model, q_continuity_diff_xp = self.penalty_q_open_loop(Q, x0)
            # Minimize the q of open loop
            q_continuity = np.sum((np.array(q_continuity_diff_model) - np.array(q_continuity_diff_xp)) ** 2)

            pivot_diff_model, pivot_diff_xp = self.theta_pivot_penalty(Q)
            pivot = (pivot_diff_model[0] - pivot_diff_xp[0]) ** 2

            x0 = Q


        return (
            self.weights[0] * (table5_xyz_all_frames + table6_xy_all_frames)
            + self.weights[1] * mark_out_all_frames
            + self.weights[2] * pivot
            + self.weights[3] * q_continuity
            + self.weights[4] * rotation_matrix_all_frames

        )

    def ik_step(
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
            The initial values of generalized coordinates fo the actual frame

        Return
        ------
        The value of the objective function
        """
        if p is not None:
            new_x = np.zeros(self.biorbd_model.nbQ())  # we add p to x because the optimization is on p so we can't
            # give all x to mininimize
            new_x[self.q_kinematic_index] = x
            new_x[self.q_parameter_index] = p
        else:
            new_x = x

        markers_model = self.biorbd_model.markers(new_x)

        vect_pos_markers = np.zeros(3 * len(markers_model))

        for m, value in enumerate(markers_model):
            vect_pos_markers[m * 3 : (m + 1) * 3] = value.to_array()

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
        #diff_model = table_model + thorax_list_model + q_continuity_diff_model + pivot_diff_model + rot_matrix_list_model
        #diff_xp = table_xp + thorax_list_xp + q_continuity_diff_xp + pivot_diff_xp + rot_matrix_list_xp
        diff_model = table_model + thorax_list_model + q_continuity_diff_model + pivot_diff_model + rot_matrix_list_model
        diff_xp = table_xp + thorax_list_xp + q_continuity_diff_xp + pivot_diff_xp + rot_matrix_list_xp


        # We converted our list into array in order to be used by least_square
        diff_tab_model = np.array(diff_model)
        diff_tab_xp = np.array(diff_xp)

        # We created the difference vector
        diff = diff_tab_model - diff_tab_xp

        return diff * self.weight_list


    def step_2(
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

        # build the bounds for step 2
        bounds_without_p_min = bounds[0][self.q_kinematic_index]
        bounds_without_p_max = bounds[1][self.q_kinematic_index]

        bounds_without_p = (bounds_without_p_min, bounds_without_p_max)

        gain = []
        for f in range(self.nb_frames_ik_step):

            if self.use_analytical_jacobians:
                jac = lambda x, p, index_table_markers, index_wu_markers, x0 : self.calibration_jacobian(x, self.biorbd_model, self.weights)

            else:
                jac = "3-point"

            x0 = self.q_ik_initial_guess[self.q_kinematic_index, 0] if f == 0 else q_output[self.q_kinematic_index, f - 1]

            start = time.time()


            IK_i = optimize.least_squares(
                fun=self.ik_step,
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

            jacobian=IK_i.jac

            markers_model = self.biorbd_model.markers(q_output[:, f])
            markers_to_compare = self.markers[:, :, f]
            espilon_markers = 0

            # sum of squared norm of difference
            for j in range(index_table_markers[0]):
                mark = np.linalg.norm(markers_model[j].to_array()[:] - markers_to_compare[:, j]) ** 2
                espilon_markers += mark

        end = time.time()
        gain.append(["time spend for the IK =",end - start, "use_analytical_jacobian=", self.use_analytical_jacobians])
        print("step 2 done")
        print(gain)

        return q_output, espilon_markers, gain[0][1],jacobian

    def calibration_jacobian(self,x, biorbd_model, weights):

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
        model = jacobians.marker_jacobian_model(x, biorbd_model,self.model_markers_idx, self.q_parameter_index )

        # Force z-axis of final segment to be vertical
        # rot_matrix_list_model  = kcc.penalty_rotation_matrix( x_with_p )
        # rot_matrix_list_xp = kcc.penalty_rotation_matrix(x_with_p)

        rotation = jacobians.rotation_matrix_jacobian(x, biorbd_model, self.segment_id_with_vertical_z, self.q_parameter_index)

        # Minimize the q of thorax
        continuity = jacobians.jacobian_q_continuity(x, self.q_parameter_index)

        # Force part 1 and 3 to not cross
        pivot = jacobians.marker_jacobian_theta(x,self.q_parameter_index)

        # concatenate all Jacobians
        # size [16  x 69 ]
        # #jacobian = np.concatenate(
        #     (table * weights[0],
        #      model * weights[1],
        #      continuity * weights[3],
        #      pivot * weights[2],
        #      rotation * weights[4],
        #      ),
        #     axis=0
        # )
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

    def solution(self):

        """
             This function returns a dictionnary which contains the global RMS and the RMS for each axes

             Parameters
             ----------

             Return
             ------
             the dictionnary with RMS
             """

        residuals_xyz = np.zeros((3, self.nb_markers_model, self.nb_frames))

        # for each frame
        for f in range(self.nb_frames):
            qi = self.q[:, f]
            mi = self.markers[:, :, f]
            markers_model = self.biorbd_model.markers(qi)
            vect_pos_markers = np.zeros(3 * len(markers_model))
            for m, value in enumerate(markers_model):
                vect_pos_markers[m * 3 : (m + 1) * 3] = value.to_array()

            # get coordinates for model and xp markers
            marker_mod, marker_xp = self.penalty_open_loop_markers(vect_pos_markers, mi)
            marker_mod = np.asarray(marker_mod)
            marker_xp = np.asarray(marker_xp)
            # self.penalty_table_markers()


            residuals = np.zeros(((len(markers_model) - 2),1))

            # determinate the residual² btwm the coordinates of each marker
            # x_y_z
            array_residual = (marker_mod - marker_xp)
            residuals_xyz[:, :, f] = array_residual.reshape(3, self.nb_markers_model, order='F')

        residuals_norm = np.linalg.norm(residuals_xyz, axis=0)
        rmse_tot = np.sqrt(np.square(residuals_norm).mean(axis=0))
        rmse_x = np.sqrt(np.square(residuals_xyz[0, :, :]).mean(axis=0))
        rmse_y = np.sqrt(np.square(residuals_xyz[1, :, :]).mean(axis=0))
        rmse_z = np.sqrt(np.square(residuals_xyz[2, :, :]).mean(axis=0))

        self.output = dict(
            rmse_x=rmse_x,
            rmse_y=rmse_y,
            rmse_z=rmse_z,
            rmse_tot=rmse_tot,
            max_marker=[self.markers_model[i] for i in np.argmax(residuals, axis=0)],
            gain_time=self.gain
            # message=[sol.message for sol in self.list_sol],
            # status=[sol.status for sol in self.list_sol],
            # success=[sol.success for sol in self.list_sol],
        )

        return self.output

