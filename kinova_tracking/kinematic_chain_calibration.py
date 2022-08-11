from typing import Union
from enum import Enum

import numpy as np

import biorbd

from scipy import optimize
from utils import get_range_q
import random


class ObjectivesFunctions(Enum):
    ALL_OBJECTIVES = "all objectives"
    ALL_OBJECTIVES_WITHOUT_FINAL_ROTMAT = "all objectives without final rotmat"


class KinematicChainCalibration:
    """

    Attributes
    ---------
    biord_model : biorbd.Model
        The biorbd Model

    Examples
    ---------
    kcc = KinematicChainCalibration()
    kcc.solve()
    kkc.results()
    """

    def __init__(self,
                 biorbd_model: biorbd.Model,
                 markers_model: list[str],
                 markers: np.array,  # [3 x nb_markers, x nb_frames]
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
                 use_analytical_jacobians: bool = False,
                 ):
        self.biorbd_model = biorbd_model

        # check if markers_model are in model
        # otherwise raise
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
            raise ValueError(f"markers and markers model must have same shape, markers shape is {markers.shape()},"
                             f" and markers_model shape is {markers_model.shape()}")
        self.closed_loop_markers = closed_loop_markers
        self.tracked_markers = tracked_markers
        self.parameter_dofs = parameter_dofs
        self.kinematic_dofs = kinematic_dofs

        # todo :
        self.parameter_idx = [kinematic_dofs.index(i) for i in self.parameter_dofs]
        # self.kinematic_idx =

        # self.objectives_function

        # number of wieghts has to be checked
        # raise Error if not the right number
        self.weights = weights

        # check if q_ik_initial_guess has the right size
        self.q_ik_initial_guess = q_ik_initial_guess
        self.nb_frames_ik_step = nb_frames_ik_step
        self.nb_frames_param_step = nb_frames_param_step
        self.randomize_param_step_frames = randomize_param_step_frames
        self.use_analytical_jacobians = use_analytical_jacobians

        self.nb_parameters_dofs = self.parameter_idx[-1]

        self.list_frames_param_step = self.frame_selector(self.nb_frames_param_step, self.nb_frames_ik_step)

    # if nb_frames_ik_step> markers.shape[2]:
    # raise error
    # self.nb_frame_ik_step = markers.shape[2] if nb_frame_ik_step is None else nb_frames_ik_step
    #

    # def solve(self, tolerance, use_analytical_jacobians:bool=True, objectives_functions: ObjectivesFunctions)

    def solve(
            self,
    ):
        """
        Parameters
        ----------

        Return
        ------
            The optimized Generalized coordinates
        """

        # prepare the size of the output of q
        q_output = np.zeros((self.biorbd_model.nbQ(), self.nb_frames_ik_step))

        # get the bounds of the model for all dofs
        bounds = [
            (mini, maxi) for mini, maxi in zip(get_range_q(self.biorbd_model)[0], get_range_q(self.biorbd_model)[1])
        ]
        kinova_q0 = np.array([(i[0] + i[1]) / 2 for i in bounds[self.parameter_idx[-1] + 1:]])
        # initialized q trajectories for each frames for dofs without a priori knowledge of the q (kinova arm here)
        for j in range((self.q_ik_initial_guess[self.parameter_idx[-1] + 1:, :].shape[1])):
            self.q_ik_initial_guess[self.parameter_idx[-1] + 1:, j] = kinova_q0

        # initialized parameters values
        p = np.zeros(len(self.parameter_idx))

        # First IK step - INITIALIZATION
        q_step_2, epsilon = self.step_2(
            p=p,
            bounds=get_range_q(self.biorbd_model),
            nb_frames=self.nb_frames_ik_step,
            q_output=q_output,

        )

        q0 = self.q_ik_initial_guess[:, 0]

        q_output = np.zeros((self.biorbd_model.nbQ(), self.markers.shape[2]))

        bounds = [
            (mini, maxi) for mini, maxi in
            zip(get_range_q(self.biorbd_model)[0], get_range_q(self.biorbd_model)[1])
        ]

        p = q_step_2[self.parameter_idx[0]: self.parameter_idx[-1] + 1, 0]

        iteration = 0
        epsilon_markers_n = 10
        epsilon_markers_n_minus_1 = 0
        delta_epsilon_markers = epsilon_markers_n - epsilon_markers_n_minus_1

        # todo: threshold as argument of the method by default seuil=1e-5
        seuil = 9
        while abs(delta_epsilon_markers) > seuil:
            q_first_ik_not_all_frames = q_step_2[:, self.list_frames_param_step]

            markers_xp_data_not_all_frames = self.markers[:, :, self.list_frames_param_step]

            print("seuil", seuil, "delta", abs(delta_epsilon_markers))

            epsilon_markers_n_minus_1 = epsilon_markers_n
            # step 1 - param opt
            param_opt = optimize.minimize(
                fun=self.objective_function_param,
                args=(
                    q_first_ik_not_all_frames,
                    q0,
                    markers_xp_data_not_all_frames
                ),
                x0=p,
                bounds=bounds[10:16],
                method="trust-constr",
                jac="3-point",
                tol=1e-5,
            )
            print(param_opt.x)

            # todo: work with indexes
            self.q_ik_initial_guess[self.parameter_idx[0]: self.parameter_idx[-1] + 1, :] = np.array(
                [param_opt.x] * self.q_ik_initial_guess.shape[1]
            ).T
            p = param_opt.x
            q_output[self.parameter_idx[0]: self.parameter_idx[-1] + 1, :] = \
                np.array([param_opt.x] * q_output.shape[1]).T

            # step 2 - ik step
            # todo : verify the metric of the step 2 please make a RMSE
            q_out, epsilon_markers_n = self.step_2(
                p=p,
                bounds=get_range_q(self.biorbd_model),
                nb_frames=self.nb_frames_ik_step,
                q_output=q_output,
            )

            delta_epsilon_markers = epsilon_markers_n - epsilon_markers_n_minus_1
            print("delta_epsilon_markers:", delta_epsilon_markers)
            print("epsilon_markers_n:", epsilon_markers_n)
            print("epsilon_markers_n_minus_1:", epsilon_markers_n_minus_1)
            iteration += 1
            print("iteration:", iteration)
            self.q_ik_initial_guess = q_output

        return q_out, p

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
        list_frames = random.sample(range(frames), frames_needed) if not all else [i for i in range(frames)]

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
        # todo: hardcoded names, put it as an argument of the method
        table5_xyz = vect_pos_markers[
                     self.markers_model.index("Table:Table5") * 3: self.markers_model.index("Table:Table5") * 3 + 3
                     ][:]
        table_xp = table_markers[:, 0].tolist()
        table6_xy = vect_pos_markers[
                    self.markers_model.index("Table:Table6") * 3: self.markers_model.index("Table:Table6") * 3 + 3][
                    :2
                    ]
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

    def penalty_markers_thorax(self, vect_pos_markers: np.ndarray, thorax_markers: np.ndarray):
        #  todo: refactor penalty_open_loop_markers and jacobian
        """
        The penalty function which minimize the difference between thorax markers position from experimental data
        and from the model

        Parameters
        ----------
        markers_names: list[str]
            The list of markers names
        vect_pos_markers: np.ndarray
            The generalized coordinates from the model
        thorax_markers: np.ndarray
            The thorax markers position form experimental data

        Return
        ------
        The value of the penalty function
        """
        thorax_list_model = []
        thorax_list_xp = []
        for j, name in enumerate(self.markers_model):
            if name != "Table:Table5" and name != "Table:Table6":
                # todo: next trainee, remove the hardcoded makers name and specify it as a new attribute of the class
                mark = vect_pos_markers[self.markers_model.index(name) * 3: self.markers_model.index(name) * 3 + 3][
                       :].tolist()
                thorax = thorax_markers[:, self.markers_model.index(name)].tolist()
                thorax_list_model += mark
                thorax_list_xp += thorax

        return thorax_list_model, thorax_list_xp

    def penalty_rotation_matrix(self, x_with_p: np.ndarray):
        """
        The penalty function which force the model to stay horizontal

        Parameters
        ----------
        biorbd_model: biorbd.Model
            The biorbd model
        x_with_p: np.ndarray
            Generalized coordinates for all dof, unique for all frames

        Return
        ------
        The value of the penalty function
        """
        rotation_matrix = self.biorbd_model.globalJCS(x_with_p, self.biorbd_model.nbSegment() - 1).to_array()

        rot_matrix_list_model = [
            rotation_matrix[2, 0],
            rotation_matrix[2, 1],
            rotation_matrix[0, 2],
            rotation_matrix[1, 2],
            (1 - rotation_matrix[2, 2]),
        ]
        rot_matrix_list_xp = [0] * len(rot_matrix_list_model)

        return rot_matrix_list_model, rot_matrix_list_xp

    def penalty_q_thorax(self, x, q_init):
        """
        Minimize the q of thorax

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

    def objective_function_param(
            self,
            p0: np.ndarray,
            x: np.ndarray,
            x0: np.ndarray,
            markers_calibration: np.ndarray
    ):
        """
        Objective function

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
        list_frames: list[int]
            The list of frames on which we will do the calibration
        markers_names: list[str]
            The list of markers names

        Return
        ------
        The value of the objective function
        """
        nb_dof = x.shape[0]
        n_adjust = p0.shape[0]
        # todo: remove the hard coded
        n_bras = nb_dof - n_adjust - 6

        # be filled in the loop
        table5_xyz_all_frames = 0
        table6_xy_all_frames = 0
        mark_out_all_frames = 0
        rotation_matrix_all_frames = 0

        Q = np.zeros(nb_dof)

        Q[n_bras: n_bras + n_adjust] = p0

        for f, frame in enumerate(self.list_frames_param_step):
            # todo: remove the HARD CODED
            thorax_markers = markers_calibration[:, 0:14, f]
            table_markers = markers_calibration[:, 14:, f]

            # todo: work with parameters and kinematic indexes
            Q[:n_bras] = x[:n_bras, f]
            Q[n_bras + n_adjust:] = x[n_bras + n_adjust:, f]

            markers_model = self.biorbd_model.markers(Q)

            vect_pos_markers = np.zeros(3 * len(markers_model))

            for m, value in enumerate(markers_model):
                vect_pos_markers[m * 3: (m + 1) * 3] = value.to_array()

            table_model, table_xp = self.penalty_table_markers(vect_pos_markers, table_markers)

            table5_xyz = np.linalg.norm(np.array(table_model[:3]) - np.array(table_xp[:3])) ** 2
            table5_xyz_all_frames += table5_xyz

            table6_xy = np.linalg.norm(np.array(table_model[3:]) - np.array(table_xp[3:])) ** 2
            table6_xy_all_frames += table6_xy

            thorax_list_model, thorax_list_xp = self.penalty_markers_thorax(vect_pos_markers,
                                                                            thorax_markers)

            mark_out = 0
            for j in range(len(thorax_markers[0, :])):
                mark = np.linalg.norm(np.array(thorax_list_model[j:j + 3]) - np.array(thorax_list_xp[j:j + 3])) ** 2
                mark_out += mark
            mark_out_all_frames += mark_out

            rot_matrix_list_model, rot_matrix_list_xp = self.penalty_rotation_matrix(Q)

            rotation_matrix = 0
            for i in rot_matrix_list_model:
                rotation_matrix += i ** 2

            rotation_matrix_all_frames += rotation_matrix

            q_continuity_diff_model, q_continuity_diff_xp = self.penalty_q_thorax(Q, x0)
            # Minimize the q of thorax
            q_continuity = np.sum((np.array(q_continuity_diff_model) - np.array(q_continuity_diff_xp)) ** 2)

            pivot_diff_model, pivot_diff_xp = self.theta_pivot_penalty(Q)
            pivot = (pivot_diff_model[0] - pivot_diff_xp[0]) ** 2

            x0 = Q
        # todo: move the weights in init
        return 100000 * (table5_xyz_all_frames + table6_xy_all_frames) + \
               10000 * mark_out_all_frames + \
               100 * rotation_matrix_all_frames + \
               50000 * pivot + \
               500 * q_continuity

    def ik_step(
            self,
            x: np.ndarray,
            p: np.ndarray,
            table_markers: np.ndarray,
            thorax_markers: np.ndarray,
            q_init: np.ndarray,
    ):
        """
        Objective function

        Parameters
        ----------
        x: np.ndarray
            Generalized coordinates for all dof except those between ulna and piece 7, unique for all frames
        p: np.ndarray
            Generalized coordinates between ulna and piece 7
        table_markers: np.ndarray
            (3 x n_markers_on_table x n_frames) marker values for all frames
        thorax_markers: np.ndarray
            (3 x n_markers_on_wu_model x n_frames) marker values for all frames
        markers_names: list(str)
            The list of markers names
        q_init: np.ndarray
            The initial values of generalized coordinates fo the actual frame

        Return
        ------
        The value of the objective function
        """
        x_with_p = np.zeros(self.biorbd_model.nbQ())
        x_with_p[:self.parameter_idx[0]] = x[:self.parameter_idx[0]]
        x_with_p[self.parameter_idx[0]: self.parameter_idx[-1] + 1] = p
        x_with_p[self.parameter_idx[-1] + 1:] = x[self.parameter_idx[0]:]

        markers_model = self.biorbd_model.markers(x_with_p)

        vect_pos_markers = np.zeros(3 * len(markers_model))

        for m, value in enumerate(markers_model):
            vect_pos_markers[m * 3: (m + 1) * 3] = value.to_array()

        # Put the pivot joint vertical
        table_model, table_xp = self.penalty_table_markers(vect_pos_markers, table_markers)

        # Minimize difference between thorax markers from model and from experimental data
        thorax_list_model, thorax_list_xp = self.penalty_markers_thorax(vect_pos_markers,
                                                                        thorax_markers)

        # Force the model horizontality
        # rot_matrix_list_model, rot_matrix_list_xp = penalty_rotation_matrix(self.biorbd_model, x_with_p)

        # Minimize the q of thorax
        q_continuity_diff_model, q_continuity_diff_xp = self.penalty_q_thorax(x, q_init)

        # Force part 1 and 3 to not cross
        pivot_diff_model, pivot_diff_xp = self.theta_pivot_penalty(x_with_p)

        # We add our vector to the main lists
        diff_model = table_model + thorax_list_model + q_continuity_diff_model + pivot_diff_model
        diff_xp = table_xp + thorax_list_xp + q_continuity_diff_xp + pivot_diff_xp

        # We converted our list into array in order to be used by least_square
        diff_tab_model = np.array(diff_model)
        diff_tab_xp = np.array(diff_xp)

        # We created the difference vector
        diff = diff_tab_xp - diff_tab_model

        # todo: move the weight in init
        # We created a vector which contains the weight of each penalty
        weight_table = [100000] * len(table_xp)
        weight_thorax = [10000] * len(thorax_list_xp)
        # weight_rot_matrix = [100] * len(rot_matrix_list_xp)
        weight_theta_13 = [50000]
        weight_continuity = [500] * x.shape[0]

        weight_list = weight_table + weight_thorax + weight_continuity + weight_theta_13

        return diff * weight_list

    def step_2(
            self,
            p,
            bounds,
            nb_frames,
            q_output,
    ):
        # todo: docstring

        index_table_markers = [i for i, value in enumerate(self.markers_model) if "Table" in value]
        index_wu_markers = [i for i, value in enumerate(self.markers_model) if "Table" not in value]

        # build the bounds for step 2
        bounds_without_p_1_min = bounds[0][:self.parameter_idx[0]]
        bounds_without_p_2_min = bounds[0][self.parameter_idx[-1] + 1:]
        bounds_without_p_1_max = bounds[1][:self.parameter_idx[0]]
        bounds_without_p_2_max = bounds[1][self.parameter_idx[-1] + 1:]

        bounds_without_p = (
            np.concatenate((bounds_without_p_1_min, bounds_without_p_2_min)),
            np.concatenate((bounds_without_p_1_max, bounds_without_p_2_max)),
        )

        for f in range(self.nb_frames_ik_step):
            # todo : comment here

            x0_1 = self.q_ik_initial_guess[:self.parameter_idx[0], 0] if f == 0\
                else q_output[:self.parameter_idx[0], f - 1]

            x0_2 = (
                self.q_ik_initial_guess[self.parameter_idx[-1] + 1:, 0]
                if f == 0
                else q_output[self.parameter_idx[-1] + 1, f - 1]
            )
            x0 = np.concatenate((x0_1, x0_2))
            IK_i = optimize.least_squares(
                fun=self.ik_step,
                args=(
                    p,
                    self.markers[:, index_table_markers, f],
                    self.markers[:, index_wu_markers, f],
                    x0,
                ),
                x0=x0,  # x0 q sans p
                bounds=bounds_without_p,
                method="trf",
                jac="3-point",
                xtol=1e-5,
            )

            q_output[:self.parameter_idx[0], f] = IK_i.x[:self.parameter_idx[0]]
            q_output[self.parameter_idx[-1] + 1:, f] = IK_i.x[self.parameter_idx[0]:]

            markers_model = self.biorbd_model.markers(q_output[:, f])
            thorax_markers = self.markers[:, index_wu_markers[0]:index_wu_markers[-1]+1, f]
            markers_to_compare = self.markers[:, :, f]
            espilon_markers = 0

            # sum of squared norm of difference
            # todo: all markers not only thorax hardcoded
            # todo: next trainee, verify the metric
            for j in range(len(thorax_markers[0, :])):
                mark = np.linalg.norm(markers_model[j].to_array()[:] - markers_to_compare[:, j]) ** 2
                espilon_markers += mark

        print("step 2 done")

        return q_output, espilon_markers
