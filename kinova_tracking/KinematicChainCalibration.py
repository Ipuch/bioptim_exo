from typing import Union
from enum import Enum

import numpy as np

import biorbd

from scipy import optimize
import utils


class ObjectivesFunctions(Enum):
    ALL_OBJECTIVES = "all objectives"
    ALL_OBJECTIVES_WITHOUT_FINAL_ROTMAT = "all objectives without final rotmat"


class KinematicChainCalibration:
    """


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
                 objectives_functions: ObjectivesFunctions,
                 weights: Union[list[float], np.ndarray],
                 q_ik_initial_guess: np.ndarray,  # [n_dof x n_frames]
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
        if markers.shape() == markers_model.shape():
            self.markers = markers
        else:
            raise ValueError(f"markers and markers model must have same shape, markers shape is {markers.shape()},"
                             f" and markers_model shape is {markers_model.shape()}")
        self.closed_loop_markers = closed_loop_markers
        self.tracked_markers = tracked_markers
        self.parameter_dofs = parameter_dofs
        self.kinematic_dofs = kinematic_dofs
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

    # if nb_frames_ik_step> markers.shape[2]:
    # raise error
    # self.nb_frame_ik_step = markers.shape[2] if nb_frame_ik_step is None else nb_frames_ik_step
    #
    # # check for randomize and nb frames.
    #
    # def solve(self, tolerance, use_analytical_jacobians:bool=True, objectives_functions: ObjectivesFunctions)

    # kcc = KinematicChainCalibration()
    # kcc.solve()
    # kkc.results()

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
        #
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
        theta_part1_3 = q[-2] + q[-1]

        theta_part1_3_lim = 7 * np.pi / 10

        if theta_part1_3 > theta_part1_3_lim:
            diff_model_pivot = [theta_part1_3]
            diff_xp_pivot = [theta_part1_3_lim]
        else:  # theta_part1_3_min < theta_part1_3:
            theta_cost = 0
            diff_model_pivot = [theta_cost]
            diff_xp_pivot = [theta_cost]

        return diff_model_pivot, diff_xp_pivot

    def penalty_markers_thorax(self, vect_pos_markers: np.ndarray, thorax_markers: np.ndarray):
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
                mark = vect_pos_markers[self.markers_model.index(name) * 3: self.markers_model.index(name) * 3 + 3][:].tolist()
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
            markers: np.ndarray,
            list_frames: list[int],
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
        markers: np.ndarray
            (3 x n_markers x n_frames) marker values for all frames
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
        n_bras = nb_dof - n_adjust - 6

        table5_xyz_all_frames = 0
        table6_xy_all_frames = 0
        mark_out_all_frames = 0
        rotation_matrix_all_frames = 0
        Q = np.zeros(nb_dof)
        Q[n_bras: n_bras + n_adjust] = p0

        for f, frame in enumerate(list_frames):
            thorax_markers = markers[:, 0:14, f]
            table_markers = markers[:, 14:, f]

            Q[:n_bras] = x[:n_bras, f]
            Q[n_bras + n_adjust:] = x[n_bras + n_adjust:, f]

            markers_model = self.biorbd_model.markers(Q)

            vect_pos_markers = np.zeros(3 * len(markers_model))

            for m, value in enumerate(markers_model):
                vect_pos_markers[m * 3: (m + 1) * 3] = value.to_array()

            table_model, table_xp = self.penalty_table_markers(self.markers_model, vect_pos_markers, table_markers)

            table5_xyz = np.linalg.norm(np.array(table_model[:3]) - np.array(table_xp[:3])) ** 2
            table5_xyz_all_frames += table5_xyz

            table6_xy = np.linalg.norm(np.array(table_model[3:]) - np.array(table_xp[3:])) ** 2
            table6_xy_all_frames += table6_xy

            thorax_list_model, thorax_list_xp = self.penalty_markers_thorax(self.markers_model, vect_pos_markers, thorax_markers)

            mark_out = 0
            for j in range(len(thorax_markers[0, :])):
                mark = np.linalg.norm(np.array(thorax_list_model[j:j + 3]) - np.array(thorax_list_xp[j:j + 3])) ** 2
                mark_out += mark
            mark_out_all_frames += mark_out

            rot_matrix_list_model, rot_matrix_list_xp = self.penalty_rotation_matrix(self.biorbd_model, Q)

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
        x_with_p[:10] = x[:10]
        x_with_p[10:16] = p
        x_with_p[16:] = x[10:]

        markers_model = self.biorbd_model.markers(x_with_p)

        vect_pos_markers = np.zeros(3 * len(markers_model))

        for m, value in enumerate(markers_model):
            vect_pos_markers[m * 3: (m + 1) * 3] = value.to_array()

        # Put the pivot joint vertical
        table_model, table_xp =self. penalty_table_markers(self.markers_model, vect_pos_markers, table_markers)

        # Minimize difference between thorax markers from model and from experimental data
        thorax_list_model, thorax_list_xp = self.penalty_markers_thorax(self.markers_model, vect_pos_markers, thorax_markers)

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
            dof,
            wu_dof,
            parameters,
            kinova_dof,
            nb_frames,
            q_first_ik,
            q_output,
            markers_xp_data,
    ):
        nb_dof_wu_model = len(wu_dof)
        nb_parameters = len(parameters)
        nb_dof_kinova = len(kinova_dof)

        index_table_markers = [i for i, value in enumerate(self.markers_model) if "Table" in value]
        index_wu_markers = [i for i, value in enumerate(self.markers_model) if "Table" not in value]

        # build the bounds for step 2
        bounds_without_p_1_min = bounds[0][:nb_dof_wu_model]
        bounds_without_p_2_min = bounds[0][nb_dof_wu_model + nb_parameters:]
        bounds_without_p_1_max = bounds[1][:nb_dof_wu_model]
        bounds_without_p_2_max = bounds[1][nb_dof_wu_model + nb_parameters:]

        bounds_without_p = (
            np.concatenate((bounds_without_p_1_min, bounds_without_p_2_min)),
            np.concatenate((bounds_without_p_1_max, bounds_without_p_2_max)),
        )

        for f in range(nb_frames):
            # todo : comment here
            x0_1 = q_first_ik[:nb_dof_wu_model, 0] if f == 0 else q_output[:nb_dof_wu_model, f - 1]
            x0_2 = (
                q_first_ik[nb_dof_wu_model + nb_parameters:, 0]
                if f == 0
                else q_output[nb_dof_wu_model + nb_parameters:, f - 1]
            )
            x0 = np.concatenate((x0_1, x0_2))
            IK_i = optimize.least_squares(
                fun=self.ik_step,
                args=(
                    p,
                    markers_xp_data[:, index_table_markers, f],  # todo: remove the raw hard coded walues
                    markers_xp_data[:, index_wu_markers, f],
                    x0,
                ),
                x0=x0,  # x0 q sans p
                bounds=bounds_without_p,
                method="trf",
                jac="3-point",
                xtol=1e-5,
            )

            q_output[:nb_dof_wu_model, f] = IK_i.x[:nb_dof_wu_model]
            q_output[nb_dof_wu_model + nb_parameters:, f] = IK_i.x[nb_dof_wu_model:]

            markers_model = self.biorbd_model.markers(q_output[:, f])
            thorax_markers = markers_xp_data[:, 0:14, f]
            markers_to_compare = markers_xp_data[:, :, f]
            espilon_markers = 0

            for j in range(len(thorax_markers[0, :])):
                mark = np.linalg.norm(markers_model[j].to_array()[:] - markers_to_compare[:, j]) ** 2
                espilon_markers += mark

        print("step 2 done")

        return q_output, espilon_markers

    def arm_support_calibration(
            self,
            biorbd_model: biorbd.Model,
            markers_names: list[str],
            markers_xp_data: np.ndarray,
            q_first_ik: np.ndarray,
            dof,
            wu_dof,
            parameters,
            kinova_dof,
            nb_frames,
            list_frames,
    ):
        """
        Parameters
        ----------
        biorbd_model: biorbd.Model
            The biorbd model
        markers_names: list[str]
            The list of markers names
        markers_xp_data: np.ndarray
            (3 x n_markers x n_frames) marker values for all frames
        q_first_ik: np.ndarray
             Generalized coordinates for all frames all dof
        nb_dof_wu_model: int
            The number of dof for the thorax and the arm
        nb_parameters: int
            The number of dof between the ulna and piece 7
        nb_frames: int
            The number of frames
        Return
        ------
            The optimized Generalized coordinates
        """
        nb_dof_wu_model = len(wu_dof)
        nb_parameters = len(parameters)
        nb_dof_kinova = len(kinova_dof)

        q0 = q_first_ik[:, 0]

        # idx_kinematic_dof = .index("Table:Table5")
        # idx_parametric_dof = [n_dof + 1, ..., n_dof + parameter]

        q_output = np.zeros((biorbd_model.nbQ(), markers_xp_data.shape[2]))

        bounds = [
            (mini, maxi) for mini, maxi in zip(utils.get_range_q(biorbd_model)[0], utils.get_range_q(biorbd_model)[1])
        ]
        # nb_frames = markers_xp_data.shape[2]
        p = q_first_ik[nb_dof_wu_model: nb_dof_wu_model + nb_parameters, 0]

        iteration = 0
        epsilon_markers_n = 10
        epsilon_markers_n_minus_1 = 0
        delta_epsilon_markers = epsilon_markers_n - epsilon_markers_n_minus_1

        seuil = 5e-5
        while abs(delta_epsilon_markers) > seuil:
            q_first_ik_not_all_frames = q_first_ik[:, list_frames]

            markers_xp_data_not_all_frames = markers_xp_data[:, :, list_frames]

            print("seuil", seuil, "delta", abs(delta_epsilon_markers))

            epsilon_markers_n_minus_1 = epsilon_markers_n
            # step 1 - param opt
            param_opt = optimize.minimize(
                fun=self.objective_function_param,
                args=(
                    q_first_ik_not_all_frames,
                    q0,
                    markers_xp_data_not_all_frames,
                    list_frames,
                ),
                x0=p,
                bounds=bounds[10:16],
                method="trust-constr",
                jac="3-point",
                tol=1e-5,
            )
            print(param_opt.x)

            q_first_ik[nb_dof_wu_model: nb_dof_wu_model + nb_parameters, :] = np.array(
                [param_opt.x] * q_first_ik.shape[1]
            ).T
            p = param_opt.x
            q_output[nb_dof_wu_model: nb_dof_wu_model + nb_parameters, :] = np.array(
                [param_opt.x] * q_output.shape[1]).T

            # step 2 - ik step
            q_out, epsilon_markers_n = self.step_2(
                biorbd_model=biorbd_model,
                p=p,
                bounds=utils.get_range_q(biorbd_model),
                dof=dof,
                wu_dof=wu_dof,
                parameters=parameters,
                kinova_dof=kinova_dof,
                nb_frames=nb_frames,
                list_frames=list_frames,
                q_first_ik=q_first_ik,
                q_output=q_output,
                markers_xp_data=markers_xp_data,
                markers_names=markers_names,
            )

            delta_epsilon_markers = epsilon_markers_n - epsilon_markers_n_minus_1
            print("delta_epsilon_markers:", delta_epsilon_markers)
            print("epsilon_markers_n:", epsilon_markers_n)
            print("epsilon_markers_n_minus_1:", epsilon_markers_n_minus_1)
            iteration += 1
            print("iteration:", iteration)
            q_first_ik = q_output

        # return support parameters, q_output
        return q_out, p
