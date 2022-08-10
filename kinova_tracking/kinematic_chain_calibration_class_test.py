"""
converged, like this!
"""

import bioviz
import calibration
import numpy as np
from ezc3d import c3d
import biorbd
from models.utils import add_header, thorax_variables
from utils import get_range_q
import random
from models.enums import Models
from data.enums import TasksKinova
import KinematicChainCalibration as KCC


def move_marker(marker_to_move: int, c3d_point: np.ndarray, offset: np.ndarray, ) -> np.array:
    """
    This function applies an offet to a marker

    Parameters
    ----------
    marker_to_move: int
        indices of the marker to move
    c3d_point: np.ndarray
        Markers trajectories.
    offset: np.ndarray
        The vector of offset to apply to the markers in mm.

    Returns
    -------
    new_points : np.array
        The markers with the displaced ones at a given distance on the horizontal plane.
    """

    new_points = c3d_point.copy()
    new_points[0, marker_to_move, :] = (c3d_point[0, marker_to_move, :] + offset[0])
    new_points[1, marker_to_move, :] = (c3d_point[1, marker_to_move, :] + offset[1])
    new_points[2, marker_to_move, :] = (c3d_point[2, marker_to_move, :] + offset[2])

    return new_points


def IK(model_path: str, points: np.array, labels_markers_ik: list[str]) -> np.array:
    """
    This function computes the inverse kinematics of the model.

    Parameters
    ----------
    model_path : str
        Path to the model.
    points : np.array
        marker trajectories over time
    labels_markers_ik : list[str]
        List of markers labels

    Returns
    -------
    q : np.array
        The generalized coordinates of the model for each frame
    """
    biorbd_model_ik = biorbd.Model(model_path)

    # Markers labels in the model
    marker_names_ik = [biorbd_model_ik.markerNames()[i].to_string() for i in range(biorbd_model_ik.nbMarkers())]

    # reformat the makers trajectories
    markers_ik = np.zeros((3, len(marker_names_ik), len(points[0, 0, :])))
    for i, name in enumerate(marker_names_ik):
        markers_ik[:, i, :] = points[:3, labels_markers_ik.index(name), :] / 1000

    # the actual inverse kinematics
    my_ik = biorbd.InverseKinematics(biorbd_model_ik, markers_ik)
    my_ik.solve("trf")

    return my_ik


def frame_selector(all: bool, frames_needed: int, frames: int):
    """
    Give a list of frames for calibration

    Parameters
    ----------
    all: bool
        True if you want all frames, False if not
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


def kinematic_chain_calibration(
        name_dof: list[str],
        wu_dof: list[str],
        parameters: list[str],
        kinova_dof: list[str],
        q_first_ik: np.ndarray,
        nb_frames: int,
        frames_list: list[int]):
    """
    Support calibration step, give the optimize parameters and generalized coordinates

    Parameters
    ----------
    name_dof: list[str]
        The list of dof name
    wu_dof: list[str]
        The list of dof name from the wu model
    parameters: list[str]
        The list of dof which are parameters for support calibration
    kinova_dof: list[str]
        The list of dof name from kinova model
    q_first_ik: np.ndarray
        The generalized coordinates from the first inverse kinematics
    nb_frames: int
        The total number of frames
     frames_list: list[int]
        The list of frames use for calibration

    Returns
    -------
    pos_ini: np.ndarray
        The generalized coordinates for optimize parameters
     parameters: np.ndarray
        The optimize parameters
    """
    nb_dof_wu_model = len(wu_dof)
    nb_parameters = len(parameters)
    nb_dof_kinova = len(kinova_dof)

    # prepare the size of the output of q
    q_output = np.zeros((biorbd_model_merge.nbQ(), nb_frames))

    # get the bounds of the model for all dofs
    bounds = [
        (mini, maxi) for mini, maxi in zip(get_range_q(biorbd_model_merge)[0], get_range_q(biorbd_model_merge)[1])
    ]
    kinova_q0 = np.array([(i[0] + i[1]) / 2 for i in bounds[nb_dof_wu_model + nb_parameters:]])
    # initialized q trajectories for each frames for dofs without a priori knowledge of the q (kinova arm here)
    for j in range((q_first_ik[nb_dof_wu_model + nb_parameters:, :].shape[1])):
        q_first_ik[nb_dof_wu_model + nb_parameters:, j] = kinova_q0

    # initialized parameters values
    p = np.zeros(nb_parameters)

    kcc = KCC.KinematicChainCalibration(biorbd_model=biorbd_model_merge,
                                        markers_model=markers_names,
                                        markers=markers,
                                        closed_loop_markers=,
                                        tracked_markers=,
                                        parameter_dofs=parameters,
                                        kinematic_dofs=name_dof,
                                        objectives_functions=,
                                        weights=,
                                        q_ik_initial_guess=q_first_ik,
                                        nb_frames_ik_step=nb_frames,
                                        nb_frames_param_step=,
                                        randomize_param_step_frames=True,
                                        use_analytical_jacobians=False)
    kcc.step_2()

    # First IK step - INITIALIZATION
    q_step_2, epsilon = calibration.step_2_least_square(
        biorbd_model=biorbd_model_merge,
        p=p,
        bounds=get_range_q(biorbd_model_merge),
        dof=name_dof,
        wu_dof=wu_dof,
        parameters=parameters,
        kinova_dof=kinova_dof,
        nb_frames=nb_frames,
        list_frames=frames_list,
        q_first_ik=q_first_ik,
        q_output=q_output,
        markers_xp_data=markers,
        markers_names=markers_names,
    )
    # b1 = bioviz.Viz(loaded_model=biorbd_model_merge, show_muscles=False, show_floor=False)
    # b1.load_experimental_markers(markers[:, :, :])
    # # b.load_movement(np.array(q0, q0).T)
    # b1.load_movement(q_step_2)
    #
    # b1.exec()

    # Second step - CALIBRATION
    pos_init, parameters = calibration.arm_support_calibration(
        biorbd_model=biorbd_model_merge,
        markers_names=markers_names,
        markers_xp_data=markers,
        q_first_ik=q_step_2,
        dof=name_dof,
        wu_dof=wu_dof,
        parameters=parameters,
        kinova_dof=kinova_dof,  # todo: rename this variable name is misleading
        nb_frames=nb_frames,
        list_frames=frames_list,  # todo: Redundant ?
    )

    return pos_init, parameters


if __name__ == "__main__":

    # c3d to treat
    c3d_path = TasksKinova.ARMPIT.value
    c3d_kinova = c3d(c3d_path)

    # Markers labels in c3d
    labels_markers = c3d_kinova["parameters"]["POINT"]["LABELS"]["value"]

    marker_move = False
    offset = np.array([0, -50, 0])  # [offsetX,offsetY,offsetZ] mm
    print("offset", offset)
    # Markers trajectories
    points_c3d = (
        c3d_kinova["data"]["points"] if not marker_move
        else move_marker(marker_to_move=labels_markers.index("Table:Table5"),
                         c3d_point=c3d_kinova["data"]["points"],
                         offset=offset)
    )

    # model for step 1.1
    model_path_without_kinova = Models.WU_INVERSE_KINEMATICS.value

    # Step 1.1: IK of wu model with floating base
    ik_with_floating_base = IK(model_path=model_path_without_kinova, points=points_c3d, labels_markers_ik=labels_markers)
    # ik_with_floating_base.animate()

    # rewrite the models with the location of the floating base
    template_file_merge = Models.WU_AND_KINOVA_WITHOUT_FLOATING_BASE_WITH_6_DOF_SUPPORT_TEMPLATE.value
    new_biomod_file_merge = Models.WU_AND_KINOVA_WITHOUT_FLOATING_BASE_WITH_6_DOF_SUPPORT_VARIABLES.value

    template_file_wu = Models.WU_WITHOUT_FLOATING_BASE_TEMPLATE.value
    new_biomod_file_wu = Models.WU_WITHOUT_FLOATING_BASE_VARIABLES.value

    thorax_values = {
        "thoraxRT1": ik_with_floating_base.q[3, :].mean(),
        "thoraxRT2": ik_with_floating_base.q[4, :].mean(),
        "thoraxRT3": ik_with_floating_base.q[5, :].mean(),
        "thoraxRT4": ik_with_floating_base.q[0, :].mean(),
        "thoraxRT5": ik_with_floating_base.q[1, :].mean(),
        "thoraxRT6": ik_with_floating_base.q[2, :].mean(),
    }

    add_header(biomod_file_name=template_file_wu, new_biomod_file_name=new_biomod_file_wu, variables=thorax_values)
    add_header(biomod_file_name=template_file_merge, new_biomod_file_name=new_biomod_file_merge, variables=thorax_values)

    # Step 1.2: IK of wu model without floating base
    ik_without_floating_base = IK(model_path=new_biomod_file_wu, points=points_c3d, labels_markers_ik=labels_markers)
    # ik_without_floating_base.animate()

    # exo for step 2
    biorbd_model_merge = biorbd.Model(new_biomod_file_merge)

    markers_names = [value.to_string() for value in biorbd_model_merge.markerNames()]
    markers = np.zeros((3, len(markers_names), len(points_c3d[0, 0, :])))

    # add the extra marker Table:Table6 to the experimental data based on the location of the Table:Table5

    new_row = np.zeros((points_c3d.shape[0], 1, points_c3d.shape[2]))
    points_c3d = np.append(points_c3d, new_row, axis=1)

    labels_markers.append("Table:Table6")

    points_c3d[:3, labels_markers.index("Table:Table6"), :] = points_c3d[:3, labels_markers.index("Table:Table5"), :]

    # apply offset to the markers
    offset = np.array([0, 0, 100])  # meters
    points_c3d = move_marker(marker_to_move=labels_markers.index("Table:Table6"), c3d_point=points_c3d, offset=offset)

    # in the class of calibration
    for i, name in enumerate(markers_names):
        if name in labels_markers:
            markers[:, i, :] = points_c3d[:3, labels_markers.index(name), :] / 1000


    #### TODO: THE SUPPORT CALIBRATION STARTS HERE ####
    #### TODO: THIS SHOULD BE A FUNCTION ####

    name_dof = [i.to_string() for i in biorbd_model_merge.nameDof()]
    wu_dof = [i for i in name_dof if not "part" in i]
    parameters = [i for i in name_dof if "part7" in i]
    kinova_dof = [i for i in name_dof if "part" in i and not "7" in i]

    nb_dof_wu_model = len(wu_dof)
    nb_parameters = len(parameters)
    nb_dof_kinova = len(kinova_dof)

    # prepare the inverse kinematics of the first step of the algorithm
    # initialize q with zeros
    q_first_ik = np.zeros((biorbd_model_merge.nbQ(), markers.shape[2]))
    # initialize human dofs with previous results of inverse kinematics
    q_first_ik[:nb_dof_wu_model, :] = ik_without_floating_base.q  # human

    nb_frames = markers.shape[2]
    nb_frames_needed = 10
    all_frames = False

    frames_list = frame_selector(all_frames, nb_frames_needed, nb_frames)

    pos_init, parameters = kinematic_chain_calibration(
        name_dof=name_dof,
        wu_dof=wu_dof,
        parameters=parameters,
        kinova_dof = kinova_dof,
        q_first_ik=q_first_ik,
        nb_frames=nb_frames,
        frames_list=frames_list
    )

    b = bioviz.Viz(loaded_model=biorbd_model_merge, show_muscles=False, show_floor=False)
    b.load_experimental_markers(markers)
    # b.load_movement(np.array(q0, q0).T)
    b.load_movement(pos_init)

    b.exec()
    print("done")

    Rototrans_matrix_world_support = biorbd_model_merge.globalJCS(
        pos_init[:, 0], biorbd_model_merge.getBodyBiorbdId("part7")).to_array()

    Rototrans_matrix_ulna_world = biorbd_model_merge.globalJCS(
        pos_init[:, 0], biorbd_model_merge.getBodyBiorbdId("ulna")).transpose().to_array()

    # Finally
    Rototrans_matrix_ulna_support = np.matmul(Rototrans_matrix_ulna_world, Rototrans_matrix_world_support)

    print(Rototrans_matrix_ulna_support)

    rototrans_values = {
        "thoraxRT1": ik_with_floating_base.q[3, :].mean(),
        "thoraxRT2": ik_with_floating_base.q[4, :].mean(),
        "thoraxRT3": ik_with_floating_base.q[5, :].mean(),
        "thoraxRT4": ik_with_floating_base.q[0, :].mean(),
        "thoraxRT5": ik_with_floating_base.q[1, :].mean(),
        "thoraxRT6": ik_with_floating_base.q[2, :].mean(),

        "rotationXX": Rototrans_matrix_ulna_support[0, 0],
        "rotationXY": Rototrans_matrix_ulna_support[0, 1],
        "rotationXZ": Rototrans_matrix_ulna_support[0, 2],
        "translationX": Rototrans_matrix_ulna_support[0, 3],

        "rotationYX": Rototrans_matrix_ulna_support[1, 0],
        "rotationYY": Rototrans_matrix_ulna_support[1, 1],
        "rotationYZ": Rototrans_matrix_ulna_support[1, 2],
        "translationY": Rototrans_matrix_ulna_support[1, 3],

        "rotationZX": Rototrans_matrix_ulna_support[2, 0],
        "rotationZY": Rototrans_matrix_ulna_support[2, 1],
        "rotationZZ": Rototrans_matrix_ulna_support[2, 2],
        "translationZ": Rototrans_matrix_ulna_support[2, 3],

    }

    template_file = "../models/KINOVA_merge_without_floating_base_with_rototrans_template.bioMod"
    new_biomod_file_new = "../models/KINOVA_merge_without_floating_base_with_rototrans_template_with_variables.bioMod"

    add_header(template_file, new_biomod_file_new, rototrans_values)
