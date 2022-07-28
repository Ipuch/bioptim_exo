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


def move_marker(marker_to_move: int, c3d_file: c3d, offset: np.ndarray, ) -> np.array:
    """
    This function applies an offet to a marker

    Parameters
    ----------
    marker_to_move: int
        indices of the marker to move
    c3d_file : c3d
        c3d file to move the markers.
    offset: np.ndarray
        The vector of offset to apply to the markers in mm.

    Returns
    -------
    new_points : np.array
        The markers with the displaced ones at a given distance on the horizontal plane.
    """

    new_points = c3d_file["data"]["points"].copy()
    new_points[0, marker_to_move, :] = (c3d_file["data"]["points"][0, marker_to_move, :] - offset[0])
    new_points[1, marker_to_move, :] = (c3d_file["data"]["points"][1, marker_to_move, :] - offset[1])
    new_points[2, marker_to_move, :] = (c3d_file["data"]["points"][2, marker_to_move, :] - offset[2])

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


if __name__ == "__main__":

    # c3d to treat
    c3d_path = "../data/F3_aisselle_01.c3d"  # todo: use an Enum for all C3D files
    c3d_kinova = c3d(c3d_path)

    # Markers labels in c3d
    labels_markers = c3d_kinova["parameters"]["POINT"]["LABELS"]["value"]

    marker_move = False
    offset = np.array([0, 50, 0])  # [offsetX,offsetY,offsetZ] mm
    print("offset", offset)
    # Markers trajectories
    points_c3d = (
        c3d_kinova["data"]["points"] if not marker_move
        else move_marker(marker_to_move=labels_markers.index("Table:Table5"),
                         c3d_file=c3d_kinova,
                         offset=offset)
    )

    # model for step 1.1
    model_path_without_kinova = "../models/wu_converted_definitif_inverse_kinematics.bioMod" # todo: use an Enum for all models

    # Step 1.1: IK of wu model with floating base
    ik_with_floating_base = IK(model_path=model_path_without_kinova, points=points_c3d, labels_markers_ik=labels_markers)
    # ik_with_floating_base.animate()

    # rewrite the models with the location of the floating base
    template_file_merge = "../models/KINOVA_merge_without_floating_base_with_6_dof_support_template.bioMod"
    new_biomod_file_merge = (
        "../models/KINOVA_merge_without_floating_base_with_6_dof_support_template_with_variables.bioMod"
    ) # todo: use an Enum for all models

    template_file_wu = "../models/wu_converted_definitif_without_floating_base_template.bioMod"
    new_biomod_file_wu = "../models/wu_converted_definitif_without_floating_base_template_with_variables.bioMod"
    # todo: use an Enum for all models

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
    # todo: use a generic function named "duplicate marker"
    labels_markers.append("Table:Table6")
    for i, name in enumerate(markers_names):
        if name in labels_markers:
            if name == "Table:Table6":
                markers[:, i, :] = points_c3d[:3, labels_markers.index("Table:Table5"), :] / 1000
            else:
                markers[:, i, :] = points_c3d[:3, labels_markers.index(name), :] / 1000

    # apply offset to the markers
    # todo: make generic the function "move_marker_table"
    offset = 0.1  # meters
    markers[2, markers_names.index("Table:Table6"), :] = markers[2, markers_names.index("Table:Table6"), :] + offset


    #### TODO: THE SUPPORT CALIBRATION STARTS HERE ####
    #### TODO: THIS SHOULD BE A FUNCTION ####

    name_dof = [i.to_string() for i in biorbd_model_merge.nameDof()]
    wu_dof = [i for i in name_dof if not "part" in i]
    parameters = [i for i in name_dof if "part7" in i]
    kinova_dof = [i for i in name_dof if "part" in i and not "7" in i]

    nb_dof_wu_model = len(wu_dof)
    nb_parameters = len(parameters)  # todo: indicates the list dofs names instead of the number of parameters
    nb_dof_kinova = len(kinova_dof)  # todo: indicates the list dofs names instead of the number of dofs

    # prepare the inverse kinematics of the first step of the algorithm
    # initialize q with zeros
    q_first_ik = np.zeros((biorbd_model_merge.nbQ(), markers.shape[2]))
    # initialize human dofs with previous results of inverse kinematics
    q_first_ik[:nb_dof_wu_model, :] = ik_without_floating_base.q  # human

    nb_frames = markers.shape[2]
    nb_frames_needed = 10
    all_frames = False
    frames_list = random.sample(range(nb_frames), nb_frames_needed) if not all_frames else [i for i in range(nb_frames)] # todo: make a frame selector function
    frames_list.sort()
    print(frames_list)
    print(nb_frames)
    # nb_frames = 50

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

    # First IK step - INITIALIZATION
    q_step_2, epsilon = calibration.step_2_least_square(
        biorbd_model=biorbd_model_merge,
        p=p,
        bounds=get_range_q(biorbd_model_merge),
        nb_dof_wu_model=nb_dof_wu_model,
        nb_parameters=nb_parameters,
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
        nb_dof_wu_model=nb_dof_wu_model, # todo: rename this variable name is misleading
        nb_parameters=nb_parameters, # todo: rename this variable name is misleading
        nb_frames=nb_frames,
        list_frames=frames_list, # todo: Redundant ?
    )

    b = bioviz.Viz(loaded_model=biorbd_model_merge, show_muscles=False, show_floor=False)
    b.load_experimental_markers(markers)
    # b.load_movement(np.array(q0, q0).T)
    b.load_movement(pos_init)

    b.exec()
    print("done")
