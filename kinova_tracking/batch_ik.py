"""
converged, like this!
"""

import time
from pathlib import Path

import biorbd
import numpy as np
from ezc3d import c3d

import bioviz
from kinematic_chain_calibration import KinematicChainCalibration
from utils import get_range_q, get_unit_division_factor

# todo: make this script as a class

new_biomod_file_new = "../models/KINOVA_merge_without_floating_base_with_rototrans_template_with_variables.bioMod"
new_model = biorbd.Model(new_biomod_file_new)

# Get all c3d of interest
file_path = Path("../data/")
file_list = list(file_path.glob("F3*01.c3d"))

name_dof = [i.to_string() for i in new_model.nameDof()]
wu_dof = [i for i in name_dof if not "part" in i]
parameters = [i for i in name_dof if "part7" in i]
kinova_dof = [i for i in name_dof if "part" in i and not "7" in i]

nb_dof_wu_model = len(wu_dof)
nb_parameters = len(parameters)

for file in file_list:
    start = time.time()

    print(file.name)
    c3d_ = c3d(str(file.joinpath()))

    labels_markers = c3d_["parameters"]["POINT"]["LABELS"]["value"]

    # Markers trajectories
    points = c3d_["data"]["points"]

    markers_names = [value.to_string() for value in new_model.markerNames()]
    markers = np.zeros((3, len(markers_names), len(points[0, 0, :])))

    labels_markers.append("Table:Table6")
    for i, name in enumerate(markers_names):
        if name in labels_markers:
            if name == "Table:Table6":
                markers[:, i, :] = points[:3, labels_markers.index("Table:Table5"), :] / get_unit_division_factor(
                    c3d_)  # todo: use get_unit_division_factor
            else:
                markers[:, i, :] = points[:3, labels_markers.index(name), :] / get_unit_division_factor(
                    c3d_)  # todo: use get_unit_division_factor

    markers[2, markers_names.index("Table:Table6"), :] = markers[2, markers_names.index("Table:Table6"), :] + 0.1

    # Markers labels in the model
    marker_names_ik = [new_model.markerNames()[i].to_string() for i in range(new_model.nbMarkers())]

    # the actual inverse kinematics
    my_ik = biorbd.InverseKinematics(new_model, markers)
    my_ik.solve("trf")

    nb_dof_wu_model = len(wu_dof)
    nb_parameters = len(parameters)

    nb_frames = len(points[0, 0, :])

    # prepare the size of the output of q
    q_output = np.zeros((new_model.nbQ(), nb_frames))

    q_first_ik = np.zeros((new_model.nbQ(), markers.shape[2]))
    # initialize human dofs with previous results of inverse kinematics
    q_first_ik = my_ik.q  # human

    # get the q_bounds of the model for all dofs
    bounds = [
        (mini, maxi) for mini, maxi in zip(get_range_q(new_model)[0], get_range_q(new_model)[1])
    ]
    kinova_q0 = np.array([(i[0] + i[1]) / 2 for i in bounds[nb_dof_wu_model + nb_parameters:]])

    # initialized q trajectories for each frames for dofs without a priori knowledge of the q (kinova arm here)
    for j in range((q_first_ik[nb_dof_wu_model + nb_parameters:, :].shape[1])):
        q_first_ik[nb_dof_wu_model + nb_parameters:, j] = kinova_q0

    # initialized parameters values
    p = np.zeros(nb_parameters)

    # First IK step - INITIALIZATION
    kcc = KinematicChainCalibration(
        biorbd_model=new_model,
        markers_model=markers_names,
        markers=markers,
        closed_loop_markers=["Table:Table5", "Table:Table6"],
        tracked_markers=markers_names,
        parameter_dofs=parameters,
        kinematic_dofs=wu_dof + kinova_dof,
        weights=np.zeros(70),  #
        q_ik_initial_guess=q_first_ik,
        nb_frames_ik_step=nb_frames,
        nb_frames_param_step=100,
        randomize_param_step_frames=True,
        use_analytical_jacobians=False,
    )

    q_step_2, epsilon = kcc.inverse_kinematics(bounds=get_range_q(new_model), q_output=q_output)

    b = bioviz.Viz(loaded_model=new_model, show_muscles=False, show_floor=False)
    b.load_experimental_markers(markers)
    b.load_movement(q_step_2)

    b.exec()
    epsilon_tab = np.array(epsilon)
    end = time.time()
    print("Duration", end - start)
    print(f"moyenne sur toutes les frames sur tous les markers: {epsilon} m")
