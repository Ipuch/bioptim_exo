"""
This examples shows how to
    1. Load a model
    2. Generate data (should be acquired via real data)
    3. Create a Kalman filter
    4. Apply the Kalman filter (inverse kinematics)
    5. Plot the kinematics (Q), velocity (Qdot) and acceleration (Qddot)
    6. Create a .txt for each c3d which contains the generalized coordinates and velocities

Please note that this example will work only with the Eigen backend.
Please also note that kalman will be VERY slow if compiled in debug
"""

import numpy as np
import biorbd
import ezc3d
from pyomeca import Markers
import glob
import os
import matplotlib.pyplot as plt

try:
    import bioviz

    biorbd_viz_found = True
except ModuleNotFoundError:
    biorbd_viz_found = False


def get_unit_division_factor(c3d_file):
    """
    Allow the users to get the length units of a c3d file

    Parameters
    ----------
    c3d_file: ezc3d
        c3d file converted into an ezc3d object

    Returns
    -------
    The division factor of length units
    """
    factor_str = c3d_file["parameters"]["POINT"]["UNITS"]["value"][0]
    if factor_str == "mm":
        factor = 1000
    elif factor_str == "m":
        factor = 1
    else:
        raise NotImplementedError("This is not implemented for this unit")

    return factor


def get_c3d_frequencies(c3d_file: ezc3d):
    """
    Allow the users to get the length units of c3d

    Parameters
    ----------
    c3d_file: ezc3d
        c3d file converted into an ezc3d object

    Returns
    -------
    The frequencies
    """
    return c3d_file["parameters"]["POINT"]["RATE"]["value"][0]


def plot_dof(q_old, q, biorbd_model):
    count = 0
    for i in range(biorbd_model.nbSegment()):  # we excluded the 3 first dof which correspond to 3 translation
        print(i)
        if biorbd_model.segment(i).nbDof() > 0:
            print(biorbd_model.segment(i))
            for j in range(biorbd_model.segment(i).nbDof()):
                plt.figure()
                plt.title(f"{biorbd_model.segment(i).name().to_string()} {j}")
                old = "old"
                plt.plot(q_old[count, :], label=f"{str(count)} {old}")
                plt.plot(q[count, :], label=str(count))
                plt.plot([biorbd_model.segment(i).QRanges()[j].min()] * q.shape[1])
                plt.plot([biorbd_model.segment(i).QRanges()[j].max()] * q.shape[1])
                count += 1
                plt.legend()
    plt.show()


def get_segment_and_dof_id_from_global_dof(biorbd_model, global_dof):
    """
    Allow the users to get the segment id which correspond to a dof of the model and the id of this dof in the segment

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model
    global_dof: int
        The global id of the dof in the model

    Returns
    -------
    seg_id: int
        The id of the segment which correspond to the dof
    count_dof: int
         The dof id in this segment
    """
    for j, seg in enumerate(biorbd_model.segments()):
        complete_seg_name = model.nameDof()[global_dof].to_string()  # We get "Segment_Name_DofName"
        seg_name = complete_seg_name.replace("_RotX", "")  # We remove "_DofName"
        seg_name = seg_name.replace("_RotY", "")
        seg_name = seg_name.replace("_RotZ", "")
        seg_name = seg_name.replace("_TransX", "")
        seg_name = seg_name.replace("_TransY", "")
        seg_name = seg_name.replace("_TransZ", "")

        if seg.name().to_string() == seg_name:
            seg_name_new = seg_name
            seg_id = j

    dof = model.nameDof()[global_dof].to_string().replace(seg_name_new, "")
    dof = dof.replace("_", "")  # we remove the _ "_DofName"
    count_dof = 0
    while model.segment(seg_id).nameDof(count_dof).to_string() != dof:
        count_dof += 1

    return seg_id, count_dof


def apply_offset(biorbd_model, dof_nb, offset):
    """
    Allow the users to apply an offset on a dof in the model if this dof is out of is range

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model
    dof_nb: int
        The global number of the dof
    offset: float
        The minimum offset that we wanted to apply on the dof

    Returns
    -------
    """
    seg_id, dof_id = get_segment_and_dof_id_from_global_dof(biorbd_model, dof_nb)
    range_min = model.segment(seg_id).QRanges()[dof_id].min()
    range_max = model.segment(seg_id).QRanges()[dof_id].max()
    value = q_recons[dof_nb, :].mean()
    delta_range_max = value - range_max
    delta_range_min = value - range_min

    offset = np.pi
    compteur = 0

    if delta_range_min > 0 and delta_range_max > 0:  # if the value is above range max
        while (delta_range_min > 0 and delta_range_max > 0) and compteur < 100:
            compteur += 1
            q_recons[dof_nb, :] = q_recons[dof_nb, :] - offset
            value = q_recons[dof_nb, :].mean()
            delta_range_max = value - range_max
            delta_range_min = value - range_min

    elif delta_range_min < 0 and delta_range_max < 0:  # if the value is below range min
        while (delta_range_min < 0 and delta_range_max < 0) and compteur < 100:
            compteur += 1
            q_recons[dof_nb, :] = q_recons[dof_nb, :] + offset
            value = q_recons[dof_nb, :].mean()
            delta_range_max = value - range_max
            delta_range_min = value - range_min
    else:
        print("Nothing has to be done.")


# Load a predefined model
model_path = "../models/wu_converted_definitif.bioMod"
model = biorbd.Model(model_path)

file_list = []
for file in glob.glob("*.c3d"):  # We get the files names with a .c3d extension
    file_list.append(file)

for file in file_list:
    c3d = ezc3d.c3d(file)  # c3d files are loaded as ezc3d object

    # initialization of kalman filter
    freq = get_c3d_frequencies(c3d)  # Hz
    params = biorbd.KalmanParam(freq)
    kalman = biorbd.KalmanReconsMarkers(model, params)

    Q = biorbd.GeneralizedCoordinates(model)
    Qdot = biorbd.GeneralizedVelocity(model)
    Qddot = biorbd.GeneralizedAcceleration(model)

    # create the list of marker from the .biomod file
    marker_names = []
    for i in range(len(model.markerNames())):
        marker_names.append(model.markerNames()[i].to_string())

    # retrieve markers from c3d file and load them with ezc3d
    Xmarkers = Markers.from_c3d(file, usecols=marker_names)

    # We converted the units of c3d to units of biomod
    markers = Xmarkers.values[:3, :, :] / get_unit_division_factor(c3d)

    # reshape marker data
    markersOverFrames = []
    for i in range(markers.shape[2]):
        markersOverFrames.append([biorbd.NodeSegment(m) for m in markers[:, :, i].T])

    # initialized q_recons with the right shape.
    n_frames = c3d["parameters"]["POINT"]["FRAMES"]["value"][0]
    q_recons = np.ndarray((model.nbQ(), n_frames))
    qdot_recons = np.ndarray((model.nbQdot(), n_frames))

    for i, targetMarkers in enumerate(markersOverFrames):
        kalman.reconstructFrame(model, targetMarkers, Q, Qdot, Qddot)
        q_recons[:, i] = Q.to_array()
        qdot_recons[:, i] = Qdot.to_array()

    q_recons_old = q_recons.copy()

    apply_offset(model, 11, 2 * np.pi)
    apply_offset(model, 13, 2 * np.pi)

    # plot_dof(q_recons_old, q_recons, model)

    np.savetxt(os.path.splitext(file)[0] + "_q.txt", q_recons)
    np.savetxt(os.path.splitext(file)[0] + "_qdot.txt", qdot_recons)


print(Xmarkers)

# if biorbd_viz_found:
#     b = bioviz.Viz(loaded_model=model, show_muscles=False)
#     b.load_experimental_markers(markers)
#     b.load_movement(q_recons)
#     b.exec()
