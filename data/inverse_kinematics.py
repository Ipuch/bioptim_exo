"""
This script loads all the c3d files contained in a folder and generates .txt files with the values of the generalized
coordinates and generalized velocities for each dof of the biorbd model and at each shooting points.

This examples shows how to
    1. Load a model
    2. Generate data (should be acquired via real data)
    3. Create a Kalman filter
    4. Apply the Kalman filter (inverse kinematics)
    5. Plot the kinematics (Q), velocity (Qdot) and acceleration (Qddot)
    6. Create a .txt for each c3d which contains the generalized coordinates, velocities and torques

Please note that this example will work only with the Eigen backend.
Please also note that kalman will be VERY slow if compiled in debug
"""

import numpy as np
import biorbd
import ezc3d
from pyomeca import Markers
import glob
import os
from pathlib import Path
from utils import get_c3d_frequencies, get_unit_division_factor, apply_offset, plot_dof

try:
    import bioviz

    biorbd_viz_found = True
except ModuleNotFoundError:
    biorbd_viz_found = False

# Load a predefined model
model_path = "../models/wu_converted_definitif_inverse_kinematics.bioMod"
model = biorbd.Model(model_path)

file_path = Path("../event_tracking/")
file_list = list(file_path.glob("*.c3d"))  # We get the file names with a .c3d extension

for file in file_list:
    c3d = ezc3d.c3d(file.name)  # c3d files are loaded as ezc3d object
    print(file.name)

    # initialization of kalman filter
    freq = get_c3d_frequencies(c3d)  # Hz
    params = biorbd.KalmanParam(freq)
    kalman = biorbd.KalmanReconsMarkers(model, params)

    Q = biorbd.GeneralizedCoordinates(model)
    Qdot = biorbd.GeneralizedVelocity(model)
    Qddot = biorbd.GeneralizedAcceleration(model)
    tau = model.InverseDynamics(Q, Qdot, Qddot)

    # create the list of marker from the .biomod file
    marker_names = [model.markerNames()[i].to_string() for i in range(len(model.markerNames()))]

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
    qddot_recons = np.ndarray((model.nbQddot(), n_frames))
    tau_recons = np.ndarray((model.nbQ(), n_frames))   # ??

    for i, targetMarkers in enumerate(markersOverFrames):
        kalman.reconstructFrame(model, targetMarkers, Q, Qdot, Qddot)
        q_recons[:, i] = Q.to_array()
        qdot_recons[:, i] = Qdot.to_array()
        qddot_recons[:, i] = Qddot.to_array()
        tau_recons[:, i] = tau.to_array()

    q_recons_old = q_recons.copy()

    # list_dof = [11, 12, 13]
    list_dof = [i for i in range(16)]
    q_recons = apply_offset(model, q_recons, list_dof, 2 * np.pi)
    plot_dof(q_recons_old, q_recons, model)

    np.savetxt(os.path.splitext(file)[0] + "_q.txt", q_recons)
    np.savetxt(os.path.splitext(file)[0] + "_qdot.txt", qdot_recons)
    np.savetxt(os.path.splitext(file)[0] + "_qddot.txt", qddot_recons)
    np.savetxt(os.path.splitext(file)[0] + "_tau.txt", tau_recons)

    if biorbd_viz_found:
        b = bioviz.Viz(loaded_model=model, show_muscles=False)
        b.load_experimental_markers(markers)
        b.load_movement(q_recons)
        b.exec()

print(Xmarkers)
