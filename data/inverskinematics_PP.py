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


# Load a predefined model
model_path = "../models/wu_converted_definitif.bioMod"
model = biorbd.Model(model_path)
nq = model.nbQ()
n_frames = 10  # ?

# Generate clapping gesture data
qinit = np.array([0, 0, -0.3, 0.35, 1.15, -0.35, 1.15, 0, 0, 0, 0, 0, 0])
qmid = np.array([0, 0, -0.3, 0.5, 1.15, -0.5, 1.15, 0, 0, 0, 0, 0, 0])
qfinal = np.array([0, 0, -0.3, 0.35, 1.15, -0.35, 1.15, 0, 0, 0, 0, 0, 0])
target_q = np.concatenate((np.linspace(qinit, qmid, n_frames).T, np.linspace(qmid, qfinal, n_frames).T), axis=1)

file_list = []
# for file in glob.glob("*.c3d"):  # We get the files names with a .c3d extension
#     file_list.append(file)
for file in glob.glob("*F0_manger_05.c3d"):  # We get the files names with a .c3d extension
    file_list.append(file)

for file in file_list:
    c3d = ezc3d.c3d(file)
    # initialization of kalman filter
    freq = get_c3d_frequencies(c3d)  # Hz
    params = biorbd.KalmanParam(freq)
    kalman = biorbd.KalmanReconsMarkers(model, params)

    Q = biorbd.GeneralizedCoordinates(model)
    Qdot = biorbd.GeneralizedVelocity(model)
    Qddot = biorbd.GeneralizedAcceleration(model) # keep qddot ?

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

    # snippet to handle conversion of angles -pi and pi
    # todo : afficher tous les ddl avant et apres la modification et plotnles bounds pour chaque ddl, verifier les discontinuites

    q_recons_old = q_recons.copy()
    q_recons[3:, :] = q_recons[3:, :] % (4 * np.pi)
    model.segment(1).QRanges()[0].min()
    count = 0

    # for i in range(len(q_recons[3:, :])+1):  # we excluded the 3 first dof which correspond to 3 translation
    #     if model.segment(3).nbDof() > 0:
    #         count += 1
    #     plt.plot(q_recons_old[i+3, :], label="old")
    #     plt.plot(q_recons[i+3, :], label=str(i+3))
    #     plt.plot(model.segment(i+3).QRanges()[0].min())
    #     plt.plot(model.segment(i+3).QRanges()[0].max())
    #     plt.legend()
    #     plt.show()

    np.savetxt(os.path.splitext(file)[0] + '_q.txt', q_recons)
    np.savetxt(os.path.splitext(file)[0] + '_qdot.txt', qdot_recons)

print(Xmarkers)

if biorbd_viz_found:
    b = bioviz.Viz(loaded_model=model, show_muscles=False)
    b.load_experimental_markers(markers)
    b.load_movement(q_recons)
    b.exec()
