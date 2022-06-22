import biorbd
import numpy as np


def marker_position(x_init_ref: np.array, n_shooting_points: int):
    """
    Get data, then create a tracking problem, and finally solve it and plot some relevant information
    """
    model_path = "/home/lim/Documents/Stage_Thasaarah/bioptim/bioptim/examples/muscle_driven_ocp/models/arm26.bioMod"
    biorbd_model = biorbd.Model(model_path)
    marker_ref = [m.to_string() for m in biorbd_model.markerNames()]

    q = np.linspace(x_init_ref[0], x_init_ref[1], n_shooting_points + 1)
    all_markers = np.zeros((3, len(marker_ref), n_shooting_points + 1))
    j = 0
    for qi in q:

        markers = biorbd_model.markers(qi)
        for i in range(len(markers)):
            all_markers[:, i, j] = markers[i].to_array() + np.random.random(3) / 100 - 0.005
        j += 1
    return all_markers


n_shooting_points = 25
x_ref = np.array([[0, 0, 0, 0], [-0.6, -0.4, 1.45, 1.3]])
q = np.linspace(x_ref[0, :2], x_ref[1, :2], n_shooting_points + 1)
marker_position(q, n_shooting_points)
