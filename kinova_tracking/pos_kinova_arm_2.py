"""
converged, like this!
"""

import bioviz

import calibration
import inverse_kinematics as ik
import numpy as np
from ezc3d import c3d
import biorbd
from models.utils import add_header, thorax_variables
from utils import get_range_q


def IK(model_path, points, labels_markers_ik):
    biorbd_model_ik = biorbd.Model(model_path)

    # Markers labels in the model
    marker_names_ik = [
        biorbd_model_ik.markerNames()[i].to_string() for i in range(biorbd_model_ik.nbMarkers())
    ]

    # reformat the makers trajectories
    markers_ik = np.zeros((3, len(marker_names_ik), len(points[0, 0, :])))
    for i, name in enumerate(marker_names_ik):
        markers_ik[:, i, :] = points[:3, labels_markers_ik.index(name), :] / 1000

    # the actual inverse kinematics
    my_ik = ik.InverseKinematics(model_path, markers_ik)
    my_ik.solve("trf")

    return my_ik


if __name__ == "__main__":

    # c3d to treat
    c3d_path = "../data/F3_aisselle_01.c3d"
    c3d_kinova = c3d(c3d_path)

    # Markers trajectories
    points_c3d = c3d_kinova["data"]["points"]

    # Markers labels in c3d
    labels_markers = c3d_kinova["parameters"]["POINT"]["LABELS"]["value"]

    # model for step 1.1
    model_path_without_kinova = "../models/wu_converted_definitif_inverse_kinematics.bioMod"

    # Step 1.1: IK of wu model with floating base
    ik_with_floating_base = IK(model_path_without_kinova, points_c3d, labels_markers)

    # rewrite the models with the location of the floating base
    template_file_merge = "../models/KINOVA_merge_without_floating_base_template.bioMod"
    new_biomod_file_merge = "../models/KINOVA_merge_without_floating_base_template_with_variables.bioMod"

    template_file_wu = "../models/wu_converted_definitif_without_floating_base_template.bioMod"
    new_biomod_file_wu = "../models/wu_converted_definitif_without_floating_base_template_with_variables.bioMod"

    # todo: rewrite this part of the code with Thasaarah's function add header
    # todo: to externalize
    thorax_values = {
        "thoraxRT1": ik_with_floating_base.q[3, :].mean(),
        "thoraxRT2": ik_with_floating_base.q[4, :].mean(),
        "thoraxRT3": ik_with_floating_base.q[5, :].mean(),
        "thoraxRT4": ik_with_floating_base.q[0, :].mean(),
        "thoraxRT5": ik_with_floating_base.q[1, :].mean(),
        "thoraxRT6": ik_with_floating_base.q[2, :].mean(),
    }

    add_header(template_file_wu, new_biomod_file_wu, thorax_values)
    add_header(template_file_merge, new_biomod_file_merge, thorax_values)

    # Step 1.2: IK of wu model without floating base
    ik_without_floating_base = IK(new_biomod_file_wu, points_c3d, labels_markers)
    # ik_without_floating_base.animate()

    # exo for step 2
    biorbd_model_merge = biorbd.Model(new_biomod_file_merge)

    markers_names = [value.to_string() for value in biorbd_model_merge.markerNames()]
    markers = np.zeros((3, len(markers_names), len(points_c3d[0, 0, :])))

    labels_markers.append("Table:Table6")
    for i, name in enumerate(markers_names):
        if name in labels_markers:
            if name == "Table:Table6":
                markers[:, i, :] = points_c3d[:3, labels_markers.index("Table:Table5"), :] / 1000
            else:
                markers[:, i, :] = points_c3d[:3, labels_markers.index(name), :] / 1000

    markers[2, markers_names.index("Table:Table6"), :] = markers[2, markers_names.index("Table:Table6"), :] + 0.1

    q_first_ik = np.zeros((biorbd_model_merge.nbQ(), markers.shape[2]))
    q_first_ik[:10, :] = ik_without_floating_base.q  # human

    nb_dof_wu_model = ik_without_floating_base.q.shape[0]  # todo: remove raw hard coded value
    nb_parameters = 6
    nb_dof_kinova = 6
    # nb_frames = markers.shape[2]
    nb_frames = 10
    q_output = np.zeros((biorbd_model_merge.nbQ(), nb_frames))
    bounds = [
        (mini, maxi) for mini, maxi in zip(get_range_q(biorbd_model_merge)[0], get_range_q(biorbd_model_merge)[1])
    ]
    p = np.zeros(6)

    q_step_2 = calibration.step_2(
        biorbd_model_merge,
        p,
        bounds,
        nb_dof_wu_model,
        nb_parameters,
        nb_frames,
        q_first_ik[:, :10],
        q_output[:, :10],
        markers[:, :, :10],
        markers_names
    )
    b1 = bioviz.Viz(loaded_model=biorbd_model_merge, show_muscles=False, show_floor=False)
    b1.load_experimental_markers(markers[:, :, :10])
    # b.load_movement(np.array(q0, q0).T)
    b1.load_movement(q_step_2)

    b1.exec()

    pos_init = calibration.arm_support_calibration(
        biorbd_model_merge,
        markers_names,
        markers[:, :, :10],
        q_step_2[:, :10],
        nb_dof_wu_model,
        nb_parameters,
        nb_frames,
    )

    b = bioviz.Viz(loaded_model=biorbd_model_merge, show_muscles=False, show_floor=False)
    b.load_experimental_markers(markers[:, :, :10])
    # b.load_movement(np.array(q0, q0).T)
    b.load_movement(pos_init)

    b.exec()
    print("done")
