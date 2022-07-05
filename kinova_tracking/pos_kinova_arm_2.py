"""
converged, like this!
"""

import bioviz

import IK_Kinova_2
import inverse_kinematics as ik
import numpy as np
from ezc3d import c3d
import biorbd
from models.utils import add_header, thorax_variables

if __name__ == "__main__":

    # c3d to treat
    c3d_path = "../data/F3_aisselle_01.c3d"
    c3d = c3d(c3d_path)

    # model for step 1
    model_path_without_kinova = "../models/wu_converted_definitif_inverse_kinematics.bioMod"
    biorbd_model_without_kinova = biorbd.Model(model_path_without_kinova)

    points = c3d["data"]["points"]

    # Markers labels in c3d and in the model
    labels_markers = c3d["parameters"]["POINT"]["LABELS"]["value"]
    marker_names_without_kinova = [
        biorbd_model_without_kinova.markerNames()[i].to_string() for i in range(biorbd_model_without_kinova.nbMarkers())
    ]

    # reformat the makers trajectories
    markers_without_kinova = np.zeros((3, len(marker_names_without_kinova), len(points[0, 0, :])))
    for i, name in enumerate(marker_names_without_kinova):
        markers_without_kinova[:, i, :] = points[:3, labels_markers.index(name), :] / 1000

    # the actual inverse kinematics
    my_ik = ik.InverseKinematics(model_path_without_kinova, markers_without_kinova)
    my_ik.solve()

    # optional
    my_ik.animate()

    # rewritte the model with the location of the floating base
    template_file = "../models/KINOVA_merge_without_floating_base_template.bioMod"
    new_biomod_file = "../models/KINOVA_merge_without_floating_base_template_with_variables.bioMod"
    # todo: rewritte this part of the code with Thasaarah's function add header
    # todo: to externalize
    thorax_values = {
        "thoraxRT1": my_ik.q[3, :].mean(),
        "thoraxRT2": my_ik.q[4, :].mean(),
        "thoraxRT3": my_ik.q[5, :].mean(),
        "thoraxRT4": my_ik.q[0, :].mean(),
        "thoraxRT5": my_ik.q[1, :].mean(),
        "thoraxRT6": my_ik.q[2, :].mean(),
    }
    add_header(template_file, new_biomod_file, thorax_values)
    biorbd_model = biorbd.Model(new_biomod_file)

    ##################################################
    # model_path_with_floating_base = "../models/KINOVA_merge_inverse_kinematics.bioMod"
    # biorbd_model_with_floating_base = biorbd.Model(model_path_with_floating_base)
    #
    # c3d_path = "../data/F3_aisselle_01.c3d"
    #
    # c3d = c3d(c3d_path)
    #
    # points = c3d["data"]["points"]
    # labels_markers = c3d["parameters"]["POINT"]["LABELS"]["value"]
    # labels_markers.append('Table:Table6')
    #
    # marker_names_with_floating_base = [
    #     biorbd_model_with_floating_base.markerNames()[i].to_string() for i in range(biorbd_model_with_floating_base.nbMarkers())
    # ]
    # markers_without_kinova = np.zeros((3, len(marker_names_with_floating_base), len(points[0, 0, :])))
    #
    # for i, name in enumerate(marker_names_with_floating_base):
    #     if name in labels_markers:
    #         if name == 'Table:Table6':
    #             markers_without_kinova[:, i, :] = points[:3, labels_markers.index('Table:Table5'), :] / 1000
    #         else:
    #             markers_without_kinova[:, i, :] = points[:3, labels_markers.index(name), :] / 1000
    #
    # markers_without_kinova[2, marker_names_with_floating_base.index('Table:Table6'), :] = markers_without_kinova[2, marker_names_with_floating_base.index('Table:Table6'), :] + 0.1
    #
    # my_ik = ik.InverseKinematics(model_path_with_floating_base, markers_without_kinova)
    # my_ik.solve("lm")
    #
    # my_ik.animate()
    #
    # thorax_values = {
    #     "thoraxRT1": my_ik.q[3, :].mean(),
    #     "thoraxRT2": my_ik.q[4, :].mean(),
    #     "thoraxRT3": my_ik.q[5, :].mean(),
    #     "thoraxRT4": my_ik.q[0, :].mean(),
    #     "thoraxRT5": my_ik.q[1, :].mean(),
    #     "thoraxRT6": my_ik.q[2, :].mean(),
    # }
    # old_biomod_file = (
    #     "../models/KINOVA_merge_without_floating_base_template.bioMod"
    # )
    # new_biomod_file = (
    #     "../models/KINOVA_merge_without_floating_base_template_with_variables.bioMod"
    # )
    # add_header(old_biomod_file, new_biomod_file, thorax_values)
    # model_path = new_biomod_file
    # biorbd_model = biorbd.Model(model_path)

    markers_names = [value.to_string() for value in biorbd_model.markerNames()]
    markers = np.zeros((3, len(markers_names), len(points[0, 0, :])))

    for i, name in enumerate(markers_names):
        if name in labels_markers:
            if name == "Table:Table6":
                markers[:, i, :] = points[:3, labels_markers.index("Table:Table5"), :] / 1000
            else:
                markers[:, i, :] = points[:3, labels_markers.index(name), :] / 1000

    markers[2, markers_names.index("Table:Table6"), :] = markers[2, markers_names.index("Table:Table6"), :] + 0.1
    experimental_markers_data = markers[:, :, :100]

    q_first_ik = np.zeros((biorbd_model.nbQ(), markers.shape[2]))
    q_first_ik[:10, :] = my_ik.q[6:16, :]  # human
    q_first_ik[16:, :] = my_ik.q[16:, :]  # exo

    pos_init = IK_Kinova_2.arm_support_calibration(
        biorbd_model, markers_names, experimental_markers_data, q_first_ik[:, :100]
    )

    b = bioviz.Viz(loaded_model=biorbd_model, show_muscles=False, show_floor=False)
    b.load_experimental_markers(experimental_markers_data)
    # b.load_movement(np.array(q0, q0).T)
    b.load_movement(pos_init)

    b.exec()
