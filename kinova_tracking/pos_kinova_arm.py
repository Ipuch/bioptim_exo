"""
converged, like this!
"""

import bioviz
import IK_Kinova
import numpy as np
from ezc3d import c3d
import biorbd
from models.utils import add_header, thorax_variables


if __name__ == "__main__":

    model_path_without_kinova = "../models/wu_converted_definitif_inverse_kinematics.bioMod"
    biorbd_model_without_kinova = biorbd.Model(model_path_without_kinova)

    c3d_path = "../data/F3_aisselle_01.c3d"

    c3d = c3d(c3d_path)

    points = c3d["data"]["points"]
    labels_markers = c3d["parameters"]["POINT"]["LABELS"]["value"]
    labels_markers.append("Table:Table6")

    marker_names_without_kinova = [
        biorbd_model_without_kinova.markerNames()[i].to_string() for i in range(biorbd_model_without_kinova.nbMarkers())
    ]
    markers_without_kinova = np.zeros((3, len(marker_names_without_kinova), len(points[0, 0, :])))

    for i, name in enumerate(marker_names_without_kinova):
        markers_without_kinova[:, i, :] = points[:3, labels_markers.index(name), :] / 1000

    my_ik = biorbd.InverseKinematics(biorbd_model_without_kinova, markers_without_kinova)
    my_ik.solve("lm")

    # my_ik.animate()

    thorax_values = {
        "thoraxRT1": my_ik.q[3, :].mean(),
        "thoraxRT2": my_ik.q[4, :].mean(),
        "thoraxRT3": my_ik.q[5, :].mean(),
        "thoraxRT4": my_ik.q[0, :].mean(),
        "thoraxRT5": my_ik.q[1, :].mean(),
        "thoraxRT6": my_ik.q[2, :].mean(),
    }
    old_biomod_file = "../models/KINOVA_merge_without_floating_base_template.bioMod"
    new_biomod_file = "../models/KINOVA_merge_without_floating_base_template_with_variables.bioMod"
    add_header(old_biomod_file, new_biomod_file, thorax_values)

    model_path = new_biomod_file

    biorbd_model = biorbd.Model(model_path)

    q0_rescue = np.array(
        [
            -0.18905905,
            -0.22494988,
            0.17328238,
            0.22634287,
            0.0068234,
            0.18016999,
            0.68947024,
            -0.01252137,
            1.10091368,
            1.0030519,
            0.0,
            0.2618,
            0.3903,
            1.7951,
            0.6878,
            0.3952,
        ]
    )

    q0_1 = my_ik.q[6:, 0]
    # q0_2 = np.array((-0.2892, 0.6695, 0.721, 0.0, 0.0, 0.0))
    q0_2 = np.array((0.0, 0.2618, 0.3903, 1.7951, 0.6878, 0.3952))
    q0 = np.concatenate((q0_1, q0_2))

    markers_names = [value.to_string() for value in biorbd_model.markerNames()]
    markers_list = biorbd_model.markers()

    count = 0
    for i, name in enumerate(markers_names):
        if name in labels_markers:
            count += 1

    markers = np.zeros((3, count, len(points[0, 0, :])))

    for i, name in enumerate(markers_names):
        if name in labels_markers:
            if name == "Table:Table6":
                markers[:, i, :] = points[:3, labels_markers.index("Table:Table5"), :] / 1000
            else:
                markers[:, i, :] = points[:3, labels_markers.index(name), :] / 1000

    markers[2, markers_names.index("Table:Table6"), :] = markers[2, markers_names.index("Table:Table6"), :] + 0.1

    # targetd = markers_list[markers_names.index('grd_contact1')].to_array()  # 0 0 0 for now
    table_markers = markers[:, 14:, 0]

    # targetp_init = markers_list[markers_names.index('mg1')].to_array()
    thorax_markers = markers[:, 0:14, 0]
    xp_data = markers[:, :, :100]
    # pos_init = IK_Kinova.IK_Kinova(biorbd_model, markers_names, q0, table_markers, thorax_markers)
    pos_init = IK_Kinova.IK_Kinova(biorbd_model, markers_names, xp_data, q0, my_ik.q[6:, :])
    q0 = pos_init

    b = bioviz.Viz(loaded_model=biorbd_model, show_muscles=False, show_floor=False)
    b.load_experimental_markers(xp_data)
    # b.load_movement(np.array(q0, q0).T)
    b.load_movement(pos_init)

    b.exec()

    # ocp = prepare_ocp(model, pos_init, pos_fin)
    # ocp.print(to_console=False, to_graph=True)
    # # Custom plots
    # ocp.add_plot_penalty(CostType.ALL)
    #
    # # --- Solve the program --- #
    # show_options = dict(show_bounds=True)
    # solver_options = {
    #     "ipopt.tol": 1e-6,
    #     "ipopt.max_iter": 2000,
    #     "ipopt.hessian_approximation": "exact",  # "exact", "limited-memory"
    #     "ipopt.limited_memory_max_history": 50,
    #     "ipopt.linear_solver": "mumps",  # "ma57", "ma86", "mumps"
    # }
    #
    # sol = ocp.solve()
    # sol.animate()
    #
    # # --- Show results --- #
    # sol.print_cost()
    # # ocp.save(sol, "Kinova.bo")
    # sol.animate()
    # # sol.graphs()
