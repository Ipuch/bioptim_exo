import biorbd as biorbd_eigen
import biorbd_casadi as biorbd
import numpy as np
from biorbd import InverseKinematics
from pyomeca import Markers
from pyorerun import BiorbdModel, RerunBiorbdPhase

SHOW_FIRST_IK = True

nb_segments = 7
kinova_model_paths = f"KINOVA_arm_right_complete_{nb_segments}_segments.bioMod"
biorbd_casadi_model = biorbd.Model(kinova_model_paths)
biorbd_eigen_model = biorbd_eigen.Model(kinova_model_paths)
# c3d_path = "../data_new/kinova_dynamique_pierre.c3d"
c3d_path = "../data_new/kinova_martin_boire_f3_1.c3d"
xp_markers = Markers.from_c3d(
    c3d_path,
    usecols=[
        "Table",
        "First_link_end",
        "second_link_end",
        "third_end",
        "fourth_link_right",
        "fourth_link_left",
        "fifth_link_start_right",
        "fifth_link_start_left",
        "fifth_link_top",
        "seventh_link_start",  # this is inverted !!!!! ok in .biomod but not in .c3d
        "sixth_link_end",  # this is inverted !!!!! ok in .biomod but not in .c3d
        "seventh_link second",
    ],
    prefix_delimiter="Kinova:",
)
xp_markers[:3, :, :] /= 1000
xp_markers.attrs["units"] = "m"

numpy_xp_markers = xp_markers.to_numpy()[0:3, :, :]
nb_frames = numpy_xp_markers.shape[2]
t_span = xp_markers.time

ik = InverseKinematics(biorbd_eigen_model, numpy_xp_markers)
q = ik.solve("only_lm")

if SHOW_FIRST_IK:
    rr_model = BiorbdModel(kinova_model_paths)
    rerun_biorbd = RerunBiorbdPhase(rr_model)
    rerun_biorbd.set_tspan(t_span)
    rerun_biorbd.set_q(q)
    rerun_biorbd.add_marker_set(
        positions=np.transpose(numpy_xp_markers, (2, 1, 0)),
        name="xp_markers",
        labels=rr_model.marker_names,
        size=0.01,
        color=np.array([0, 0, 0]),
    )
    rerun_biorbd.show_labels(True)
    rerun_biorbd.rerun("animation")
