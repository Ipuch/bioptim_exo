import biorbd as biorbd_eigen
import numpy as np
from biorbd import InverseKinematics
from pyomeca import Markers
from pyorerun import BiorbdModel, RerunBiorbd

SHOW_IK = True
nb_segments = 3
kinova_model_paths = f"KINOVA_arm_right_complete_{nb_segments}_segments.bioMod"
biorbd_eigen_model = biorbd_eigen.Model(kinova_model_paths)

kinova_model_opt_path = f"KINOVA_arm_right_complete_{nb_segments}_segments_opt.bioMod"
biorbd_eigen_model_opt = biorbd_eigen.Model(kinova_model_opt_path)

# c3d_path = "../data_new/kinova_dynamique_pierre.c3d"
c3d_path = "../data_new/kinova_martin_tete_f6_2.c3d"
xp_markers = Markers.from_c3d(
    c3d_path,
    usecols=[
        "Table",
        "First_link_end",
        "second_link_end",
        # "third_end",
        # "fourth_link_right",
        # "fourth_link_left",
        # "fifth_link_start_right",
        # "fifth_link_top",
        # "fifth_link_start_left",
        # "seventh_link_start",
        # "sixth_link_end",
        # "seventh_link second",
    ],
    prefix_delimiter="Kinova:",
)
xp_markers[:3, :, :] /= 1000
xp_markers.attrs["units"] = "m"

numpy_xp_markers = xp_markers.to_numpy()[0:3, :, :]
nb_frames = numpy_xp_markers.shape[2]
t_span = xp_markers.time

ik = InverseKinematics(biorbd_eigen_model, numpy_xp_markers)
q = ik.solve()

if SHOW_IK:
    rr_sim = RerunBiorbd()
    rr_model = BiorbdModel(kinova_model_paths)
    rr_sim.add_phase(rr_model, t_span, q, phase=0, window="original")
    rr_sim.add_marker_set(
        positions=np.transpose(numpy_xp_markers, (2, 1, 0)),
        name="xp_markers",
        labels=rr_model.marker_names,
        size=0.01,
        color=np.array([0, 0, 0]),
        phase=0,
    )
    rr_model_opt = BiorbdModel(kinova_model_opt_path)
    rr_sim.add_phase(rr_model_opt, t_span, q, phase=1, window="optimized")
    rr_sim.add_marker_set(
        positions=np.transpose(numpy_xp_markers, (2, 1, 0)),
        name="xp_markers",
        labels=rr_model_opt.marker_names,
        size=0.01,
        color=np.array([0, 0, 0]),
        phase=1,
    )
    rr_sim.rerun()
