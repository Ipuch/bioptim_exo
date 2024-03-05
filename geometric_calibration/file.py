import biorbd as biorbd_eigen
import biorbd_casadi as biorbd
import numpy as np
from biorbd import InverseKinematics
from pyomeca import Markers
from pyorerun import BiorbdModel, RerunBiorbdPhase

from marker_calibration import MarkerOptimization
from models.biorbd_model import KinovaModel
from models.enums import Models

SHOW_FIRST_IK = True

kinova_model = KinovaModel(model=Models.KINOVA_RIGHT_COMPLETE)
biorbd_casadi_model = biorbd.Model(kinova_model.model_path)
biorbd_eigen_model = biorbd_eigen.Model(kinova_model.model_path)
# c3d_path = "../data_new/kinova_dynamique_pierre.c3d"
c3d_path = "../data_new/kinova_martin_tete_f6_2.c3d"
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
        "fifth_link_top",
        "fifth_link_start_left",
        "seventh_link_start",
        "sixth_link_end",
        # "seventh_link second",
    ],
    prefix_delimiter="Kinova:",
)
xp_markers[:3, :, :] /= 1000
xp_markers.attrs["units"] = "m"

print(xp_markers.sel(channel="Table").to_numpy().mean(axis=1))

numpy_xp_markers = xp_markers.to_numpy()[0:3, :, :]
nb_frames = numpy_xp_markers.shape[2]
equally_spaced_step = int(nb_frames / 20)
t_span = xp_markers.time

ik = InverseKinematics(biorbd_eigen_model, numpy_xp_markers)
q = ik.solve()

if SHOW_FIRST_IK:
    rr_model = BiorbdModel(kinova_model.model_path)
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

marker_bounds = {
    # "Table": {"lbx": [-0.01, -0.01, 0.01], "ubx": [0.01, 0.01, 0.035]},
    "First_link_end": {"lbx": [-0.01, -0.01, 0], "ubx": [0.01, 0.01, 0.04]},
    "second_link_end": {"lbx": [-0.01, -0.01, 0], "ubx": [0.01, 0.01, 0.04]},
    "third_end": {"lbx": [-0.015, -0.01, 0.03], "ubx": [0.015, 0.01, 0.06]},
    "fourth_link_right": {"lbx": [0.001, 0.02, 0.1], "ubx": [0.002, 0.04, 0.2]},
    "fourth_link_left": {"lbx": [0.001, -0.04, 0.1], "ubx": [0.002, -0.02, 0.2]},
    "fifth_link_start_right": {"lbx": [-0.1, 0.01, 0.01], "ubx": [-0.05, 0.03, 0.03]},
    "fifth_link_top": {"lbx": [0.250, 0.0160, 0.012], "ubx": [0.255, 0.017, 0.014]},
    "fifth_link_start_left": {"lbx": [-0.1, 0.01, -0.03], "ubx": [-0.05, 0.03, -0.01]},
    "sixth_link_end": {"lbx": [0.030, -0.01, -0.11], "ubx": [0.040, 0.01, -0.09]},
    "seventh_link_start": {
        "lbx": [0.050, -0.065, -0.03],
        "ubx": [0.060, -0.055, -0.01],
    },
    "seventh_link second": {
        "lbx": [-0.030, 0.040, 0.005],
        "ubx": [-0.00, 0.060, 0.025],
    },
}

equally_spaced_xp_markers = xp_markers.to_numpy()[0:3, :, ::equally_spaced_step]
markers_opti = MarkerOptimization(biorbd_casadi_model, equally_spaced_xp_markers)
result = markers_opti.solve(marker_bounds=marker_bounds)

print("marker_location :", result[1])
for i in range(result[1].shape[1]):
    print("marker_name :", biorbd_eigen_model.markerNames()[i].to_string())
    print("now marker_location :", result[1][:, i])
    print("before marker_location :", biorbd_eigen_model.marker(i, False).to_array())
    print("\n")

if markers_opti.with_root:
    print("now root_location :", result[2][0:3, 0])
    print("before root_location :", biorbd_eigen_model.localJCS(1).trans().to_array())

for i in range(result[1].shape[1]):
    print(f"marker\t{biorbd_eigen_model.markerNames()[i].to_string()}")
    segment_id = biorbd_eigen_model.marker(i, False).parentId()
    print(f"\tparent\t{biorbd_eigen_model.segments()[segment_id].name().to_string()}")
    print(f"\tposition\t{result[1][0, i]}\t{result[1][1, i]}\t{result[1][2, i]}")
    print(f"endmarker")
    print(f"\n")
