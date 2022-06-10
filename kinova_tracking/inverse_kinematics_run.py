import inverse_kinematics as ik
import time
import numpy as np
from ezc3d import c3d
import biorbd
import os

model_path = "../models/KINOVA_merge_without_floating_base.bioMod"
model_path = "../models/wu_converted_definitif_without_floating_base.bioMod"
biorbd_model = biorbd.Model(model_path)

c3d_path = "../data/kinova_arm/F3_aisselle_01.c3d"
# c3d_path = "../data/F0_aisselle_05.c3d"

c3d = c3d(c3d_path)
points = c3d["data"]["points"]
labels_markers = c3d["parameters"]["POINT"]["LABELS"]["value"]
# labels_markers.append('Table:Table6')

marker_names = [
            biorbd_model.markerNames()[i].to_string() for i in range(biorbd_model.nbMarkers())
        ]
markers = np.zeros((3, len(marker_names), len(points[0, 0, :])))

for i, name in enumerate(marker_names):
    # if name == 'Table:Table6':
    #     markers[:, i, :] = points[:3, labels_markers.index('Table:Table5'), :] / 1000
    # else:
    markers[:, i, :] = points[:3, labels_markers.index(name), :] / 1000

# markers[2, 15, :] = markers[2, 15, :] - 0.1

# markers = markers[:, :, 0:10]
my_ik = ik.InverseKinematics(model_path, markers)
start_c3d = time.time()
my_ik.solve("trf")
end_c3d = time.time()
print("The time used to execute with least square and with a c3d file instead of a c3d path of markers is given below")
print(end_c3d - start_c3d)

my_ik.animate()

# np.savetxt(os.path.splitext(c3d_path)[0] + "_q.txt", my_ik.q)

