import biorbd
import bioviz
import numpy as np
import IK_Kinova

# model_name = "KINOVA_arm_deprecated.bioMod"
model = "KINOVA_arm_reverse.bioMod"

q0 = np.array((0.0, 0.0, 0.0, 0.0, -0.1709, 0.0515, -0.2892, 0.6695, 0.721, 0.0, 0.0, 0.0))
targetd = np.zeros((1, 3))

m = biorbd.Model(model)
X = m.markers()
targetp_init = X[4].to_array()
targetp_fin = X[5].to_array()

pos_init = IK_Kinova.IK_Kinova(model, q0, targetd, targetp_init)
pos_fin = IK_Kinova.IK_Kinova(model, pos_init, targetd, targetp_fin)

pos_init = IK_Kinova.IK_Kinova_RT(model, q0, targetd, targetp_init)
pos_fin = IK_Kinova.IK_Kinova_RT(model, pos_init, targetd, targetp_fin)

q = np.linspace(pos_init, pos_fin, 20).T
biorbd_viz = bioviz.Viz(model)
biorbd_viz.load_movement(q)
biorbd_viz.exec()
