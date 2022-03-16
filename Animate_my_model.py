import bioviz
import numpy as np

# model_name = "KINOVA_arm_deprecated.bioMod"
model_name = "models/KINOVA_arm_reverse_left.bioMod"
# Load the model - for bioviz
biorbd_viz = bioviz.Viz(model_name, show_floor=False, show_gravity_vector=False)

# Create a movement
n_frames = 2
q = np.zeros((biorbd_viz.nQ, n_frames))
for ii in range(biorbd_viz.nQ):
    q[ii, :] = np.linspace(-1, 1, n_frames)

# Animate the mode
manually_animate = False
if manually_animate:
    i = 0
    while biorbd_viz.vtk_window.is_active:
        biorbd_viz.set_q(q[:, i])
        i = (i + 1) % n_frames
else:
    biorbd_viz.load_movement(q)
    biorbd_viz.exec()
