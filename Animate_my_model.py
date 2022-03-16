import bioviz
import numpy as np

# model_name = "KINOVA_arm_deprecated.bioMod"
model_name = "models/KINOVA_arm_reverse_left.bioMod"
# Load the model - for bioviz
biorbd_viz = bioviz.Viz(model_name, show_floor=False, show_gravity_vector=False)
biorbd_viz.exec()
