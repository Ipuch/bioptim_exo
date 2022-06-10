import bioviz
import numpy as np

# model_name = "KINOVA_arm_deprecated.bioMod"
model_name = "models/KINOVA_arm_reverse_left.bioMod"

# Load the model - for bioviz
biorbd_viz = bioviz.Viz(
    model_name,
    show_floor=False,
    show_gravity_vector=False,
    show_meshes=True,
    show_global_center_of_mass=False,
    show_segments_center_of_mass=False,
    show_global_ref_frame=False,
    show_local_ref_frame=True,
    show_markers=True,
    show_muscles=False,
    show_wrappings=False,
    mesh_opacity=0.97,
)
biorbd_viz.exec()
