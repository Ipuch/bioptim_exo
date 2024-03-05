import biorbd
import numpy as np
from pyorerun import BiorbdModel, PhaseRerun

model_path = "KINOVA_arm_right_complete_7_segments_extra_markers.bioMod"
yo = biorbd.Model(model_path)
for i in range(yo.nbSegment()):
    print(yo.segment(i).characteristics().mesh().path().absolutePath().to_string())

print(model_path)
rr_model = BiorbdModel(model_path)

nb_frames = 1
nb_seconds = 1
t_span = np.linspace(0, nb_seconds, nb_frames)

# building some generalized coordinates
q = np.zeros((rr_model.model.nbQ(), nb_frames))

rerun_biorbd = PhaseRerun(t_span)
rerun_biorbd.add_animated_model(rr_model, q)
rerun_biorbd.biorbd_models.rerun_models[0].ligaments.ligament_properties.radius = 0.005
rerun_biorbd.rerun("KINOVA")
