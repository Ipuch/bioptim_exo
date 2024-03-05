import biorbd
import numpy as np
from pyorerun import BiorbdModel, RerunBiorbdPhase

model_path = "KINOVA_arm_right_complete_7_segments.bioMod"
yo = biorbd.Model(model_path)
for i in range(yo.nbSegment()):
    print(yo.segment(i).characteristics().mesh().path().absolutePath().to_string())

print(model_path)
rr_model = BiorbdModel(model_path)
rerun_biorbd = RerunBiorbdPhase(rr_model)

nb_frames = 5
nb_seconds = 0.01
t_span = np.linspace(0, nb_seconds, nb_frames)

# building some generalized coordinates
q = np.zeros((rr_model.model.nbQ(), nb_frames))
q[5, :] = np.linspace(0, - 2 * np.pi / 3, nb_frames)

rerun_biorbd.set_tspan(t_span)
rerun_biorbd.set_q(q)
rerun_biorbd.rerun("animation")
