import biorbd
import numpy as np
from pyorerun import BiorbdModel, RerunBiorbdPhase

from enums import Models

model_path = Models.KINOVA_RIGHT_COMPLETE.value
yo = biorbd.Model(model_path)
for i in range(yo.nbSegment()):
    print(yo.segment(i).characteristics().mesh().path().absolutePath().to_string())

print(model_path)
rr_model = BiorbdModel(model_path)
rerun_biorbd = RerunBiorbdPhase(rr_model)

nb_frames = 200
nb_seconds = 1
t_span = np.linspace(0, nb_seconds, nb_frames)

# building some generalized coordinates
q = np.zeros((rr_model.model.nbQ(), nb_frames))

rerun_biorbd.set_tspan(t_span)
rerun_biorbd.set_q(q)
rerun_biorbd.rerun("animation")
