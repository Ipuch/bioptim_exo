from bioptim import OptimalControlProgram
import bioviz
from load_experimental_data import LoadData, C3dData
import os

# print(os.chdir("tracking"))
ocp, sol = OptimalControlProgram.load("save/F0_aisselle_05_crop_3s_weight_100000_converge.bo")
marker_ref = ocp.nlp[0].J[1].target

print(sol.real_time_to_optimize / 60)
model_path = "../models/wu_converted_definitif_without_floating_base_and_thorax_markers.bioMod"
b = bioviz.Viz(model_path=model_path)
b.load_movement(sol.states["q"])  # Q from kalman array(nq, nframes)
b.load_experimental_markers(marker_ref)  # expr markers array(3, nmarkers, nframes)
b.exec()

sol.animate()
sol.graphs()
