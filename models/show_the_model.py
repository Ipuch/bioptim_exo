import bioviz
from enums import Models

model_path = Models.WU_INVERSE_KINEMATICS.value
b = bioviz.Viz(model_path, show_floor=False, show_global_ref_frame=False)
b.exec()

