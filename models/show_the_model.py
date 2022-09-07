import bioviz
from enums import Models

model_path = Models.WU_WITHOUT_FLOATING_BASE_QUAT_VARIABLES.value
b = bioviz.Viz(model_path, show_floor=False, show_global_ref_frame=False)
b.exec()
