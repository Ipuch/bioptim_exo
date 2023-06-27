import bioviz
from enums import Models

model_path = Models.KINOVA_RIGHT.value
b = bioviz.Viz(model_path, show_floor=False, show_global_ref_frame=False)
b.exec()
