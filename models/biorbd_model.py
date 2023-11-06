import biorbd
import numpy as np

from .utils import thorax_variables, add_header
from .enums import Models


class NewModel:
    """
    Upper limb model with the thorax variables, changes the header of the model
    """
    def __init__(self,
                 model: Models,
                 ):
        self.model = model
        self.model_path = model.value

    def add_header(self, model_template: Models, q_file_path: str):
        thorax_values = thorax_variables(q_file_path)
        add_header(model_template.value, self.model_path, thorax_values)


class KinovaModel:
    """
    Upper limb model with the thorax variables, changes the header of the model
    """
    def __init__(self,
                 model: Models,
                 ):
        self.model = model
        self.model_path = model.value

    def add_header(self, model_template: Models, attachment_point_location: np.ndarray, offset: np.ndarray):

        values = {
            "KinovaLocationX": attachment_point_location[0] + offset[0],
            "KinovaLocationY": attachment_point_location[1] + offset[1],
            "KinovaLocationZ": attachment_point_location[2] + offset[2],
                  }

        add_header(model_template.value, self.model_path, values)




