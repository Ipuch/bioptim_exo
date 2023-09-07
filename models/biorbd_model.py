import biorbd
from .utils import thorax_variables, add_header
from .enums import Models


class NewModel:
    def __init__(self,
                 model: Models,
                 ):
        self.model = model
        self.model_path = model.value

    def add_header(self, model_template: Models, q_file_path: str):
        thorax_values = thorax_variables(q_file_path)
        add_header(model_template.value, self.model_path, thorax_values)


