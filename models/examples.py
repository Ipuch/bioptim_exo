from utils import add_header, thorax_variables
import os.path

file = "models/wu_converted_definitif_without_floating_base_template.bioMod"
new_file = "models/wu_converted_definitif_without_floating_base_template_with_variables.bioMod"
file_path = "data/F0_manger_05_q.txt"
# print(os.getcwd())
os.chdir(os.path.dirname(os.getcwd()))
# print(os.getcwd())


thorax_values = thorax_variables(file_path)

add_header(file, new_file, thorax_values)
