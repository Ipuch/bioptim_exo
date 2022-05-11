from utils import add_header

file = "wu_converted_definitif_without_floating_base_template.bioMod"
new_file = "wu_converted_definitif_without_floating_base_template_with_variables.bioMod"
thorax_values = {
    'thoraxRT1': 1.5136279122166798,
    'thoraxRT2': 0.02508004601838823,
    'thoraxRT3': -0.12026902121482706,
    'thoraxRT4': -0.5902166829985935,
    'thoraxRT5': 0.6147618471392384,
    'thoraxRT6': 0.35577136570126267
}

add_header(file, new_file, thorax_values)
