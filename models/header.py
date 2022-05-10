def add_header(file_name: str, parameter: dict):
    """
    create a header introducing the values associated with each variable used in the file

    Parameters:
    ---------
    file: str
        name of the file without the header
    parameters: dict
        dictionary pairing each parameter with its default value

    """
    # name the new file
    new_file_name = file_name.removesuffix(".bioMod") + "_with_variables.bioMod"

    biomod_file = open(new_file_name, 'w')

    # copy the first line of the original file onto the new file
    biomod_file = open(new_file_name, 'w')
    with open(file_name, "r") as file_object:
        biomod_file.write(file_object.readline())

        biomod_file.write("variables")

        # add a line for each variable in the header
        for variable in parameter:
            biomod_file.write("\n\t$" + variable + ' ')
            biomod_file.write(str(parameter[variable]))

        biomod_file.write("\nendvariables\n")

        # copy the rest of the original file
        for line in file_object:
            biomod_file.write(line)

    biomod_file.close()


file = "wu_converted_definitif_without_floating_base_template.bioMod"
thorax_values = {
    'thoraxRT1': 1.5136279122166798,
    'thoraxRT2': 0.02508004601838823,
    'thoraxRT3': -0.12026902121482706,
    'thoraxRT4': -0.5902166829985935,
    'thoraxRT5': 0.6147618471392384,
    'thoraxRT6': 0.35577136570126267
}

add_header(file, thorax_values)
