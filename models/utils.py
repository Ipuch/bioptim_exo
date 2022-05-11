def add_header(biomod_file_name: str, new_biomod_file_name: str, variables: dict):
    """
    create a header introducing the values associated with each variable used in the file

    Parameters:
    ---------
    biomod_file_name: str
        name of the original biomod file without the header
    new_biomod_file_name: str
        name of new file biomod created with a header
    variables: dict
        dictionary pairing each variable with its default value

    """

    biomod_file = open(new_biomod_file_name, 'w')

    # copy the first line of the original file onto the new file
    with open(biomod_file_name, "r") as file_object:
        biomod_file.write(file_object.readline())

        biomod_file.write("variables")

        # add a line in the header for each variable
        for variable in variables:
            biomod_file.write("\n\t$" + variable + ' ')
            biomod_file.write(str(variables[variable]))

        biomod_file.write("\nendvariables\n")

        # copy the rest of the original file
        for line in file_object:
            biomod_file.write(line)

    biomod_file.close()
