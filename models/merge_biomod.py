"""
This script merge to text files into one. They are defined by .bioMod files.
"version X" should be specified only once in the final file at the top of the file.
"""
import os


def merge_biomod(file1, file2, output_file):
    """
    Merge two biomod files into one.

    Parameters
    ----------
    file1: str
        The first file to merge
    file2: str
        The second file to merge
    output_file: str
        The output file

    Returns
    -------
    None
    """
    # Check if the files exist
    if not os.path.isfile(file1):
        raise RuntimeError(f"File {file1} does not exist")
    if not os.path.isfile(file2):
        raise RuntimeError(f"File {file2} does not exist")

    # Read the first file
    with open(file1, "r") as f:
        file1_content = f.readlines()
    # Read the second file
    with open(file2, "r") as f:
        file2_content = f.readlines()

    # Check if the files are bioMod files
    if not file1_content[0].startswith("version"):
        raise RuntimeError(f"File {file1} is not a bioMod file")
    if not file2_content[0].startswith("version"):
        raise RuntimeError(f"File {file2} is not a bioMod file")

    # Check if the version is the same
    if file1_content[0] != file2_content[0]:
        raise RuntimeError(f"Files {file1} and {file2} are not the same version")

    # Read and merge "variables" blocks
    variables = {}
    idx1 = 1
    if "variables\n" in file1_content[idx1:]:
        while "variables\n" not in file1_content[idx1]:
            idx1 += 1
        idx1 += 1
        while "endvariables\n" not in file1_content[idx1]:
            var_name, var_value = file1_content[idx1].split()
            variables[var_name] = var_value
            idx1 += 1

    idx2 = 1
    if "variables\n" in file2_content[idx2:]:
        while "variables\n" not in file2_content[idx2]:
            idx2 += 1
        idx2 += 1
        while "endvariables\n" not in file2_content[idx2]:
            var_name, var_value = file2_content[idx2].split()
            variables[var_name] = var_value
            idx2 += 1

    # Merge the files, file1 on top of file2 in the output file
    with open(output_file, "w") as f:
        # Write version
        f.writelines(file1_content[:1])
        # Write merged variables block
        if variables:
            f.write("variables\n")
            for var_name, var_value in variables.items():
                # add an identation before
                f.write(f"\t{var_name} {var_value}\n")
            f.write("endvariables\n")
        # Write the rest of the content
        f.writelines(file1_content[idx1 + 1:])
        f.writelines(file2_content[idx2 + 1:])
    print(f"Files {file1} and {file2} merged into {output_file}")