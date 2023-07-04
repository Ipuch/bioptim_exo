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

    # Merge the files, file1 on top of file2 in the output file
    with open(output_file, "w") as f:
        f.writelines(file1_content)
        f.writelines(file2_content[1:])
    print(f"Files {file1} and {file2} merged into {output_file}")

