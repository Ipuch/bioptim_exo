import numpy as np
import biorbd
import ezc3d
from pyomeca import Markers
import glob
import os
import matplotlib.pyplot as plt

# Load a predefined model
model_path = "../models/wu_converted_definitif.bioMod"
model = biorbd.Model(model_path)

file_list = []
for file in glob.glob("*.c3d"):  # We get the files names with a .c3d extension
    file_list.append(file)

for file in file_list:
    c3d = ezc3d.c3d(file)  # c3d files are loaded as ezc3d object

    # initialization of kalman filter
    freq = c3d["parameters"]["POINT"]["RATE"]["value"][0]
    nb_frames = c3d["parameters"]["POINT"]["FRAMES"]["value"][0]
    duration = nb_frames / freq

    c3d_name = os.path.splitext(file)[0]

    print(c3d_name, " Duration:", duration)
