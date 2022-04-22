import numpy as np


def print_dof(path):
    data_loaded = np.loadtxt(path)
    print("RT",
          data_loaded[3].mean(),
          data_loaded[4].mean(),
          data_loaded[5].mean(),
          "xyz",
          data_loaded[0].mean(),
          data_loaded[1].mean(),
          data_loaded[2].mean())


file_path = "F0_aisselle_05_crop_q.txt"

print_dof(file_path)
