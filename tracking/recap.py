import biorbd
import ezc3d
import glob
import os
from bioptim import OptimalControlProgram, CostType
from viz_tracking import add_custom_plots

# data_path = Path("../tracking/")
os.chdir("../tracking/")
file_list = []
for file in glob.glob("save/2022_04_29/*.bo"):  # We get the files names with a .bo extension
    file_list.append(file)

for file in file_list:
    name = os.path.splitext(file)[0]
    print(name.split("/")[-1])
    ocp, sol = OptimalControlProgram.load(file)
    ocp.add_plot_penalty(CostType.ALL)
    list_markers = ["MET5", "CLAV_SC"]
    ocp = add_custom_plots(ocp, list_markers)
    sol.graphs()
    print("time to optimize:", sol.real_time_to_optimize / 60)
    print("iteration:", sol.iterations)
    print("cost:", sol.cost)
    print("status:", sol.status)
    print("--------------")
