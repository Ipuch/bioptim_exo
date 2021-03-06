import biorbd
import ezc3d
import glob
import os
from bioptim import OptimalControlProgram, CostType
from viz_tracking import add_custom_plots
import pandas as pd
from pathlib import Path

# data_path = Path("../tracking/")
# os.chdir("../tracking/")

df = pd.DataFrame(columns=["c3d", "time_to_optimize", "iteration", "cost", "statues"])

file_directory = "2022_05_02"
file_path = Path(f"save/{file_directory}/")
file_list = list(file_path.glob("*.bo"))

for file in file_list:
    name = file.name.removesuffix(file.suffix)
    print(name)
    ocp, sol = OptimalControlProgram.load(file.joinpath())
    ocp.add_plot_penalty(CostType.ALL)
    list_markers = ["MET5", "CLAV_SC"]
    print("time to optimize:", sol.real_time_to_optimize / 60)
    print("iteration:", sol.iterations)
    print("cost:", sol.cost)
    print("status:", sol.status)
    # if sol.status == 1:
    print("let's see the graphs")
    ocp = add_custom_plots(ocp, list_markers)
    # sol.graphs()
    print("--------------")
    cur_dict = dict(
        c3d=file,
        time_to_optimize=sol.real_time_to_optimize / 60,
        iteration=sol.iterations,
        cost=float(sol.cost),
        status=sol.status,
    )
    row_df = pd.DataFrame([cur_dict])
    df = pd.concat([df, row_df], ignore_index=True)

file_names = []
i = 0
# if another file has the same name, it indents the name with 0, 1, ...
for file in glob.glob(f"save/csv_save/*{file_directory}*"):
    file_names.append(file)

for file in file_names:
    i += 1
df.to_csv(f"save/csv_save/recap_{file_directory}_{i}")
