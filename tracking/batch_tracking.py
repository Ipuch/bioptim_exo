from viz_tracking import add_custom_plots
from datetime import datetime
from tracking_ocp import TrackingOcp
from bioptim import (
    Solver,
    CostType,
)
import ezc3d
from pathlib import Path


def run_ocp(with_floating_base, c3d_file, n_shooting_points, nb_iteration):

    list_markers = ["SEML", "MET2", "MET5"]
    my_ocp = TrackingOcp(
        with_floating_base=with_floating_base,
        c3d_path=str(c3d_file.joinpath()),
        n_shooting_points=n_shooting_points,
        nb_iteration=nb_iteration,
        markers_tracked=list_markers,
    )

    my_ocp.ocp.add_plot_penalty(CostType.ALL)
    my_ocp.ocp = add_custom_plots(my_ocp.ocp, list_markers)

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver.set_linear_solver("ma57")
    solver.set_maximum_iterations(nb_iteration)
    sol = my_ocp.ocp.solve(solver)
    # sol.print_cost()

    # --- Save --- #
    c3d_name = c3d_file.name.removesuffix(c3d_file.suffix)
    save_path = f"save/{datetime.now().date()}/{c3d_name}_batch_{datetime.now()}"
    save_path = save_path.replace(" ", "_").replace("-", "_").replace(":", "_").replace(".", "_")
    my_ocp.ocp.save(sol, save_path)

    # --- Plot --- #
    # sol.graphs(show_bounds=True)
    # sol.animate(n_frames=100)


def main():
    """
    Get data, then create a tracking problem, and finally solve it and plot some relevant information
    """

    # Define the problem
    with_floating_base = False

    file_path = Path("../data/")
    file_list = list(file_path.glob("*.c3d"))

    for file in file_list:
        print(file.name)
        c3d = ezc3d.c3d(str(file.joinpath()))
        # c3d_path = f"../data/{file.name}"
        freq = c3d["parameters"]["POINT"]["RATE"]["value"][0]
        nb_frames = c3d["parameters"]["POINT"]["FRAMES"]["value"][0]
        duration = nb_frames / freq
        n_shooting_points = int(duration * 100)
        nb_iteration = 0
        run_ocp(with_floating_base, file, n_shooting_points, nb_iteration)


if __name__ == "__main__":
    main()
