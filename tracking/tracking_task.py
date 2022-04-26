from viz_tracking import add_custom_plots
from datetime import datetime
from tracking_ocp import TrackingOcp
from bioptim import (
    Solver,
    CostType,
)
import os


def main():
    """
    Get data, then create a tracking problem, and finally solve it and plot some relevant information
    """

    # Define the problem
    with_floating_base = False

    # c3d_path = "../data/F0_aisselle_05.c3d"
    # c3d_path = "../data/F0_aisselle_05_crop.c3d"
    c3d_path = "../data/F0_aisselle_05_crop_2s.c3d"
    n_shooting_points = 100
    nb_iteration = 1000

    my_ocp = TrackingOcp(with_floating_base, c3d_path, n_shooting_points, nb_iteration)

    my_ocp.ocp.add_plot_penalty(CostType.ALL)
    list_markers = ["MET5", "CLAV_SC"]
    my_ocp.ocp = add_custom_plots(my_ocp.ocp, list_markers)

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=True, show_options=dict(show_bounds=True))
    solver.set_maximum_iterations(nb_iteration)
    sol = my_ocp.ocp.solve(solver)
    # sol.print_cost()

    # --- Save --- #
    c3d_str = c3d_path.split("/")
    c3d_name = os.path.splitext(c3d_str[-1])[0]
    save_path = f"save/{c3d_name}_{datetime.now()}"
    save_path = save_path.replace(" ", "_").replace("-", "_").replace(":", "_").replace(".", "_")
    my_ocp.ocp.save(sol, save_path)

    # --- Plot --- #
    sol.graphs(show_bounds=True)
    sol.animate(n_frames=100)


if __name__ == "__main__":
    main()
