"""
This file contains functions to plot extra information for the optimal control problem of the Miller.
"""

import numpy as np
import biorbd_casadi as biorbd
from bioptim import PlotType, NonLinearProgram, OptimalControlProgram


def marker_model(x, nlp: NonLinearProgram, idx):
    """
    Compute the linear momentum of the system.

    Parameters
    ----------
    x
        State vector
    nlp: NonLinearProgram
        Non linear program
    """
    m = nlp.model
    q = x[:m.nbQ(),:]

    marker_pos = np.zeros((3, x.shape[1]))
    marker_pos_func = biorbd.to_casadi_func("marker_pos", m.markers(nlp.states["q"].mx)[idx].to_mx(), nlp.states["q"].mx)

    for i, qi in enumerate(q.T):
        marker_pos[:, i] = np.squeeze(marker_pos_func(qi))

    return marker_pos


def marker_ref(t, x, nlp: NonLinearProgram, index_marker):
    """
    Compute the angular momentum of the system.

    Parameters
    ----------
    x
        State vector
    nlp: NonLinearProgram
        Non linear program
    """
    dt = nlp.dt
    if isinstance(t, float) and np.isnan(t):
        n_list = 0
    else:
        n_list = np.round((t / dt), 2).tolist()
        n_list = [int(i) for i in n_list]
    # keep in mind that if switch objective functions it will not work anymore.
    return nlp.J[1].target[:, index_marker, n_list]


def add_custom_plots(ocp: OptimalControlProgram, id_marker_1: int, id_marker_2: int):
    """
    Add extra plots to the OCP.

    Parameters
    ----------
    ocp: OptimalControlProgram
        Optimal control program
    """
    nlp = ocp.nlp
    ocp.add_plot(
        f"{nlp[0].model.segment(id_marker_1).name().to_string()}",
        lambda t, x, u, p: marker_ref(t, x, nlp[0], id_marker_1),
        legend=[f"{nlp[0].model.segment(id_marker_1).name().to_string()} {'x'}",
                f"{nlp[0].model.segment(id_marker_1).name().to_string()} {'y'}",
                f"{nlp[0].model.segment(id_marker_1).name().to_string()} {'z'}"],
        plot_type=PlotType.STEP,
        node_idx=[nlp[0].dt * i for i in range(0, nlp[0].ns + 1)]
    )
    ocp.add_plot(
        "marker_ref",
        lambda t, x, u, p: marker_ref(t, x, nlp[0], id_marker_2),
        legend=[f"{nlp[0].model.segment(id_marker_2).name().to_string()} {'x'}",
                f"{nlp[0].model.segment(id_marker_2).name().to_string()} {'y'}",
                f"{nlp[0].model.segment(id_marker_2).name().to_string()} {'z'}"],
        plot_type=PlotType.STEP,
        node_idx=[nlp[0].dt * i for i in range(0, nlp[0].ns + 1)]
    )

    ocp.add_plot(
        f"{nlp[0].model.segment(id_marker_1).name().to_string()}",
        lambda t, x, u, p: marker_model(x, nlp[0], id_marker_1),
        legend=[f"{nlp[0].model.segment(id_marker_1).name().to_string()} {'x'}",
                f"{nlp[0].model.segment(id_marker_1).name().to_string()} {'y'}",
                f"{nlp[0].model.segment(id_marker_1).name().to_string()} {'z'}"],
        plot_type=PlotType.PLOT,
        node_idx=None,
    )
    ocp.add_plot(
        "marker_model",
        lambda t, x, u, p: marker_model(x, nlp[0], id_marker_2),
        legend=[f"{nlp[0].model.segment(id_marker_2).name().to_string()} {'x'}",
                f"{nlp[0].model.segment(id_marker_2).name().to_string()} {'y'}",
                f"{nlp[0].model.segment(id_marker_2).name().to_string()} {'z'}"],
        plot_type=PlotType.PLOT,
        node_idx=None,
    )
    return ocp