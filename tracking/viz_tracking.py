"""
This file contains functions to plot extra information for the optimal control problem of the Miller.
"""
from typing import Union
import numpy as np
import biorbd_casadi as biorbd
from bioptim import PlotType, NonLinearProgram, OptimalControlProgram


def marker_model(x: list, nlp: NonLinearProgram, idx: int):
    """
    Get the position of the model's markers

    Parameters
    ----------
    x: list
        values of the states
    nlp: NonLinearProgram
        Non linear program
    idx: int
        Index of the marker
    Returns:
        position of the markers on the three axis
    """
    m = nlp.model
    q = x[: m.nbQ(), :]

    marker_pos = np.zeros((3, x.shape[1]))
    marker_pos_func = biorbd.to_casadi_func(
        "marker_pos", m.markers(nlp.states["q"].mx)[idx].to_mx(), nlp.states["q"].mx
    )

    for i, qi in enumerate(q.T):
        marker_pos[:, i] = np.squeeze(marker_pos_func(qi))

    return marker_pos


def marker_ref(t: list, nlp: NonLinearProgram, index_marker):
    """
    Get the position of the c3d's markers

    Parameters
    ----------
    t: list
        list of time
    nlp: NonLinearProgram
        Non linear program
    idx: int
        Index of the marker
    Returns:
        position of the markers on the three axis
    """
    dt = nlp.dt
    if isinstance(t, float) and np.isnan(t):
        n_list = 0
    else:
        n_list = np.round((t / dt), 2).tolist()
        n_list = [int(i) for i in n_list]
    # keep in mind that if switch objective functions it will not work anymore.

    return nlp.J[1].target[:, index_marker, n_list]


def plot_marker(id_marker: int, ocp: OptimalControlProgram, nlp: list[NonLinearProgram]):
    """
    plot the markers posiions

    Parameters
    ----------
    id_marker: int
         The marker's id
    ocp: OptimalControlProgram
        Optimal control program
    nlp: list[NonLinearProgram]

    """
    ocp.add_plot(
        f"{'Marker'} {nlp[0].model.markerNames()[id_marker].to_string()}",
        lambda t, x, u, p: marker_ref(t, nlp[0], id_marker),
        legend=[f"Marker {id_marker} x", f"Marker {id_marker} y", f"Marker {id_marker} z"],
        plot_type=PlotType.STEP,
        node_idx=[nlp[0].dt * i for i in range(0, nlp[0].ns + 1)],
    )
    ocp.add_plot(
        f"{'Marker'} {nlp[0].model.markerNames()[id_marker].to_string()}",
        lambda t, x, u, p: marker_model(x, nlp[0], id_marker),
        legend=[f"Marker {id_marker} x", f"Marker {id_marker} y", f"Marker {id_marker} z"],
        plot_type=PlotType.PLOT,
        node_idx=None,
    )


def add_custom_plots(ocp: OptimalControlProgram, list_markers: Union[list[int], list[str]]):
    """
    Add extra plots to the OCP.

    Parameters
    ----------
    ocp: OptimalControlProgram
        Optimal control program
    list_markers: Union[list[int], list[str]]
        The list of marker's name or id
    """
    nlp = ocp.nlp
    model_markers = [m.to_string() for m in nlp[0].model.markerNames()]
    marker_idx = [model_markers.index(m) for m in list_markers]

    for idx in marker_idx:
        plot_marker(idx, ocp, nlp)

    return ocp
