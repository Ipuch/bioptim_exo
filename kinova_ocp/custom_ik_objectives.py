from biorbd import marker_index
from casadi import sumsqr, vertcat, MX, norm_1, horzcat, transpose, cross, sqrt
import numpy as np

from bioptim.limits.penalty import PenaltyFunctionAbstract
from bioptim.limits.penalty_option import PenaltyOption
from bioptim.limits.penalty_controller import PenaltyController


def superimpose_matrix(
        controller: PenaltyController,
        track_segment: int,
        first_model_marker: list[str],
        second_model_marker: str | int,
        third_model_marker: str | int,
):
    """
    Calculate the penalty cost for rotation matrix

    Parameters
    ----------
    x : MX
        the entire vector with q and p

    Returns
    -------
    The cost of the penalty function

    """
    rotation_matrix = controller.model.extra_models[0].homogeneous_matrices_in_global(
        controller.controls["q_k"].mx,
        controller.model.extra_models[0].nb_segments - 1,
    ).rot().to_mx()

    markers = controller.model.markers(controller.states["q"].mx)

    p1_index = marker_index(controller.model.models[0].model, first_model_marker)
    p2_index = marker_index(controller.model.models[0].model, second_model_marker)
    p3_index = marker_index(controller.model.models[0].model, third_model_marker)

    p1 = markers[p1_index]
    p2 = markers[p2_index]
    p3 = markers[p3_index]

    rotation_matrix_ref = create_frame_from_three_points(p1, p2, p3)[:3, :3]

    R = transpose(rotation_matrix) @ rotation_matrix_ref

    rot_matrix_list_model = [
        (R[0,0] - 1),
        R[0,1],
        R[0,2],
        R[1,0],
        (R[1,1] - 1),
        R[1,2],
        R[2,0],
        R[2,1],
        (R[2,2] - 1),
    ]

    return controller.mx_to_cx(
        f"superimpose_matrix",
        sumsqr(vertcat(*rot_matrix_list_model)),
        controller.states["q"], controller.controls["q_k"],
    )


def superimpose_markers(
        controller: PenaltyController,
        first_model_marker: list[str],
        second_model_marker: str | int,
        axes: tuple | list = None,
):
    """
    Minimize the distance between two markers
    By default this function is quadratic, meaning that it minimizes distance between them.

    Parameters
    ----------
    controller: PenaltyController
        The penalty node elements
    first_model_marker: str | int
        The name or index of one of the two markers
    second_model_marker: str | int
        The name or index of one of the two markers
    axes: tuple | list
        The axes to project on. Default is all axes
    """

    first_marker_idx = tuple([controller.model.marker_index(marker_str) for marker_str in first_model_marker])

    second_marker_idx = (
        controller.model.extra_models[0].marker_index(second_model_marker)
    )
    PenaltyFunctionAbstract._check_idx(
        "marker", [first_marker_idx[0], second_marker_idx], controller.model.nb_markers
    )
    mean_first_marker = 1/2 * (
            controller.model.marker(
            controller.states["q"].mx, first_marker_idx[0]) + controller.model.marker(
            controller.states["q"].mx, first_marker_idx[1])
        )

    diff_markers = mean_first_marker - controller.model.extra_models[0].marker(controller.controls["q_k"].mx, second_marker_idx)

    return controller.mx_to_cx(
        f"diff_markers",
        diff_markers,
        controller.states["q"], controller.controls["q_k"],
    )


def create_frame_from_three_points(p1, p2, p3) -> MX:
    """
    Create a homogenous transform from three points. middle of p1 and p2 is the origin of the frame
    p1->p2 is the y axis of the frame
    z from cross product of p1->p2 and midp1p2->p3
    x from cross product of y and z
    """
    mid_p1_p2 = (p1 + p2) / 2

    x = p3 - mid_p1_p2
    x /= sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)

    y_temp = (p2 - p1)
    y_temp /= sqrt(y_temp[0] ** 2 + y_temp[1] ** 2 + y_temp[2] ** 2)

    z = cross(x, y_temp)
    z /= sqrt(z[0] ** 2 + z[1] ** 2 + z[2] ** 2)

    y = cross(z, x)
    y /= sqrt(y[0] ** 2 + y[1] ** 2 + y[2] ** 2)

    x0 = vertcat(x, 0)
    y0 = vertcat(y, 0)
    z0 = vertcat(z, 0)
    t1 = vertcat(mid_p1_p2, 1)

    return horzcat(x0, y0, z0, t1)


# NOTE : no need as polar coordinates i think so.
# def distance_constraints(
#         controller: PenaltyController,
#         max = 0.120,
# ):
#     """
#     max distance of the three link kinematic chain.
#
#     Parameters
#     ----------
#     controller: PenaltyController
#         The penalty node elements
#     """
#
#     origin_kinova = np.array([0.0, 0.0])
#     radius_three_bar = np.linalg.norm([-0.08842, -0.02369]) + 2 * 0.120 - np.linalg.norm(origin_kinova)
#     # it should stay in the circle defined by the radius of the three bar mechanism
#     q_sym = controller.controls["q_k"].mx
#     constraints += [q_sym[0] ** 2 + q_sym[1] ** 2 - radius_three_bar ** 2]
#     lbg += [-np.inf]
#     ubg += [0]
#
#     return controller.mx_to_cx(
#         f"diff_markers",
#         diff_markers,
#         controller.states["q"], controller.controls["q_k"],
#     )