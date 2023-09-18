from casadi import sumsqr, vertcat, MX, norm_1, horzcat, transpose, cross
from bioptim.limits.penalty import PenaltyFunctionAbstract
from bioptim.limits.penalty_option import PenaltyOption
from bioptim.limits.penalty_controller import PenaltyController


def penalty_rotation_matrix_cas(model, x: MX, rotation_matrix_ref) -> MX:
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
    rotation_matrix = model.globalJCS(x, model.nbSegment() - 1).rot().to_mx()
    # we want to compare the rotation matrix of the part 7 with the rotation matrix of the reference

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

    return sumsqr(vertcat(*rot_matrix_list_model))


def superimpose_markers(
        controller: PenaltyController,
        first_model_marker: str | int,
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

    first_marker_idx = (
        controller.model.marker_index(first_model_marker)
    )
    second_marker_idx = (
        controller.model.extra_models[0].marker_index(second_model_marker)
    )
    PenaltyFunctionAbstract._check_idx(
        "marker", [first_marker_idx, second_marker_idx], controller.model.nb_markers
    )

    diff_markers = controller.model.marker(
        controller.states["q"].mx, first_marker_idx
    ) - controller.model.extra_models[0].marker(controller.controls["q_k"].mx, second_marker_idx)

    return controller.mx_to_cx(
        f"diff_markers",
        diff_markers,
        controller.states["q"], controller.controls["q_k"],
    )


def penalty_not_too_far_from_previous_pose_case(model, x: MX, previous_pose) -> MX:
    """
    Calculate the penalty cost for position with respect to the previous pose
    """

    return sumsqr(x - previous_pose)


def create_frame_from_three_points(p1, p2, p3) -> MX:
    """
    Create a homogenous transform from three points. middle of p1 and p2 is the origin of the frame
    p1->p2 is the y axis of the frame
    z from cross product of p1->p2 and midp1p2->p3
    x from cross product of y and z
    """
    mid_p1_p2 = (p1 + p2) / 2

    x = (p3 - mid_p1_p2) / norm_1(p3 - mid_p1_p2)
    y_temp = (p2 - p1) / norm_1(p2 - p1)
    z = cross(x, y_temp) / norm_1(cross(x, y_temp))
    y = cross(z, x) / norm_1(cross(z, x))

    x0 = vertcat((x, [0]))
    y0 = vertcat((y, [0]))
    z0 = vertcat((z, [0]))
    t1 = vertcat((mid_p1_p2, [1]))

    return horzcat((x0.reshape(4, 1), y0.reshape(4, 1), z0.reshape(4, 1), t1.reshape(4, 1)))

