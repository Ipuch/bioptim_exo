import numpy as np


def marker_jacobian_model(q, biorbd_model, wu_markers):
    """
    Generate the Jacobian matrix for each frame.

    Parameters:
    -----------
    jacobian_matrix: np.ndarray
        The Jacobian matrix of the model

    Return:
    ------
        The Jacobian matrix with right dimension
    """
    jacobian_matrix = biorbd_model.technicalMarkersJacobian(q)[wu_markers[0] : wu_markers[1] + 1]
    nb_markers = len(jacobian_matrix)
    jacobian = np.zeros((3 * nb_markers, len(q)))

    for m, value in enumerate(jacobian_matrix):
        jacobian[m * 3 : (m + 1) * 3, :] = value.to_array()

    l = [i for i in range(22) if i < 10 or i > 15] # removed hard code, add parameters index
    jacobian_without_p = jacobian[:, l]

    return jacobian_without_p


def marker_jacobian_table(q, biorbd_model, table_markers):
    """
    Generate the Jacobian matrix for each frame.

    Parameters:
    -----------
    jacobian_matrix: np.ndarray
        The Jacobian matrix of the model

    Return:
    ------
        The Jacobian matrix with right dimension
    """
    jacobian_matrix = biorbd_model.technicalMarkersJacobian(q)[table_markers[0] : table_markers[1] + 1]
    nb_markers = len(jacobian_matrix)
    jacobian = np.zeros((3 * nb_markers, len(q)))

    for m, value in enumerate(jacobian_matrix):
        jacobian[m * 3 : (m + 1) * 3, :] = value.to_array()

    l = [i for i in range(22) if i < 10 or i > 15]
    jacobian_without_p = jacobian[:, l]

    jacobian_without_p = jacobian_without_p[:5, :]  # we removed the value in z for the market Table:Table6

    return jacobian_without_p


def marker_jacobian_theta(q):
    """
    Generate the Jacobian matrix for each frame.

    Parameters:
    -----------
    jacobian_matrix: np.ndarray
        The Jacobian matrix of the model

    Return:
    ------
        The Jacobian matrix with right dimension
    """

    theta_part1_3 = q[20] + q[21]

    theta_part1_3_lim = 7 * np.pi / 10

    if theta_part1_3 > theta_part1_3_lim:
        J = np.zeros((1, 16))
        J[0, 20] = 1
        J[0, 21] = 1
        return J  # ajouter les poids
    else:  # theta_part1_3_min < theta_part1_3:
        return np.zeros((1, 16))


def jacobian_q_continuity(q):
    """
    Minimize the q of thorax

    Parameters
    ----------
    q: np.ndarray
        Generalized coordinates for all dof except those between ulna and piece 7, unique for all frames
    q_init: np.ndarray
        The initial values of generalized coordinates fo the actual frame

    Return
    ------
    The value of the penalty function
    """

    # return np.eye(q.shape[0])
    return np.eye(16)


def calibration_jacobian(q, biorbd_model, p, table_markers, wu_markers, markers_names, x0):

    index_table_markers = [i for i, value in enumerate(markers_names) if "Table" in value]
    table_markers = (index_table_markers[0], index_table_markers[-1])

    index_wu_markers = [i for i, value in enumerate(markers_names) if "Table" not in value]
    wu_markers = (index_wu_markers[0], index_wu_markers[-1])

    q_with_p = np.zeros(biorbd_model.nbQ())
    q_with_p[:10] = q[:10]
    q_with_p[10:16] = p
    q_with_p[16:] = q[10:]

    table = marker_jacobian_table(q_with_p, biorbd_model, table_markers)

    # Minimize difference between thorax markers from model and from experimental data
    model = marker_jacobian_model(q_with_p, biorbd_model, wu_markers)

    # Force z-axis of final segment to be vertical
    # rot_matrix_list_model, rot_matrix_list_xp = penalty_rotation_matrix(biorbd_model, x_with_p)

    # Minimize the q of thorax
    continuity = jacobian_q_continuity(q_with_p)

    # Force part 1 and 3 to not cross
    pivot = marker_jacobian_theta(q_with_p)
    jacobian = np.concatenate(
        (table * 100000, model * 10000, continuity * 500, pivot * 50000)
    )  # ligne par ligne axis=0

    return jacobian
