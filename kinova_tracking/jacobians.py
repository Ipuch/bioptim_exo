import numpy as np


def marker_jacobian_model(q, biorbd_model, wu_markers):
def marker_jacobian_model(x_with_p, biorbd_model, id_wu_markers):
    """
    Generate the Jacobian matrix for each frame.

    Parameters:
    -----------
    x_with_p : np.ndarray
        vector with generalised coordinates and parameters values
    biorbd_model : biorbd.Models
        the model used
    id_wu_markers : tuple(int)
        index of the first and the last model's marker

    Return:
    ------
        The Jacobian matrix of the model without table's markers with right dimension
    """
    jacobian_matrix = biorbd_model.technicalMarkersJacobian(q)[wu_markers[0] : wu_markers[1] + 1]
    nb_markers = len(jacobian_matrix)
    jacobian = np.zeros((3 * nb_markers, len(q)))

    for m, value in enumerate(jacobian_matrix):
        jacobian[m * 3 : (m + 1) * 3, :] = value.to_array()

    l = [i for i in range(22) if i < 10 or i > 15] # removed hard code, add parameters index
    jacobian_without_p = jacobian[:, l]

    return jacobian_without_p


def marker_jacobian_table(x_with_p, biorbd_model, id_table_markers):
    """
    Generate the Jacobian matrix for each frame.

    Parameters:
    -----------
    x_with_p : np.ndarray
        vector with generalised coordinates and parameters values

    biorbd_model : biorbd.Models
        the model used

    id_table_markers : tuple(int)
            index of the markers associated with the table


    Return:
    ------
        The Jacobian matrix of the table with right dimension for the model
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


def marker_jacobian_theta(x_with_p):
    """
    Generate the Jacobian matrix for each frame.

    Parameters:
    -----------
    x_with_p : np.ndarray
        vector with generalised coordinates and parameters values

    Return:
    ------
        The Jacobian matrix associated with right dimension
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


def jacobian_q_continuity(x_with_p):
    """
    Minimize the q of thorax

    Parameters
    ----------
    x_with_p : np.ndarray
        vector with generalised coordinates and parameters values
    q_init: np.ndarray
        The initial values of generalized coordinates fo the actual frame

    Return
    ------
    identity matrix with the shape of generalised coordinates
    """

    # return np.eye(q.shape[0])
    return np.eye(16)

def rotation_matrix_jacobian(x_with_p, biorbd_model, id_segment):
    """
        This function return the analytical Jacobian matrix of rotation

        Parameters
        ----------
        x_with_p : np.ndarray
            vector with generalised coordinates and parameters values
        biorbd_model: biorbd.Models
            the model used
        id_segment
            the segment where the Jacobian matrix of rotation will be calculated

        Return
        ------
        the Jacobian of the rotation matrix
def calibration_jacobian(x, biorbd_model, p, tracked_markers_idx, closed_loop_markers_idx, weights ):

    """
         This function return the entire Jacobian of the system

         Parameters
         ----------
         x: np.ndarray
             Generalized coordinates WITHOUT parameters values
         biorbd_model: biorbd.Models
             the model used
         p : np.ndarray
            parameters values
        tracked_markers_idx : list[int]
            index of tracked marker, without those associated to the table
        closed_loop_markers_idx : list[int]
            index of markers associated to the table
        weights : list[int]
            list of the weight associated for each Jacobians


         Return
         ------
         the Jacobian of the entire system
         """

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
    # rot_matrix_list_model, rot_matrix_list_xp = c(biorbd_model, x_with_p)

    # Minimize the q of thorax
    continuity = jacobian_q_continuity(q_with_p)

    # Force part 1 and 3 to not cross
    pivot = marker_jacobian_theta(q_with_p)
    jacobian = np.concatenate(
        (table * 100000, model * 10000, continuity * 500, pivot * 50000)
    )  # ligne par ligne axis=0

    return jacobian
