import numpy as np


def marker_jacobian_model(x, biorbd_model, idx_model_markers, q_parameters_idx):
    """
    Generate the Jacobian matrix for each frame.

    Parameters:
    -----------
    x : np.ndarray
        vector with generalised coordinates and without parameters values
    biorbd_model : biorbd.Models
        the model used
    idx_model_markers : list(int)
        index of the model's marker in the markers list
    q_parameters_idx : list[int]
        list of parameters' index in the generalised coordinates q vector

    Return:
    ------
        The Jacobian matrix of the model without table's markers with right dimension (i.e without parameters)
    """

    # x_with_p = parameters_idx
    # x + size parameters
    # we fill x in x_with_p jumping off parameters idx
    x_with_p_shape = x.shape[0] + len(q_parameters_idx)
    x_with_p = np.zeros(x_with_p_shape)
    kinematic_idx = np.setdiff1d([i for i in range(x_with_p_shape)], q_parameters_idx)
    x_with_p[kinematic_idx] = x

    jacobian_matrix = biorbd_model.technicalMarkersJacobian(x_with_p)[idx_model_markers[0]: idx_model_markers[-1] + 1]
    nb_markers = len(jacobian_matrix)
    jacobian = np.zeros((3 * nb_markers, len(x_with_p)))

    for m, value in enumerate(jacobian_matrix):
        jacobian[m * 3: (m + 1) * 3, :] = value.to_array()

    l = [i for i in range(22) if i not in q_parameters_idx]
    jacobian_without_p = jacobian[:, l]

    return jacobian_without_p


def markers_jacobian_model_parameters(p, biorbd_model, idx_model_markers, q_kinematic_idx):
    x_with_p_shape = p.shape[0] + len(q_kinematic_idx)
    x_with_p = np.zeros(x_with_p_shape)
    parameter_idx = np.setdiff1d([i for i in range(x_with_p_shape)], q_kinematic_idx)
    x_with_p[parameter_idx] = p

    jacobian_matrix = biorbd_model.technicalMarkersJacobian(x_with_p)[idx_model_markers[0]: idx_model_markers[-1] + 1]
    nb_markers = len(jacobian_matrix)
    jacobian = np.zeros((3 * nb_markers, len(x_with_p)))

    for m, value in enumerate(jacobian_matrix):
        jacobian[m * 3: (m + 1) * 3, :] = value.to_array()

    l = [i for i in range(22) if i not in q_kinematic_idx]
    jacobian_without_x = jacobian[:, l]

    return jacobian_without_x


def marker_jacobian_table(x, biorbd_model, idx_markers_table, q_parameters_idx ):
    """
    Generate the Jacobian matrix for each frame.

    Parameters:
    -----------
    x : np.ndarray
        vector with generalised coordinates and without parameters values
    biorbd_model : biorbd.Models
        the model used
    idx_markers_table : list(int)
        index of the table's marker in the markers list
    q_parameters_idx : list[int]
        list of parameters' index in the generalised coordinates q vector

    Return:
    ------
        The Jacobian matrix of the table with right dimension for the model (i.e without parameters)
    """
    # x_with_p = parameters_idx
    # x + size parameters
    # we fill x in x_with_p jumping off parameters idx
    x_with_p_shape = x.shape[0] + len(q_parameters_idx)
    x_with_p = np.zeros(x_with_p_shape)
    kinematic_idx = np.setdiff1d([i for i in range(x_with_p_shape)], q_parameters_idx)
    x_with_p[kinematic_idx] = x

    jacobian_matrix = biorbd_model.technicalMarkersJacobian(x_with_p)[idx_markers_table[0]: idx_markers_table[1] + 1]
    nb_markers = len(jacobian_matrix)
    jacobian = np.zeros((3 * nb_markers, len(x_with_p)))

    for m, value in enumerate(jacobian_matrix):
        jacobian[m * 3: (m + 1) * 3, :] = value.to_array()

    l = [i for i in range(22) if i < 10 or i > 15]
    jacobian_without_p = jacobian[:, l]

    jacobian_without_p = jacobian_without_p[:5, :]  # we removed the value in z for the market Table:Table6

    return jacobian_without_p


def jacobian_table_parameters(p, biorbd_model, idx_markers_table, q_kinematic_idx ):
    x_with_p_shape = p.shape[0] + len(q_kinematic_idx)
    x_with_p = np.zeros(x_with_p_shape)
    parameter_idx = np.setdiff1d([i for i in range(x_with_p_shape)], q_kinematic_idx)
    x_with_p[parameter_idx] = p

    jacobian_matrix = biorbd_model.technicalMarkersJacobian(x_with_p)[idx_markers_table[0]: idx_markers_table[1] + 1]
    nb_markers = len(jacobian_matrix)
    jacobian = np.zeros((3 * nb_markers, len(x_with_p)))

    for m, value in enumerate(jacobian_matrix):
        jacobian[m * 3: (m + 1) * 3, :] = value.to_array()

    l = [i for i in range(22) if i >= 10 and i <16]
    jacobian_without_x = jacobian[:, l]

    jacobian_without_x = jacobian_without_x[:5, :]  # we removed the value in z for the market Table:Table6
    return jacobian_without_x



def marker_jacobian_theta(x, q_parameters_idx):
    """
    Generate the Jacobian matrix for each frame.

    Parameters:
    -----------
    x : np.ndarray
        vector with generalised coordinates and without parameters values
    q_parameters_idx : list[int]
        list of parameters' index in the generalised coordinates q vector

    Return:
    ------
        The Jacobian matrix associated with right dimension (i.e without parameters)
    """
    # x_with_p = parameters_idx
    # x + size parameters
    # we fill x in x_with_p jumping off parameters idx
    x_with_p_shape = x.shape[0] + len(q_parameters_idx)
    x_with_p = np.zeros(x_with_p_shape)
    kinematic_idx = np.setdiff1d([i for i in range(x_with_p_shape)], q_parameters_idx)
    x_with_p[kinematic_idx] = x

    theta_part1_3 = x_with_p[20] + x_with_p[21]

    theta_part1_3_lim = 7 * np.pi / 10

    if theta_part1_3 > theta_part1_3_lim:
        J = np.zeros((1, 22))
        J[0, 20] = 1
        J[0, 21] = 1
        l = [i for i in range(22) if i not in q_parameters_idx]
        j_without_p = J[:, l]
        return j_without_p
    else:  # theta_part1_3_min < theta_part1_3:
        return np.zeros((1, 16))


def marker_jacobian_theta_parameters():
    return np.zeros((1, 6))


def jacobian_q_continuity(x, q_parameters_idx):
    """
    Minimize the q of thorax

    Parameters
    ----------
     x : np.ndarray
        vector with generalised coordinates and without parameters values
    q_parameters_idx : list[int]
        list of parameters' index in the generalised coordinates q vector

    Return
    ------
    identity matrix with the shape of generalised coordinates associated to the model
    """
    # x_with_p = parameters_idx
    # x + size parameters
    # we fill x in x_with_p jumping off parameters idx
    x_with_p_shape = x.shape[0] + len(q_parameters_idx)
    x_with_p = np.zeros(x_with_p_shape)
    kinematic_idx = np.setdiff1d([i for i in range(x_with_p_shape)], q_parameters_idx)
    x_with_p[kinematic_idx] = x

    Jacob_q_continuity_analytic = np.eye(16)

    return Jacob_q_continuity_analytic


def  jacobian_q_continuity_parameters():
    Jacob_q_continuity_analytic = np.eye(6)

    return Jacob_q_continuity_analytic


def rotation_matrix_jacobian(x, biorbd_model, segment_idx, q_parameters_idx):
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
        the Jacobian of the rotation matrix with the right dimension (i.e without parameters)
        """

    # x_with_p = parameters_idx
    # x + size parameters
    # we fill x in x_with_p jumping off parameters idx
    x_with_p_shape = x.shape[0] + len(q_parameters_idx)
    x_with_p = np.zeros(x_with_p_shape)
    kinematic_idx = np.setdiff1d([i for i in range(x_with_p_shape)], q_parameters_idx)
    x_with_p[kinematic_idx] = x

    jacob_rot_matrix_analytic_total = biorbd_model.JacobianSegmentRotMat(x_with_p, segment_idx, True).to_array()
    # remove column for parameters values
    column_to_keep = np.unravel_index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21], (5, 22))[1]

    # keep lines we want, correpond to values we want to minimize to reach 0
    row_to_keep = np.ravel_multi_index(([2, 2, 0, 1, 2], [0, 1, 2, 2, 2]), (3, 3))
    row_to_keep.sort()

    A = jacob_rot_matrix_analytic_total[row_to_keep[:], :]
    A = A[:, column_to_keep[:]]
    return A

def rotation_matrix_parameter_jacobian(p, biorbd_model, segment_idx, q_kinematic_idx):
    x_with_p_shape = p.shape[0] + len(q_kinematic_idx)
    x_with_p = np.zeros(x_with_p_shape)
    parameter_idx = np.setdiff1d([i for i in range(x_with_p_shape)], q_kinematic_idx)
    x_with_p[parameter_idx] = p

    jacob_rot_matrix_analytic_total = biorbd_model.JacobianSegmentRotMat(x_with_p, segment_idx, True).to_array()

    # remove column for parameters values
    column_to_keep = np.unravel_index([10,11,12,13,14,15], (5, 22))[1]

    # keep lines we want, correpond to values we want to minimize to reach 0
    row_to_keep = np.ravel_multi_index(([2, 2, 0, 1, 2], [0, 1, 2, 2, 2]), (3, 3))
    row_to_keep.sort()

    A = jacob_rot_matrix_analytic_total[row_to_keep[:], :]
    A = A[:, column_to_keep[:]]
    return A
