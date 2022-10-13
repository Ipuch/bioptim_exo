import biorbd
import pytest

from models import enums
import numpy as np
from scipy.optimize import approx_fprime
from data.enums import TasksKinova
from jacobians import (
    rotation_matrix_jacobian,
    marker_jacobian_model,
    marker_jacobian_table,
    jacobian_q_continuity,
    marker_jacobian_theta,
)
from main import prepare_kcc as prep_kcc
import random as rd

model_name = '/home/nicolas/Documents/Stage_Robin/bioptim_exo/models/KINOVA_merge_without_floating_base_with_6_dof_support_template_with_variables.bioMod'
# model_name = enums.Models.WU_AND_KINOVA_WITHOUT_FLOATING_BASE_WITH_6_DOF_SUPPORT_VARIABLES
biorbd_model = biorbd.Model(model_name)


# get the function use in the Jacobian thank's to ik_step method. Return Table+Thorax+q_continuity+theta+rotation
# in a 1D array which size is(69,1):  [0-4/4-46/46-63/63-64/64-69]


def test_global_rotation_matrix():
    def rotation_matrix(q, id_segment):
        return biorbd_model.globalJCS(q, id_segment).rot().to_array().flatten(
            'F')  # reshape column by column into a vector in this way [0,0], [1,0], [2,0], [0,1],...,

    q = np.ones(biorbd_model.nbQ())
    id_segment = 32
    handle = lambda q: rotation_matrix(q, id_segment)
    Jacob_matrix_numeric = approx_fprime(q, handle, epsilon=1e-10)
    Jacob_matrix_analytic = biorbd_model.JacobianSegmentRotMat(q, id_segment, True).to_array()  # take a vector build
    print(np.shape(Jacob_matrix_analytic))
    np.testing.assert_allclose(Jacob_matrix_analytic, Jacob_matrix_numeric, rtol=1e-3, atol=1e-3)


def test_rotation_matrix():
    task = TasksKinova.DRINK
    nb_frame_param_step = 100
    use_analytical_jacobians = False
    x = np.ones(16)

    kcc_object = prep_kcc(task, nb_frame_param_step, use_analytical_jacobians)[2]
    q_idx_param = kcc_object.q_parameter_index

    table_markers = kcc_object.markers[:, kcc_object.table_markers_idx, 0]
    thorax_markers = kcc_object.markers[:, kcc_object.model_markers_idx, 0]
    q_init = kcc_object.q_ik_initial_guess
    weights = kcc_object.weights

    L = np.asarray([64, 65, 66, 67, 68])
    objective_function = lambda x: \
    kcc_object.ik_step(x, np.zeros(len(q_idx_param)), table_markers, thorax_markers, q_init[:, 0])[L]

    Jacob_rotation_numeric = approx_fprime(x, objective_function, epsilon=1e-10)
    jacobian_analytic = lambda x: rotation_matrix_jacobian(x, biorbd_model, 45, q_idx_param)
    Jacob_rotation_analytic = jacobian_analytic(x)

    def add_weight(Jacob_rotation_analytic):
        return Jacob_rotation_analytic[:, :] * weights[4]

    Jacob_rotation_analytic1 = add_weight(Jacob_rotation_analytic)

    # acob_rotation_analytic1[4,:] *= (-1)

    np.testing.assert_allclose(Jacob_rotation_analytic1, Jacob_rotation_numeric, rtol=1e-4, atol=1e-4)


def test_marker_jacobian_model():
    task = TasksKinova.DRINK
    nb_frame_param_step = 100
    use_analytical_jacobians = False
    x = np.ones(16)

    kcc_object = prep_kcc(task, nb_frame_param_step, use_analytical_jacobians)[2]
    q_idx_param = kcc_object.q_parameter_index
    idx_markers_model = kcc_object.model_markers_idx

    table_markers = kcc_object.markers[:, kcc_object.table_markers_idx, 0]
    thorax_markers = kcc_object.markers[:, kcc_object.model_markers_idx, 0]
    q_init = kcc_object.q_ik_initial_guess
    weights = kcc_object.weights

    objective_function = lambda x: kcc_object.ik_step(x, np.zeros(len(q_idx_param)), table_markers, thorax_markers,
                                                      q_init[:, 0])[5:47]

    Jacob_model_numeric = approx_fprime(x, objective_function, epsilon=1e-10)
    jacobian_analytic = lambda x: marker_jacobian_model(x, biorbd_model, idx_markers_model, q_idx_param)
    Jacob_model_analytic = jacobian_analytic(x)

    def add_weight(Jacob_model_analytic):
        return Jacob_model_analytic[:, :] * weights[1]

    Jacob_model_analytic1 = add_weight(Jacob_model_analytic)

    # print("analytic", np.shape(Jacob_model_analytic))
    # print("numeric",np.shape(Jacob_model_numeric))
    # first verify if we get zeros at the same place
    np.where(Jacob_model_analytic1 == 0)
    np.testing.assert_equal(np.where(Jacob_model_analytic1 == 0), np.where(Jacob_model_numeric == 0))
    # first verify non zeros values afterwards
    non_zero_idx = np.where(Jacob_model_analytic1 != 0)
    np.testing.assert_allclose(
        Jacob_model_analytic1[non_zero_idx[0], non_zero_idx[1]],
        Jacob_model_numeric[non_zero_idx[0], non_zero_idx[1]], atol=1e-3, rtol=1e-3)
    # tolerance is high because the weight is 1e5
    # if the weight wouldn't have been taken into account the tol would have been 1e-6


def test_marker_jacobian_table():
    task = TasksKinova.DRINK
    nb_frame_param_step = 100
    use_analytical_jacobians = False
    x = np.ones(16)

    kcc_object = prep_kcc(task, nb_frame_param_step, use_analytical_jacobians)[2]
    q_idx_param = kcc_object.q_parameter_index
    idx_markers_table = kcc_object.table_markers_idx

    table_markers = kcc_object.markers[:, kcc_object.table_markers_idx, 0]
    thorax_markers = kcc_object.markers[:, kcc_object.model_markers_idx, 0]
    q_init = kcc_object.q_ik_initial_guess
    weights = kcc_object.weights

    objective_function = lambda x: kcc_object.ik_step(x, np.zeros(len(q_idx_param)), table_markers, thorax_markers,
                                                      q_init[:, 0])[0:5]

    jacobian_analytic = lambda x: marker_jacobian_table(x, biorbd_model, idx_markers_table, q_idx_param)
    Jacob_table_numeric = approx_fprime(x, objective_function, epsilon=1e-10)
    Jacob_table_analytic = jacobian_analytic(x)

    def add_weight(Jacob_table_analytic):
        return Jacob_table_analytic[:, :] * weights[0]

    Jacob_table_analytic1 = add_weight(Jacob_table_analytic)

    np.where(Jacob_table_analytic1 == 0)
    np.testing.assert_equal(np.where(Jacob_table_analytic1 == 0), np.where(Jacob_table_numeric == 0))

    np.testing.assert_allclose(Jacob_table_analytic1, Jacob_table_numeric, rtol=1e-03, atol=1e-3)


def test_q_continuity():
    task = TasksKinova.DRINK
    nb_frame_param_step = 100
    use_analytical_jacobians = False
    x = np.ones(16)

    kcc_object = prep_kcc(task, nb_frame_param_step, use_analytical_jacobians)[2]
    q_idx_param = kcc_object.q_parameter_index

    table_markers = kcc_object.markers[:, kcc_object.table_markers_idx, 0]
    thorax_markers = kcc_object.markers[:, kcc_object.model_markers_idx, 0]
    q_init = kcc_object.q_ik_initial_guess
    weights = kcc_object.weights

    jacobian_analytic = lambda x: jacobian_q_continuity(x, q_idx_param)
    objective_function = lambda x: kcc_object.ik_step(x, np.zeros(len(q_idx_param)), table_markers, thorax_markers,
                                                      q_init[:, 0])[47:63]

    Jacob_q_continuity_numeric = approx_fprime(x, objective_function, epsilon=1e-10)
    Jacob_q_continuity_analytic = jacobian_analytic(x)

    def add_weight(Jacob_q_continuity_analytic):
        return Jacob_q_continuity_analytic[:, :] * (weights[3])

    Jacob_q_continuity_analytic1 = add_weight(Jacob_q_continuity_analytic)

    # print("Jacob_q_continuity_analytic=", Jacob_q_continuity_analytic1)
    # print("Jacob_q_continuity_numeric=", Jacob_q_continuity_numeric)

    np.where(Jacob_q_continuity_analytic1 == 0)
    np.testing.assert_equal(np.where(Jacob_q_continuity_analytic1 == 0), np.where(Jacob_q_continuity_numeric == 0))

    np.testing.assert_allclose(Jacob_q_continuity_analytic1, Jacob_q_continuity_numeric, rtol=1e-04, atol=1e-4)


def test_pivot_theta():
    task = TasksKinova.DRINK
    nb_frame_param_step = 100
    use_analytical_jacobians = False
    x = np.ones(16)

    kcc_object = prep_kcc(task, nb_frame_param_step, use_analytical_jacobians)[2]
    q_idx_param = kcc_object.q_parameter_index

    table_markers = kcc_object.markers[:, kcc_object.table_markers_idx, 0]
    thorax_markers = kcc_object.markers[:, kcc_object.model_markers_idx, 0]
    q_init = kcc_object.q_ik_initial_guess
    weights = kcc_object.weights

    jacobian_analytic = lambda x: marker_jacobian_theta(x, q_idx_param)
    objective_function = lambda x: kcc_object.ik_step(x, np.zeros(len(q_idx_param)), table_markers, thorax_markers,
                                                      q_init[:, 0])[63:64]

    jacob_theta_numeric = approx_fprime(x, objective_function, epsilon=1e-10)
    # resize as a 1D vector
    Jacob_theta_numeric = np.zeros((1, 16))
    for i in range(16):
        Jacob_theta_numeric[0, i] = jacob_theta_numeric[i]

    Jacob_theta_analytic = jacobian_analytic(x)

    def add_weight(Jacob_theta_analytic):
        return Jacob_theta_analytic[:, :] * (weights[3])

    Jacob_theta_analytic1 = add_weight(Jacob_theta_analytic)

    np.where(Jacob_theta_analytic == 0)
    np.testing.assert_equal(np.where(Jacob_theta_analytic1 == 0), np.where(Jacob_theta_numeric == 0))

    np.testing.assert_allclose(Jacob_theta_numeric, Jacob_theta_analytic1, rtol=1e-04, atol=1e-4)


@pytest.mark.parametrize("x",
                         [np.ones(16),
                          np.array([rd.randrange(0, 50) for i in range(16)]),
                          np.arange(0, 16),
                          np.arange(0, 16) * 50,
                          ]
                         )
def test_entire_jacobian(x):
    task = TasksKinova.DRINK
    nb_frame_param_step = 100
    use_analytical_jacobians = False

    kcc_object = prep_kcc(task, nb_frame_param_step, use_analytical_jacobians)[2]
    q_idx_param = kcc_object.q_parameter_index
    idx_markers_model = kcc_object.model_markers_idx
    idx_markers_table = kcc_object.table_markers_idx
    weights = kcc_object.weights

    # get all the analytical Jacobian
    jacobian_analytic = lambda x: rotation_matrix_jacobian(x, biorbd_model, 45, q_idx_param)
    Jacob_rotation_analytic = jacobian_analytic(x)

    jacobian_analytic = lambda x: marker_jacobian_model(x, biorbd_model, idx_markers_model, q_idx_param)
    Jacob_model_analytic = jacobian_analytic(x)

    jacobian_analytic = lambda x: marker_jacobian_table(x, biorbd_model, idx_markers_table, q_idx_param)
    Jacob_table_analytic = jacobian_analytic(x)

    jacobian_analytic = lambda x: jacobian_q_continuity(x, q_idx_param)
    Jacob_q_continuity_analytic = jacobian_analytic(x)

    jacobian_analytic = lambda x: marker_jacobian_theta(x, q_idx_param)
    Jacob_theta_analytic = jacobian_analytic(x)

    # concatenate all Jacobians
    jacobian_total_analytic = np.concatenate(
        # (Jacob_table_analytic * weights[0],
        #  Jacob_model_analytic * weights[1],
        #  Jacob_q_continuity_analytic * weights[3],
        #  Jacob_theta_analytic * weights[2],
        #  Jacob_rotation_analytic * weights[4],
        #  ),
        (Jacob_table_analytic,
         Jacob_model_analytic,
         Jacob_q_continuity_analytic,
         Jacob_theta_analytic,
         Jacob_rotation_analytic,
         ),
        axis=0
    )

    # get all the numerical Jacobian

    table_markers = kcc_object.markers[:, kcc_object.table_markers_idx, 0]
    thorax_markers = kcc_object.markers[:, kcc_object.model_markers_idx, 0]
    q_init = kcc_object.q_ik_initial_guess

    objective_function1 = lambda x: kcc_object.ik_step(x, np.zeros(len(q_idx_param)), table_markers, thorax_markers,
                                                       q_init[:, 0])[64:69]
    Jacob_rotation_numeric = approx_fprime(x, objective_function1, epsilon=1e-10)

    objective_function2 = lambda x: kcc_object.ik_step(x, np.zeros(len(q_idx_param)), table_markers, thorax_markers,
                                                       q_init[:, 0])[5:47]
    Jacob_model_numeric = approx_fprime(x, objective_function2, epsilon=1e-10)

    objective_function3 = lambda x: kcc_object.ik_step(x, np.zeros(len(q_idx_param)), table_markers, thorax_markers,
                                                       q_init[:, 0])[0:5]
    Jacob_table_numeric = approx_fprime(x, objective_function3, epsilon=1e-10)

    objective_function4 = lambda x: kcc_object.ik_step(x, np.zeros(len(q_idx_param)), table_markers, thorax_markers,
                                                       q_init[:, 0])[47:63]
    Jacob_q_continuity_numeric = approx_fprime(x, objective_function4, epsilon=1e-10)

    objective_function5 = lambda x: kcc_object.ik_step(x, np.zeros(len(q_idx_param)), table_markers, thorax_markers,
                                                       q_init[:, 0])[63:64]
    jacob_theta_numeric = approx_fprime(x, objective_function5, epsilon=1e-10)
    # resize as a 1D vector
    Jacob_theta_numeric = np.zeros((1, 16))
    for i in range(16):
        Jacob_theta_numeric[0, i] = jacob_theta_numeric[i]

    # concatenate all Jacobians
    jacobian_total_numeric = np.concatenate(
        (Jacob_table_numeric / weights[0],
         Jacob_model_numeric / weights[1],
         Jacob_q_continuity_numeric / weights[3],
         Jacob_theta_numeric / weights[2],
         Jacob_rotation_numeric / weights[4],
         ),
        axis=0
    )

    jacob_total_diff = jacobian_total_analytic - jacobian_total_numeric
    index_max_diff = np.where(jacob_total_diff == np.max(jacob_total_diff))

    print("the maximum difference is equal to ", np.max(jacob_total_diff))
    print("index where the difference is maximun", index_max_diff)

    np.testing.assert_allclose(jacobian_total_analytic, jacobian_total_numeric, rtol=1e-03, atol=1e-3)
