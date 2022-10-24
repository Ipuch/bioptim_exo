"""
Main script to calibrate the arm support
"""
from typing import Tuple
import numpy as np

from ezc3d import c3d
import biorbd
import bioviz

from models.utils import add_header
from utils import get_unit_division_factor
from models.enums import Models
from data.enums import TasksKinova

from kinematic_chain_calibration import KinematicChainCalibration

def move_marker(
    marker_to_move: int,
    c3d_point: np.ndarray,
    offset: np.ndarray,
) -> np.array:
    """
    This function applies an offet to a marker

    Parameters
    ----------
    marker_to_move: int
        indices of the marker to move
    c3d_point: np.ndarray
        Markers trajectories.
    offset: np.ndarray
        The vector of offset to apply to the markers in mm.

    Returns
    -------
    new_points : np.array
        The markers with the displaced ones at a given distance on the horizontal plane.
    """

    new_points = c3d_point.copy()
    new_points[0, marker_to_move, :] = c3d_point[0, marker_to_move, :] + offset[0]
    new_points[1, marker_to_move, :] = c3d_point[1, marker_to_move, :] + offset[1]
    new_points[2, marker_to_move, :] = c3d_point[2, marker_to_move, :] + offset[2]

    return new_points


def inverse_kinematics_inferface(c3d: c3d, model_path: str, points: np.array, labels_markers_ik: list[str]) -> np.array:
    # todo: reformat inverse_kinematics_inferface
    """
    This function computes the inverse kinematics of the model.

    Parameters
    ----------
    c3d : c3d
        The c3d
    model_path : str
        Path to the model.
    points : np.ndarray
        marker trajectories over time
    labels_markers_ik : list[str]
        List of markers labels

    Returns
    -------
    q : np.array
        The generalized coordinates of the model for each frame
    """
    biorbd_model_ik = biorbd.Model(model_path)

    # Markers labels in the model
    marker_names_ik = [biorbd_model_ik.markerNames()[i].to_string() for i in range(biorbd_model_ik.nbMarkers())]

    # reformat the makers trajectories
    markers_ik = np.zeros((3, len(marker_names_ik), len(points[0, 0, :])))
    for i, name in enumerate(marker_names_ik):
        markers_ik[:, i, :] = points[:3, labels_markers_ik.index(name), :] / get_unit_division_factor(c3d)

        # the actual inverse kinematics
    my_ik = biorbd.InverseKinematics(biorbd_model_ik, markers_ik)
    my_ik.solve("trf")

    return my_ik


def two_step_inverse_kinematics(
    c3d_file,
    points_c3d,
    labels_markers,
    model_path_6_dofs,
    model_path_fixed_template,
    model_path_fixed_with_variables,
):
    """
    specific to our topic

    Firstly calculate generalised coordinates with the upper limb model and 6 dof
    Secondly fixe the thorax generalised coordinates, calculate the generalised coordinates
    of the entire upper limb WITHOUT the Kinova arm.

    Parameters
    ----------
    c3d_file : c3d
        the c3d
    points_c3d : np.ndarray
        marker trajectories over time
    labels_markers : list[str]
        list of markers labels
    model_path_6_dofs : str
        Path to the model with 6 dof
    model_path_fixed_template : str
        Path to the  fixed model with template
    model_path_fixed_with_variables : str
        Path to the fixed model with variables

    Returns
    -------
    ik_without_floating_base.q : np.ndarray
        The generalised coordinates from the second inverse kinematics without floating base
    thorax_values : dict
        value of each thorax rotation used to fixe the base
    """

    # Step 1.1: IK of wu model with floating base
    ik_with_floating_base = inverse_kinematics_inferface(
        c3d=c3d_file, model_path=model_path_6_dofs, points=points_c3d, labels_markers_ik=labels_markers
    )
    # ik_with_floating_base.animate()

    thorax_values = {
        "thoraxRT1": ik_with_floating_base.q[3, :].mean(),
        "thoraxRT2": ik_with_floating_base.q[4, :].mean(),
        "thoraxRT3": ik_with_floating_base.q[5, :].mean(),
        "thoraxRT4": ik_with_floating_base.q[0, :].mean(),
        "thoraxRT5": ik_with_floating_base.q[1, :].mean(),
        "thoraxRT6": ik_with_floating_base.q[2, :].mean(),
    }

    add_header(
        biomod_file_name=model_path_fixed_template,
        new_biomod_file_name=model_path_fixed_with_variables,
        variables=thorax_values,
    )

    # Step 1.2: IK of wu model without floating base
    ik_without_floating_base = inverse_kinematics_inferface(
        c3d=c3d_file, model_path=model_path_fixed_with_variables, points=points_c3d, labels_markers_ik=labels_markers
    )

    return thorax_values, ik_without_floating_base.q


def export_to_biomod(
    pos_init: np.ndarray,
    biorbd_model_merge: biorbd.Model,
    task : TasksKinova
):
    """
    This function exports the calibrated model to
    Models.WU_AND_KINOVA_WITHOUT_FLOATING_BASE_WITH_ROTOTRANS_SUPPORT_VARIABLES
    with the template Models.WU_AND_KINOVA_WITHOUT_FLOATING_BASE_WITH_ROTOTRANS_SUPPORT_VARIABLES

    Parameters
    ----------
    pos_init: np.ndarray
        calibrated paramter dofs
    q_ik_with_floating_base
        generalized coordinates of the inverse kinematics with the model including 6 dofs on the floating base
    biorbd_model_merge : biorbd.Model
        The model used during the kinematic chain calibration
        Models.WU_AND_KINOVA_WITHOUT_FLOATING_BASE_WITH_6_DOF_SUPPORT_VARIABLES

    task : TasksKinova
        task used
    """
    Rototrans_matrix_world_support = biorbd_model_merge.globalJCS(
        pos_init[:, 0], biorbd_model_merge.getBodyBiorbdId("part7")
    ).to_array()

    Rototrans_matrix_ulna_world = (
        biorbd_model_merge.globalJCS(pos_init[:, 0], biorbd_model_merge.getBodyBiorbdId("ulna")).transpose().to_array()
    )

    # Finally
    Rototrans_matrix_ulna_support = np.matmul(Rototrans_matrix_ulna_world, Rototrans_matrix_world_support)

    rototrans_values = {
        "rotationXX": Rototrans_matrix_ulna_support[0, 0],
        "rotationXY": Rototrans_matrix_ulna_support[0, 1],
        "rotationXZ": Rototrans_matrix_ulna_support[0, 2],
        "translationX": Rototrans_matrix_ulna_support[0, 3],
        "rotationYX": Rototrans_matrix_ulna_support[1, 0],
        "rotationYY": Rototrans_matrix_ulna_support[1, 1],
        "rotationYZ": Rototrans_matrix_ulna_support[1, 2],
        "translationY": Rototrans_matrix_ulna_support[1, 3],
        "rotationZX": Rototrans_matrix_ulna_support[2, 0],
        "rotationZY": Rototrans_matrix_ulna_support[2, 1],
        "rotationZZ": Rototrans_matrix_ulna_support[2, 2],
        "translationZ": Rototrans_matrix_ulna_support[2, 3],
    }

    c3d_kinova, labels_markers, points_c3d=load_c3d_file(task.value)
    thorax_values = two_step_inverse_kinematics(c3d_kinova, points_c3d, labels_markers, Models.WU_INVERSE_KINEMATICS.value, Models.WU_WITHOUT_FLOATING_BASE_TEMPLATE.value, Models.WU_WITHOUT_FLOATING_BASE_VARIABLES.value)[0]

    #merge the 2 dictionnaries
    thorax_and_rototrans_values = thorax_values | rototrans_values

    new_biomod_file_new = Models.WU_AND_KINOVA_WITHOUT_FLOATING_BASE_WITH_ROTOTRANS_SUPPORT_VARIABLES.value
    template_file = Models.WU_AND_KINOVA_WITHOUT_FLOATING_BASE_WITH_ROTOTRANS_SUPPORT_TEMPLATE.value

    add_header(biomod_file_name=template_file, new_biomod_file_name=new_biomod_file_new, variables=thorax_and_rototrans_values)

    #a faire ?
    return thorax_and_rototrans_values


def load_c3d_file(task: TasksKinova) -> Tuple:
    """
    load all the data about markers required to run the script

    Parameters
    ----------
    task : TasksKinova
        task to realise

    Returns
    -------
    c3d_kinova : c3d
        c3d of the Kinova arm
    labels_markers : Any
        markers labels in c3d
    points_c3d : Any
        Markers trajectories
    """

    c3d_path = task.value
    c3d_kinova = c3d(c3d_path)

    # Markers labels in c3d
    labels_markers = c3d_kinova["parameters"]["POINT"]["LABELS"]["value"]
    points_c3d = c3d_kinova["data"]["points"][:, :, :]

    marker_move = False
    offset = np.array([0, -50, 0])  # [offsetX,offsetY,offsetZ] mm
    print("offset", offset)
    # Markers trajectories
    points_c3d = (
        points_c3d
        if not marker_move
        else move_marker(
            marker_to_move=labels_markers.index("Table:Table5"), c3d_point=points_c3d, offset=offset
        )
    )

    return c3d_kinova, labels_markers, points_c3d


def prepare_kcc(
    task: TasksKinova,
    nb_frame_param_step: int ,
    use_analytical_jacobians : bool,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    this function is the main script executed in function of the Task and parameters

    Parameters
    ----------
    task: TasksKinova
        the task
    nb_frame_param_step:
        Number of franes used for the parameter optimisation step
    use_analytical_jacobians : bool
        indicate if we use an analytical jacobian matrix during the IK or not

    Returns
    -------
    tuple
        - pos_init :  np.ndarray
        - parameters: np.ndarray
        - output dict
    """

    c3d_kinova, labels_markers, points_c3d = load_c3d_file(task)

    thorax_values, q_upper_limb = two_step_inverse_kinematics(
        c3d_file=c3d_kinova,
        labels_markers=labels_markers,
        points_c3d=points_c3d,
        model_path_6_dofs=Models.WU_INVERSE_KINEMATICS.value,
        model_path_fixed_template=Models.WU_WITHOUT_FLOATING_BASE_TEMPLATE.value,
        model_path_fixed_with_variables=Models.WU_WITHOUT_FLOATING_BASE_VARIABLES.value,
    )

    # rewrite the models with the location of the floating base
    template_file_merge = Models.WU_AND_KINOVA_WITHOUT_FLOATING_BASE_WITH_6_DOF_SUPPORT_TEMPLATE.value
    new_biomod_file_merge = Models.WU_AND_KINOVA_WITHOUT_FLOATING_BASE_WITH_6_DOF_SUPPORT_VARIABLES.value
    add_header(
        biomod_file_name=template_file_merge, new_biomod_file_name=new_biomod_file_merge, variables=thorax_values
    )

    # exo for step 2
    biorbd_model_merge = biorbd.Model(new_biomod_file_merge)

    markers_names = [value.to_string() for value in biorbd_model_merge.markerNames()]
    tracked_markers = markers_names.copy()
    tracked_markers.remove('Table:Table5')
    tracked_markers.remove('Table:Table6')

    markers = np.zeros((3, len(markers_names), len(points_c3d[0, 0, :])))

    # add the extra marker Table:Table6 to the experimental data based on the location of the Table:Table5

    new_row = np.zeros((points_c3d.shape[0], 1, points_c3d.shape[2]))
    points_c3d = np.append(points_c3d, new_row, axis=1)

    # add marker to keep the origin of kineva arm in horizontal plan
    labels_markers.append("Table:Table6")

    points_c3d[:3, labels_markers.index("Table:Table6"), :] = points_c3d[:3, labels_markers.index("Table:Table5"), :]

    # apply offset to the markers
    offset = np.array([0, 0, 100])  # meters
    points_c3d = move_marker(marker_to_move=labels_markers.index("Table:Table6"), c3d_point=points_c3d, offset=offset)

    # in the class of calibration
    for i, name in enumerate(markers_names):
        if name in labels_markers:
            markers[:, i, :] = points_c3d[:3, labels_markers.index(name), :] / get_unit_division_factor(c3d_kinova)

    name_dof = [i.to_string() for i in biorbd_model_merge.nameDof()]
    kinematic_dof = [i for i in name_dof if "part7" not in i]
    wu_dof = [i for i in name_dof if not "part" in i]
    parameters = [i for i in name_dof if "part7" in i]
    kinova_dof = [i for i in name_dof if "part" in i and not "7" in i]

    nb_dof_wu_model = len(wu_dof)
    # nb_parameters = len(parameters)
    # nb_dof_kinova = len(kinova_dof)

    # prepare the inverse kinematics of the first step of the algorithm
    # initialize q with zeros
    q_first_ik = np.zeros((biorbd_model_merge.nbQ(), markers.shape[2]))
    # initialize human dofs with previous results of inverse kinematics
    q_first_ik[:nb_dof_wu_model, :] = q_upper_limb  # human

    nb_frames = markers.shape[2]

    #weight correpond to [table, model, continuity, theta, rotation]
    weight = np.array([100000, 10000, 50000, 500,100])

    #the last segment is the number 45

    kcc = KinematicChainCalibration(
        biorbd_model=biorbd_model_merge,
        markers_model=markers_names,
        markers=markers,
        closed_loop_markers=["Table:Table5", "Table:Table6"],
        tracked_markers=tracked_markers,
        parameter_dofs=parameters,
        kinematic_dofs=kinematic_dof,
        weights=weight,
        q_ik_initial_guess=q_first_ik,
        nb_frames_ik_step=nb_frames,
        nb_frames_param_step=nb_frame_param_step,
        randomize_param_step_frames=True,
        use_analytical_jacobians=use_analytical_jacobians,
        segment_id_with_vertical_z=45,
        param_solver= "leastsquare",
        ik_solver= "ipopt",
    )

    return biorbd_model_merge, markers, kcc


def main(
    task: TasksKinova,
    show_animation: bool,
    export_model: bool,
    nb_frame_param_step: int,
    use_analytical_jacobians: bool,
):
    """
        this function is the main script executed in function of the Task and parameters

        Parameters
        ----------
        task: TasksKinova
            the task
        show_animation: bool
            if true we animate the result
        export_model: bool
            the biorbd.Model is export to the pre-defined .bioMod
        nb_frame_param_step:
            Number of franes used for the parameter optimisation step
        use_analytical_jacobians : bool
            indicate if we use an analytical jacobian matrix during the IK or not

        Returns
        -------
        tuple
            - pos_init :  np.ndarray
            - parameters: np.ndarray
            - output dict
        """

    biorbd_model_merge, markers, kcc = prepare_kcc(
        task=task,
        nb_frame_param_step= nb_frame_param_step,
        use_analytical_jacobians=use_analytical_jacobians
    )

    q_out, parameters = kcc.solve(threshold=1e-5)[0],kcc.solve(threshold=1e-5)[1]
    output = kcc.solution()

    if show_animation:
        b = bioviz.Viz(loaded_model=biorbd_model_merge, show_muscles=False, show_floor=False)
        b.load_experimental_markers(markers)
        # b.load_movement(np.array(q0, q0).T)
        b.load_movement(q_out)
        b.exec()

        print("done")

    if export_model:
        export_to_biomod(
            pos_init=q_out,
            # q_ik_with_floating_base=ik_with_floating_base.q,
            task=task,
            biorbd_model_merge=biorbd_model_merge,
        )

    return q_out, parameters, output




if __name__ == "__main__":
    main(task=TasksKinova.DRINK, show_animation=True,export_model=False, nb_frame_param_step=100 , use_analytical_jacobians=False)

