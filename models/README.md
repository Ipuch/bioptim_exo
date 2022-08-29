# Models

## KINOVA_DEPRECATED
This model is a KINOVA arm model but without using STL files, It isn't used anymore. It was the first one build with the denavit-hartenberg formalism, modelling the three hinges as two slides.

## KINOVA_LEFT
This model is a KINOVA arm model in left configuration. The origin of this model in on the arm support.

## KINOVA_RIGHT
This model is a KINOVA arm model in right configuration. The origin of this model in on the arm support.

## WU
The floating remain free. The initial RT is changed. This model is the wu model but with initial position changed. The initial position is adapted with the inverse kinematics output.
This model is used in tracking_ocp, it has thorax markers (MAN, XYP, C7, T10).

## WU_INVERSE_KINEMATICS
This model is for inverse kinematics script, ranges are adapted and the floating base is free.
This model is not used in tracking_ocp, it has thorax markers (MAN, XYP, C7, T10).
The initial RT is changed.

## WU_INVERSE_KINEMATICS_XYZ
The XYZ rotations of the glenohumeral joint are defined as successive rotations around orthogonal axes.
This model is used to perform inverse kinematics and provide the necessary files, where the state values at the glenohumeral joint can be converted to a quaternion.

## WU_INVERSE_KINEMATICS_XYZ_OFFSET
A rotation offset is introduced to the glenohumeral joint, allowing the shoulder to be located behind the frontal plane.
This model is used to perform inverse kinematics and provide state and control values compatible with the `WU_WITHOUT_FLOATING_BASE_OFFSET_VARIABLE` model.

## WU_WITHOUT_FLOATING_BASE
The initial RT is changed. This model has no dof for the first segment. 
This model is used in tracking_ocp, it does not have thorax markers (MAN, XYP, C7, T10).

## WU_WITHOUT_FLOATING_BASE_TEMPLATE
The initial RT values are replaced by variables names. 
This model is used as a template in utils.

## WU_WITHOUT_FLOATING_BASE_VARIABLES
A header introducing the variables and their values is added to the template, right below the first line stating the version.
This model is used in tracking_ocp, in situations without floating.

## WU_WITHOUT_FLOATING_BASE_FIXED_TEMPLATE
The DOF associated to the sternoclavicular and scapular joints are eliminated.
This model is used as a template in utils.

## WU_WITHOUT_FLOATING_BASE_FIXED_VARIABLES
`WU_WITHOUT_FLOATING_BASE_FIXED_TEMPLATE` with a header introducing the RT values of the thorax.

## WU_WITHOUT_FLOATING_BASE_OFFSET_TEMPLATE
A rotation offset is introduced to the glenohumeral joint, allowing the shoulder to be located behind the frontal plane.

## WU_WITHOUT_FLOATING_BASE_OFFSET_VARIABLES
`WU_WITHOUT_FLOATING_BASE_OFFSET_TEMPLATE` with a header introducing the RT values of the thorax.

## WU_WITHOUT_FLOATING_BASE_QUAT_TEMPLATE
The XYZ rotations of the glenohumeral joint is replaced by a quaternion.
This model is used as a template in utils.

## WU_WITHOUT_FLOATING_BASE_QUAT_VARIABLES
`WU_WITHOUT_FLOATING_BASE_QUAT_TEMPLATE` with a header introducing the RT values of the thorax.

## WU_WITHOUT_FLOATING_BASE_QUAT_DEGROOTE_TEMPLATE
This model is based on `WU_WITHOUT_FLOATING_BASE_QUAT_TEMPLATE` but uses degroote type muscles.

## WU_WITHOUT_FLOATING_BASE_QUAT_DEGROOTE_VARIABLES
`WU_WITHOUT_FLOATING_BASE_QUAT_DEGROOTE_TEMPLATE` with a header introducing the RT values of the thorax.

## WU_WITHOUT_MODIF
This model is the wu model with almost no changement except for the range of q to allow convinient inverse kinematics with it. It comes from the converter Opensim to Biomod. It still has the 2 segment of the thorax that has been removed from the other model: thorax_rotation_transform and thorax_reset_axis.
This model has thorax markers (MAN, XYP, C7, T10).

## WU_WITHOUT_MODIF_QUAT
The XYZ rotations of the glenohumeral joint of the wu_converted_definitif_without_modif model are replaced by a quaternion.

## WU_AND_KINOVA
This is the model with KINOVA arm and wu model merged. The ulna is the parent of the KINOVA arm.

## WU_AND_KINOVA_INVERSE_KINEMATICS

## WU_AND_KINOVA_WITHOUT_FLOATING_BASE_TEMPLATE

## WU_AND_KINOVA_WITHOUT_FLOATING_BASE_VARIABLES

## WU_AND_KINOVA_WITHOUT_FLOATING_BASE_WITH_6_DOF_SUPPORT_TEMPLATE

## WU_AND_KINOVA_WITHOUT_FLOATING_BASE_WITH_6_DOF_SUPPORT_VARIABLES

## WU_AND_KINOVA_WITHOUT_FLOATING_BASE_WITH_ROTOTRANS_SUPPORT_TEMPLATE

## WU_AND_KINOVA_WITHOUT_FLOATING_BASE_WITH_ROTOTRANS_SUPPORT_VARIABLES








