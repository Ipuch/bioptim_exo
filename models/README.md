# Models

## KINOVA_arm_deprecated
This model is a KINOVA arm model but without using STL files, It isn't used anymore. It was the first one build with the denavit-hartenberg formalism, modelling the three hinges as two slides.

## KINOVA_arm_reverse_left
This model is a KINOVA arm model in left configuration. The origin of this model in on the arm support.

## KINOVA_arm_reverse_right
This model is a KINOVA arm model in right configuration. The origin of this model in on the arm support.

## KINOVA_merge
This is the model with KINOVA arm and wu model merged. The ulna is the parent of the KINOVA arm.

## KINOVA_merge_6dof
This is the model is the same as KINOVA_merge but with 6 dof between the ulna and the support KINOVA arm (segment 0).
This model is used to get the rototrans matrix between ulna and kinova arm.

## wu_converted_definitif
The floating remain free. The initial RT is changed. This model is the wu model but with initial position changed. The initial position is adapted with the inverse kinematics output.
This model is used in tracking_ocp, it has thorax markers (MAN, XYP, C7, T10).

## wu_converted_definitif_inverse_kinematics
This model is for inverse kinematics script, ranges are adapted and the floating base is free.
This model is not used in tracking_ocp, it has thorax markers (MAN, XYP, C7, T10).
The initial RT is changed.

## wu_converted_definitif_without_floating_base
The initial RT is changed. This model has no dof for the first segment. 
This model is used in tracking_ocp, it does not have thorax markers (MAN, XYP, C7, T10).

## wu_converted_definitif_without_floating_base_template
The initial RT values are replaced by variables names. 
This model is used as a template in utils.

## wu_converted_definitif_without_floating_base_template_with_variables
A header introducing the variables and their values is added to the template, right below the first line stating the version.
This model is used in tracking_ocp, in situations without floating.

## wu_converted_definitif_without_floating_base_template_fixed
The DOF associated to the sternoclavicular and scapulothoracic joints are eliminated
This model is used as a template in utils.

## wu_converted_definitif_without_floating_base_template_quat
The XYZ rotations of the glenohumeral and scapulothoracic joints are replaced by quaternions
This model is used as a template in utils.

## wu_converted_definitif_without_modif
This model is the wu model with almost no changement except for the range of q to allow convinient inverse kinematics with it. It comes from the converter Opensim to Biomod. It still has the 2 segment of the thorax that has been removed from the other model: thorax_rotation_transform and thorax_reset_axis.
This model has thorax markers (MAN, XYP, C7, T10).

## wu_converted_definitif_without_modif_quat
The XYZ rotations of the glenohumeral joint of the wu_converted_definitif_without_modif model are replaced by a quaternion






