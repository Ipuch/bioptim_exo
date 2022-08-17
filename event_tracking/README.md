# OCP 

## tracking_task
torque driven OCP aiming at matching the position of the markers at the beginning and the end of a phase to those 
provided by the c3d files 
## constraint_task
torque driven OCP constraining the q values at the beginning and the end of a phase to those obtained by performing inverse 
kinematics using the c3d files
## constraint_task_fixed
OCP based on `constraint_task` using a model without the DOF associated to the sternoclavicular and scapular joints
## constraint_task_quat
OCP based on `constraint_task` using a model replacing the XYZ rotations of the glenohumeral joint by a quaternion
## constraint_task_quat_multiphase
OCP based on `constraint_task_quat` considering multiple phases
## constraint_task_quat_muscles
muscle driven OCP based on `constraint_task_quat`