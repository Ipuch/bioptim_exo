import biorbd as biorbd_eigen
import numpy as np

from models.enums import Models


def main():
    model_path_kinova = Models.KINOVA_RIGHT.value
    # model_mx_kinova = biorbd.Model(model_path_kinova)
    model_eigen_kinova = biorbd_eigen.Model(model_path_kinova)

    q_kinova = np.array([-0.26523624, 0.198916, 0.02813227, 0.09130462, -1.27216358,
                         0.26166478])

    # two markers on the end effector in order to get torques applied on the end effector
    # but the total force applied on the end effector is the sum of the two forces

    J1 = model_eigen_kinova.markersJacobian(q_kinova)[-2].to_array()
    J2 = model_eigen_kinova.markersJacobian(q_kinova)[-1].to_array()
    print(J1.T)
    print(J2.T)

    Jtot_T = np.concatenate((J1.T, J2.T), axis=1)
    Jtot_T_inv = np.linalg.inv(Jtot_T)
    print(Jtot_T_inv)

    tau = model_eigen_kinova.ligamentsJointTorque(q_kinova, np.zeros((6,))).to_array()
    print("Tau", tau)

    end_effector_force = Jtot_T_inv @ tau

    total_end_effector_force = end_effector_force[0:3] + end_effector_force[3:6]

    print("Force", end_effector_force)
    print("Somme des forces", total_end_effector_force)

    # total_torque applied on the end effector first marker
    markers = model_eigen_kinova.markers(q_kinova)
    markers_1 = markers[-2].to_array()
    markers_2 = markers[-1].to_array()
    lever_arm = markers_1 - markers_2

    total_torque_in_1 = np.cross(lever_arm, end_effector_force[3:6])
    print("Torque in 1", total_torque_in_1)


if __name__ == "__main__":
    main()

# ligament_joint_torque(q,qdot)
