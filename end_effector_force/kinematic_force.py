import biorbd as biorbd_eigen
import matplotlib.pyplot as plt
import numpy as np
from biorbd import InverseKinematics
from pyomeca import Markers
from pyorerun import BiorbdModel, PhaseRerun
from scipy.signal import butter, filtfilt

SHOW_FIRST_IK = False


def inverse_kinematics(biorbd_model, c3d_path, animated=True):
    xp_markers = Markers.from_c3d(
        c3d_path,
        usecols=[
            "Table",
            "First_link_end",
            "second_link_end",
            "third_end",
            "fourth_link_right",
            "fourth_link_left",
            "fifth_link_start_right",
            "fifth_link_start_left",
            "fifth_link_top",
            "seventh_link_start",  # this is inverted !!!!! ok in .biomod but not in .c3d
            "sixth_link_end",  # this is inverted !!!!! ok in .biomod but not in .c3d
            "seventh_link second",
        ],
        prefix_delimiter="Kinova:",
    )
    xp_markers[:3, :, :] /= 1000
    xp_markers.attrs["units"] = "m"

    numpy_xp_markers = xp_markers.to_numpy()[0:3, :, :]
    nb_frames = numpy_xp_markers.shape[2]
    t_span = xp_markers.time

    ik = InverseKinematics(biorbd_model, numpy_xp_markers)
    q = ik.solve("only_lm")

    # animate()

    return q


def animate():
    pass


def compute_end_effector_force_static(biorbd_model, q_kinova):
    # two markers on the end effector in order to get torques applied on the end effector
    # but the total force applied on the end effector is the sum of the two forces

    J1 = biorbd_model.markersJacobian(q_kinova)[-2].to_array()
    J2 = biorbd_model.markersJacobian(q_kinova)[-1].to_array()
    print(J1.T)
    print(J2.T)

    Jtot_T = np.concatenate((J1.T, J2.T), axis=1)
    # Jtot_T_inv = np.linalg.inv(Jtot_T) not square pseudo inverse
    Jtot_T_inv = np.linalg.pinv(Jtot_T)

    print(Jtot_T_inv)

    tau = biorbd_model.ligamentsJointTorque(q_kinova, np.zeros((biorbd_model.nbQ(),))).to_array()
    print("Tau", tau)
    qdot_kinova = np.zeros((biorbd_model.nbQ(),))
    gravitational_effects = biorbd_model.NonLinearEffect(q_kinova, qdot_kinova).to_array()
    end_effector_force = Jtot_T_inv @ (- tau + gravitational_effects)

    total_end_effector_force = end_effector_force[0:3] + end_effector_force[3:6]

    print("Force", end_effector_force)
    print("Somme des forces", total_end_effector_force)

    # total_torque applied on the end effector first marker
    markers = biorbd_model.markers(q_kinova)
    markers_1 = markers[-2].to_array()
    markers_2 = markers[-1].to_array()
    lever_arm = markers_1 - markers_2

    total_torque_in_1 = np.cross(lever_arm, end_effector_force[3:6])
    print("Torque in 1", total_torque_in_1)

    return np.concatenate((total_torque_in_1, total_end_effector_force))


def compute_end_effector_force_dynamic(biorbd_model, q_kinova, qdot_kinova, qddot_kinova):
    # two markers on the end effector in order to get torques applied on the end effector
    # but the total force applied on the end effector is the sum of the two forces

    J1 = biorbd_model.markersJacobian(q_kinova)[-2].to_array()
    J2 = biorbd_model.markersJacobian(q_kinova)[-1].to_array()
    print(J1.T)
    print(J2.T)

    Jtot_T = np.concatenate((J1.T, J2.T), axis=1)
    # Jtot_T_inv = np.linalg.inv(Jtot_T) not square pseudo inverse
    Jtot_T_inv = np.linalg.pinv(Jtot_T)

    print(Jtot_T_inv)

    tau = biorbd_model.ligamentsJointTorque(q_kinova, np.zeros((biorbd_model.nbQ(),))).to_array()
    # tau is already the ligament nonlinear_effect vector
    # print("Tau", tau)
    mass_matrix = biorbd_model.massMatrix(q_kinova).to_array()
    nonlinear_effects = biorbd_model.NonLinearEffect(q_kinova, qdot_kinova).to_array()
    end_effector_force = Jtot_T_inv @ (-tau + mass_matrix @ qddot_kinova + nonlinear_effects)
    # end_effector_force = Jtot_T_inv @ (mass_matrix @ qddot_kinova + nonlinear_effects)

    total_end_effector_force = end_effector_force[0:3] + end_effector_force[3:6]

    print("Force", end_effector_force)
    print("Somme des forces", total_end_effector_force)

    # total_torque applied on the end effector first marker
    markers = biorbd_model.markers(q_kinova)
    markers_1 = markers[-2].to_array()
    markers_2 = markers[-1].to_array()
    lever_arm = markers_1 - markers_2

    total_torque_in_1 = np.cross(lever_arm, end_effector_force[3:6])
    print("Torque in 1", total_torque_in_1)

    return np.concatenate((total_torque_in_1, total_end_effector_force))


def main():
    nb_segments = 7
    kinova_model_paths = f"KINOVA_arm_right_complete_{nb_segments}_segments.bioMod"
    biorbd_eigen_model = biorbd_eigen.Model(kinova_model_paths)
    # c3d_path = "../data_new/kinova_dynamique_pierre.c3d"
    c3d_path = "../data_new/kinova_martin_boire_f3_1.c3d"

    q = inverse_kinematics(biorbd_eigen_model, c3d_path)
    frame_rate = Markers.from_c3d(c3d_path).rate

    if SHOW_FIRST_IK:
        rr_model = BiorbdModel(kinova_model_paths)

        nb_frames = len(q[0, :])
        nb_seconds = 2
        t_span = np.linspace(0, nb_seconds, nb_frames)

        rerun_biorbd = PhaseRerun(t_span)
        rerun_biorbd.add_animated_model(rr_model, q)
        rerun_biorbd.biorbd_models.rerun_models[0].ligaments.ligament_properties.radius = 0.005
        rerun_biorbd.rerun("KINOVA")

    spring_length = np.zeros((q.shape[1], 1))
    for frame in range(q.shape[1]):
        spring_length[frame] = biorbd_eigen_model.ligament(0).length(biorbd_eigen_model, q[:, frame])

    plt.figure()
    plt.plot(spring_length, label="Spring length")
    plt.plot(np.ones_like(spring_length) * 0.125, "--", label="Rest length")
    plt.legend()
    # plt.show()

    # use butter and filtfilt before differentiating
    cut_freq = 5
    normalized_frequency = cut_freq / (frame_rate / 2)
    b, a = butter(4, normalized_frequency, 'low')
    q_filtered = np.zeros_like(q)
    for i in range(q.shape[0]):
        q_filtered[i, :] = filtfilt(b, a, q[i, :])

    qdot = np.gradient(q_filtered, 1 / frame_rate, axis=1)
    qddot = np.gradient(qdot, 1 / frame_rate, axis=1)
    # qdot = np.zeros_like(q)
    # qddot = np.zeros_like(q)

    nb_q = biorbd_eigen_model.nbQ()
    fig, ax = plt.subplots(round(np.sqrt(nb_q)), round(np.sqrt(nb_q)))
    for i in range(nb_q):
        row, col = i // round(np.sqrt(nb_q)), i % round(np.sqrt(nb_q))
        ax[row, col].plot(q[i, :])
        ax[row, col].plot(q_filtered[i, :])
        ax[row, col].set_title(f"q{i}")

    fig, ax = plt.subplots(round(np.sqrt(nb_q)), round(np.sqrt(nb_q)))
    for i in range(nb_q):
        row, col = i // round(np.sqrt(nb_q)), i % round(np.sqrt(nb_q))
        ax[row, col].plot(qdot[i, :])
        ax[row, col].set_title(f"qdot{i}")

    fig, ax = plt.subplots(round(np.sqrt(nb_q)), round(np.sqrt(nb_q)))
    for i in range(nb_q):
        row, col = i // round(np.sqrt(nb_q)), i % round(np.sqrt(nb_q))
        ax[row, col].plot(qddot[i, :])
        ax[row, col].set_title(f"qddot{i}")

    # plt.show()

    model_path_kinova = f"KINOVA_arm_right_complete_{nb_segments}_segments_extra_markers.bioMod"
    model_eigen_kinova = biorbd_eigen.Model(model_path_kinova)

    total_torque_static = np.zeros((6, q.shape[1]))
    total_torque_dynamic = np.zeros((6, q.shape[1]))
    for frame in range(q.shape[1]):
        q_frame = q_filtered[:, frame]

        total_torque_static[:, frame] = compute_end_effector_force_static(model_eigen_kinova, q_frame)
        total_torque_dynamic[:, frame] = compute_end_effector_force_dynamic(model_eigen_kinova, q_frame, qdot[:, frame],
                                                                            qddot[:, frame])

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(total_torque_static[0:3].T, label=["Mx", "My", "Mz"])
    ax[0].plot(total_torque_dynamic[0:3].T, label=["Mx", "My", "Mz"], ls="--")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Torque (N.m)")
    ax[0].legend()

    ax[1].plot(total_torque_static[3:6].T, label=["Fx", "Fy", "Fz"])
    ax[1].plot(total_torque_dynamic[3:6].T, label=["Fx", "Fy", "Fz"], ls="--")
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Force (N)")
    ax[1].legend()

    plt.show()


if __name__ == "__main__":
    main()
