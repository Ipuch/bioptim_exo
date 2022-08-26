from bioptim import OptimalControlProgram
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import biorbd
from quat import quat2eul
from data.enums import Tasks
from models.enums import Models
import data.load_events as load_events


def get_muscular_torque(x, act, model):
    """
    Get the muscular torque.
    """
    muscular_torque = np.zeros((model.nbQ(), x.shape[1]))
    states = model.stateSet()  # Get the muscle state set
    for i in range(act.shape[1]):
        for a, state in zip(act[:, i], states):
            state.setActivation(a)  # And fill it with the current value
        muscular_torque[:, i] = model.muscularJointTorque(
            states, x[: model.nbQ(), i], x[model.nbQ() : model.nbQ() * 2, i]
        ).to_array()
    return muscular_torque


def plot(task, solution_tau: str, solution_muscles: str, solution_offset: str):
    c3d_path = task.value

    ocp_tau, sol_tau = OptimalControlProgram.load(solution_tau)
    n_shooting_tau = ocp_tau.original_values["n_shooting"] + 1

    ocp_muscles, sol_muscles = OptimalControlProgram.load(solution_muscles)
    n_shooting_muscles = ocp_muscles.original_values["n_shooting"] + 1

    ocp_offset, sol_offset = OptimalControlProgram.load(solution_offset)
    n_shooting_offset = ocp_offset.original_values["n_shooting"] + 1

    m = biorbd.Model(Models.WU_WITHOUT_FLOATING_BASE_VARIABLES.value)
    marker_ref = [m.to_string() for m in m.markerNames()]
    # names = [i.to_string() for i in m.nameDof()]
    names = [
        "sterno-clavicular protraction(+)/retraction(-)",
        "sterno-clavicular elevation(+)/depression(-)",
        None,
        "scapular upward(-)/downward(+) rotation",
        "scapular protraction(+)/retraction(-)",
        "scapular elevation(-)/depression(+)",
        "shoulder",
        "shoulder abduction(+)/adduction(-)",
        "shoulder",
        "elbow flexion(+)/extension(-)",
        "wrist pronation(+)/supination(-)",
    ]

    event = load_events.LoadEvent(c3d_path=c3d_path, marker_list=marker_ref)

    if c3d_path == Tasks.EAT.value:
        start_frame = event.get_frame(1)
        end_frame = event.get_frame(2)
        phase_time = event.get_time(2) - event.get_time(1)
    else:
        start_frame = event.get_frame(0)
        end_frame = event.get_frame(1)
        phase_time = event.get_time(1) - event.get_time(0)

    data_path = c3d_path.removesuffix(c3d_path.split("/")[-1])

    ik_path = data_path + Models.WU_INVERSE_KINEMATICS_XYZ.name + "_" + c3d_path.split("/")[-1].removesuffix(".c3d")
    ik_offset_path = (
        data_path + Models.WU_INVERSE_KINEMATICS_XYZ_OFFSET.name + "_" + c3d_path.split("/")[-1].removesuffix(".c3d")
    )

    q_ik = np.loadtxt(ik_path + "_q.txt")[6:][:, start_frame : end_frame + 1]
    q_ik_offset = np.loadtxt(ik_offset_path + "_q.txt")[6:][:, start_frame : end_frame + 1]

    q_euler = np.zeros((3, n_shooting_tau))
    Quaternion = np.zeros(4)
    for i in range(n_shooting_tau):
        Q = sol_tau.states["q"][:, i]
        Quaternion[0] = Q[10]
        Quaternion[1:] = Q[5:8]
        euler = quat2eul(Quaternion)
        q_euler[:, i] = np.array(euler)
    q_tau = np.zeros((m.nbQ(), n_shooting_tau))
    q_tau[:5] = sol_tau.states["q"][:5]
    q_tau[5:8] = q_euler
    q_tau[8:] = sol_tau.states["q"][8:10]

    q_offset = sol_offset.states["q"]

    q_euler = np.zeros((3, n_shooting_muscles))
    Quaternion = np.zeros(4)
    for i in range(n_shooting_muscles):
        Q = sol_muscles.states["q"][:, i]
        Quaternion[0] = Q[10]
        Quaternion[1:] = Q[5:8]
        euler = quat2eul(Quaternion)
        q_euler[:, i] = np.array(euler)
    q_muscles = np.zeros((m.nbQ(), n_shooting_muscles))
    q_muscles[:5] = sol_muscles.states["q"][:5]
    q_muscles[5:8] = q_euler
    q_muscles[8:] = sol_muscles.states["q"][8:10]

    qdot_ik = np.loadtxt(ik_path + "_qdot.txt")[6:][:, start_frame : end_frame + 1]
    qdot_ik_offset = np.loadtxt(ik_offset_path + "_qdot.txt")[6:][:, start_frame : end_frame + 1]

    qdot_tau = sol_tau.states["qdot"]
    qdot_offset = sol_offset.states["qdot"]
    qdot_muscles = sol_muscles.states["qdot"]

    tau_ik = np.loadtxt(ik_path.replace("_XYZ", "") + "_tau.txt")[6:][:, start_frame : end_frame + 1]
    tau_ik_offset = np.loadtxt(ik_offset_path + "_tau.txt")[6:][:, start_frame : end_frame + 1]

    tau_tau = sol_tau.controls["tau"]
    tau_offset = sol_offset.controls["tau"]
    tau_muscles = sol_muscles.controls["tau"] + get_muscular_torque(
        np.concatenate([q_muscles, qdot_muscles]), sol_muscles.controls["muscles"], m
    )

    y_ik = [q_ik, qdot_ik, tau_ik]
    y_ik_offset = [q_ik_offset, qdot_ik_offset, tau_ik_offset]
    y_tau = [q_tau, qdot_tau, tau_tau]
    y_offset = [q_offset, qdot_offset, tau_offset]
    y_muscles = [q_muscles, qdot_muscles, tau_muscles]
    title = ["generalized coordinates", "generalized velocities", "generalized torque"]
    y_title = ["(°)", "(°/s)", "(N/m)"]
    colors = px.colors.qualitative.Set2[2:]
    rows = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4]
    columns = [1, 2, 1, 2, 3, 1, 2, 3, 1, 2]

    for n in range(3):
        ratio = 1 if n == 2 else 180 / np.pi
        fig = make_subplots(rows=5, cols=3, subplot_titles=names)
        j = 0
        x_ik = np.linspace(0, phase_time, end_frame - start_frame + 1)
        x_tau = np.linspace(0, phase_time, n_shooting_tau)
        x_offset = np.linspace(0, phase_time, n_shooting_offset)
        x_muscles = np.linspace(0, phase_time, n_shooting_muscles)
        for i in range(q_ik.shape[0]):
            showlegend = True if i == 0 else False
            ii = i - j * 4
            fig.add_trace(
                go.Scatter(
                    x=x_ik,
                    y=y_ik[n][i] * ratio,
                    legendgroup="1",
                    name="inverse kinematics",
                    showlegend=showlegend,
                    line=dict(color=colors[0]),
                ),
                row=rows[i],
                col=columns[i],
            )
            fig.add_trace(
                go.Scatter(
                    x=x_ik,
                    y=y_ik_offset[n][i] * ratio,
                    legendgroup="2",
                    name="inverse kinematics with offset",
                    showlegend=showlegend,
                    line=dict(color=colors[1]),
                ),
                row=rows[i],
                col=columns[i],
            )
            fig.add_trace(
                go.Scatter(
                    x=x_tau,
                    y=y_tau[n][i] * ratio,
                    legendgroup="3",
                    name="torque driven",
                    showlegend=showlegend,
                    line=dict(color=colors[2]),
                ),
                row=rows[i],
                col=columns[i],
            )
            fig.add_trace(
                go.Scatter(
                    x=x_muscles,
                    y=y_muscles[n][i] * ratio,
                    legendgroup="4",
                    name="muscle driven",
                    showlegend=showlegend,
                    line=dict(color=colors[3]),
                ),
                row=rows[i],
                col=columns[i],
            )
            fig.add_trace(
                go.Scatter(
                    x=x_offset,
                    y=y_offset[n][i] * ratio,
                    legendgroup="5",
                    name="torque driven with offset",
                    showlegend=showlegend,
                    line=dict(color=colors[4]),
                ),
                row=rows[i],
                col=columns[i],
            )
            if ii == 3:
                j += 1

        fig.update_layout(
            height=1000,
            # width=1800,
            title={"text": f"{task.name} : {title[n]}", "y": 1, "x": 0.5, "xanchor": "center", "yanchor": "top"},
            template="simple_white",
        )
        fig.update_xaxes(title_text="time (s)")
        fig.update_yaxes(title_text=y_title[n])
        fig.show()


# plot(
#     Tasks.TEETH,
#     "solutions/torque_driven/without_tracking_markers/dents/F0_dents_05_2022_08_25_16_58_37_810470.bo",
#     "solutions/muscle_driven/dents/F0_dents_05_2022_08_17_19_38_12_717853.bo",
#     "solutions/torque_driven_offset/dents/F0_dents_05_2022_08_26_13_53_24_362296.bo"
#      )

# plot(
#     Tasks.EAT,
#     "solutions/torque_driven/without_tracking_markers/manger/F0_manger_05_2022_08_25_18_01_43_130921.bo",
#     "solutions/muscle_driven/manger/F0_manger_05_2022_08_23_19_26_42_078272.bo",
#     "solutions/torque_driven_offset/manger/F0_manger_05_2022_08_26_14_48_55_312670.bo"
# )

# plot(
#     Tasks.HEAD,
#     "solutions/torque_driven/without_tracking_markers/tete/F0_tete_05_2022_08_25_17_18_04_468630.bo",
#     "solutions/muscle_driven/tete/F0_tete_05_2022_08_24_10_55_31_451971.bo",
#     "solutions/torque_driven_offset/tete/F0_tete_05_2022_08_26_13_58_21_171507.bo"
# )

plot(
    Tasks.ARMPIT,
    "solutions/torque_driven/without_tracking_markers/aisselle/F0_aisselle_05_2022_08_25_16_14_00_581409.bo",
    "solutions/muscle_driven/aisselle/F0_aisselle_05_2022_08_26_10_01_36_700173.bo",
    "solutions/torque_driven_offset/aisselle/F0_aisselle_05_2022_08_26_14_05_20_110908.bo",
)
