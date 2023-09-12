import biorbd
from casadi import MX, SX, vertcat, horzcat, transpose, inv
from bioptim import (
    BiorbdModel,
    Node,
    OptimalControlProgram,
    DynamicsList,
    ConfigureProblem,
    DynamicsFcn,
    DynamicsFunctions,
    RigidBodyDynamics,
    NonLinearProgram,
    DynamicsEvaluation,
    FatigueList,
    DefectType,
)
from external_forces import transport_spatial_force


def custom_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram):
    """
    Tell the program which variables are states and controls.
    The user is expected to use the ConfigureProblem.configure_xxx functions.

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    my_additional_factor: int
        An example of an extra parameter sent by the user
    """

    ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)

    name = "q_k"
    name_q_k = [f"{name}_{n}" for n in nlp.model.extra_models[0].name_dof]
    ConfigureProblem.configure_new_variable(
        ConfigureProblem.configure_new_variable(
            name, name_q_k, ocp, nlp, as_states=False, as_controls=True, fatigue=False, axes_idx=None
        )
    )

    ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.torque_driven)


def torque_driven(
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        stochastic_variables: MX.sym,
        nlp,
        with_contact: bool,
        with_passive_torque: bool,
        with_ligament: bool,
        with_friction: bool,
        rigidbody_dynamics: RigidBodyDynamics,
        fatigue: FatigueList,
) -> DynamicsEvaluation:
    """
    Forward dynamics driven by joint torques, optional external forces can be declared.

    Parameters
    ----------
    states: MX.sym
        The state of the system
    controls: MX.sym
        The controls of the system
    parameters: MX.sym
        The parameters of the system
    nlp: NonLinearProgram
        The definition of the system
    with_contact: bool
        If the dynamic with contact should be used
    with_passive_torque: bool
        If the dynamic with passive torque should be used
    with_ligament: bool
        If the dynamic with ligament should be used
    with_friction: bool
        If the dynamic with friction should be used
    rigidbody_dynamics: RigidBodyDynamics
        which rigidbody dynamics should be used
    fatigue : FatigueList
        A list of fatigue elements

    Returns
    ----------
    DynamicsEvaluation
        The derivative of the states and the defects of the implicit dynamics
    """

    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)

    dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)

    tau = DynamicsFunctions.__get_fatigable_tau(nlp, states, controls, fatigue)
    tau = tau + nlp.model.passive_joint_torque(q, qdot) if with_passive_torque else tau
    tau = tau + nlp.model.ligament_joint_torque(q, qdot) if with_ligament else tau
    tau = tau + nlp.model.friction_coefficients @ qdot if with_friction else tau

    q_k = DynamicsFunctions.get(nlp.controls["q_k"], controls)
    f_ext_from_kinova = compute_kinova_terminal_forces(nlp.model.extra_models[0].model, q_k)
    f_ext = biorbd.VecBiorbdSpatialVector()
    f_ext.append(f_ext_from_kinova)

    ddq = nlp.model.forward_dynamics(q, qdot, tau, f_ext)

    dxdt = MX(nlp.states.shape, ddq.shape[1])
    dxdt[nlp.states["q"].index, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
    dxdt[nlp.states["qdot"].index, :] = ddq

    defects = None
    if rigidbody_dynamics is not RigidBodyDynamics.ODE or (
            rigidbody_dynamics is RigidBodyDynamics.ODE and nlp.ode_solver.defects_type == DefectType.IMPLICIT
    ):
        if not with_contact and fatigue is None:
            qddot = DynamicsFunctions.get(nlp.states_dot["qddot"], nlp.states_dot.scaled.mx_reduced)
            tau_id = nlp.model.inverse_dynamics(q, qdot, qddot, f_ext)
            defects = MX(dq.shape[0] + tau_id.shape[0], tau_id.shape[1])

            dq_defects = []
            for _ in range(tau_id.shape[1]):
                dq_defects.append(
                    dq
                    - DynamicsFunctions.compute_qdot(
                        nlp,
                        q,
                        DynamicsFunctions.get(nlp.states_dot.scaled["qdot"], nlp.states_dot.scaled.mx_reduced),
                    )
                )
            defects[: dq.shape[0], :] = horzcat(*dq_defects)
            # We modified on purpose the size of the tau to keep the zero in the defects in order to respect the dynamics
            defects[dq.shape[0]:, :] = tau - tau_id

    return DynamicsEvaluation(dxdt, defects)


def compute_kinova_terminal_forces(model, q_k):
    # markers positions in global and jacobians
    J1 = model.markersJacobian(q_k)[-2].to_array()
    J2 = model.markersJacobian(q_k)[-1].to_array()

    markers = model.markers(q_k)
    markers_1 = markers[-2].to_mx()
    markers_2 = markers[-1].to_mx()

    # tau = J^T * F => F = J^T \ tau, in static case
    Jtot_T = horzcat(transpose(J1), transpose(J2))
    Jtot_T_inv = inv(Jtot_T)
    tau = model.ligamentsJointTorque(q_k, MX.zeros(6, 1)).to_mx()

    end_effector_force = Jtot_T_inv @ tau

    force_1 = end_effector_force[0:3]  # force on first marker
    force_2 = end_effector_force[3:6]  # force on second marker

    spatial_vector_1 = biorbd.SpatialVector(vertcat(MX.zeros(3, 1), force_1))
    spatial_vector_2 = biorbd.SpatialVector(vertcat(MX.zeros(3, 1), force_2))

    spatial_vector_2_in_1 = transport_spatial_force(
        spatial_vector_2,
        current_application_point=markers_2,
        new_application_point=markers_1,
    )

    spatial_vector_tot_in_1 = biorbd.SpatialVector(
        spatial_vector_1.to_mx() + spatial_vector_2_in_1.to_mx()
    )

    spatial_vector_tot_in_0 = transport_spatial_force(
        spatial_vector_tot_in_1,
        current_application_point=markers_1,
        new_application_point=MX.zeros(3, 1),
    )

    return spatial_vector_tot_in_0
