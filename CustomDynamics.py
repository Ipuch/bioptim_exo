import biorbd_casadi as biorbd
import numpy as np
from casadi import MX, vertcat
from numpy import ndarray

from bioptim import (
    PenaltyNode,
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    DynamicsList,
    Bounds,
    BoundsList,
    QAndQDotBounds,
    InitialGuess,
    InitialGuessList,
    ObjectiveFcn,
    Objective,
    ObjectiveList,
    OdeSolver,
    Node,
    ConstraintFcn,
    ConstraintList,
    DynamicsFunctions,
    NonLinearProgram,
    ConfigureProblem,
)
import spring


def dispatch_q_qdot_tau(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact,
                        springs: list) -> tuple:
    """
     Forward dynamics driven by joint torques with springs.

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
    with_contact
        bool
    springs
        list
    Returns
    ----------
    MX.sym
        The derivative of the states
    """

    DynamicsFunctions.apply_parameters(parameters, nlp)
    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

    nb_tau = len(nlp.controls["tau"].index)
    tau_p = MX.zeros(nb_tau)

    # Spring parameters
    for ii in range(nb_tau):
        if springs[ii]:
            tau_p[ii] = springs[ii].torque(q[ii])
    return q, qdot, tau + tau_p


def custom_contact(states, controls, parameters, nlp, with_contact: bool, springs: list) -> tuple:
    """
     Forward dynamics driven by joint torques with springs.

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
    with_contact
    springs
    Returns
    ----------
    MX.sym
        The derivative of the states
    """

    q, qdot, tau = dispatch_q_qdot_tau(states, controls, parameters, nlp, with_contact, springs)

    contact = nlp.model.ContactForcesFromForwardDynamicsConstraintsDirect(q, qdot, tau).to_mx()

    return contact


def custom_dynamic(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact: bool,
                   springs: list) -> tuple:
    """
     Forward dynamics driven by joint torques with springs.

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
    with_contact
    springs
    Returns
    ----------
    MX.sym
        The derivative of the states
    """

    q, qdot, tau = dispatch_q_qdot_tau(states, controls, parameters, nlp, with_contact, springs)

    dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
    ddq = nlp.model.ForwardDynamicsConstraintsDirect(q, qdot, tau).to_mx()

    return dq, ddq


def dynamic_config(ocp: OptimalControlProgram, nlp: NonLinearProgram, with_contact: bool, springs: list):
    ConfigureProblem.configure_q(nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_tau(nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamic,
                                                 with_contact=with_contact, springs=springs)

    if with_contact:
        ConfigureProblem.configure_contact_function(ocp, nlp, custom_contact,
                                                    with_contact=with_contact, springs=springs)


def plot_passive_torque(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, springs,idx) -> ndarray:
    """
    Create a used defined plot function with extra_parameters

    Parameters
    ----------

    Returns
    -------
    The value to plot
    """
    # q, qdot, tau = dispatch_q_qdot_tau(states, controls, parameters, nlp, True, springs)
    q = states[nlp.states.elements[0].index]
    tau_p = np.zeros((len(idx), np.size(states, 1)))
    for ii, id in enumerate(idx):
        if springs[id]:
            tau_p[ii, :] = springs[id].torque(q[id])
        # tau_p = np.expand_dims(tau_p, axis=1)
    return np.tile(tau_p, 1)
