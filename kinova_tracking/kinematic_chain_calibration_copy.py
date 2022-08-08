from typing import Union
from enum import Enum

import numpy as np

import biorbd


class ObjectivesFunctions(Enum):
    ALL_OBJECTIVES = "all objectives"
    ALL_OBJECTIVES_WITHOUT_FINAL_ROTMAT = "all objectives without final rotmat"


class KinematicChainCalibration:
    """


    Examples
    ---------
    kcc = KinematicChainCalibration()
    kcc.solve()
    kkc.results()
    """
    def __init__(self,
                 model: biorbd.Model,
                 markers_model: list[str],
                 markers: np.array, # [3 x nb_markers, x nb_frames]
                 closed_loop_markers: list[str],
                 tracked_markers: list[str],
                 parameter_dofs: list[str],
                 kinematic_dofs: list[str],
                 objectives_functions: ObjectivesFunctions,
                 weights: Union[list[float], np.ndarray],
                 q_ik_initial_guess: np.ndarray, # [n_dof x n_frames]
                 nb_frames_ik_step: int = None,
                 nb_frames_param_step: int = None,
                 randomize_param_step_frames: bool= True,
                 use_analytical_jacobians: bool = False,
                 ):
        self.model = model

        # check if markers_model are in model
        # otherwise raise
        for marker in markers_model:
            if marker not in [i.to_string() for i in model.markerNames()]:
                raise ValueError(f"The following marker is not in markers_model:{marker}")
            else :
                self.markers_model = markers_model

        # check if markers model and makers have the same size
        # otherwise raise
        if markers.shape() == markers_model.shape():
            self.markers = markers
        else:
            raise ValueError(f"markers and markers model must have same shape, markers shape is { markers.shape()},"
                             f" and markers_model shape is { markers_model.shape()}")
        self.closed_loop_markers = closed_loop_markers
        self.tracked_markers = tracked_markers
        self.parameter_dofs = parameter_dofs
        self.kinematic_dofs = kinematic_dofs
        # self.objectives_function

        # number of wieghts has to be checked
        # raise Error if not the right number
        self.weights = weights

        # check if q_ik_initial_guess has the right size
        self.q_ik_initial_guess = q_ik_initial_guess
        self.nb_frames_ik_step = nb_frames_ik_step
        self.nb_frames_param_step = nb_frames_param_step
        self.randomize_param_step_frames = randomize_param_step_frames
        self.use_analytical_jacobians = use_analytical_jacobians


    # if nb_frames_ik_step> markers.shape[2]:
    # raise error
    # self.nb_frame_ik_step = markers.shape[2] if nb_frame_ik_step is None else nb_frames_ik_step
    #
    # # check for randomize and nb frames.
    #
    # def solve(self, tolerance, use_analytical_jacobians:bool=True, objectives_functions: ObjectivesFunctions)

# kcc = KinematicChainCalibration()
# kcc.solve()
# kkc.results()