"""
Example script for animating markers
"""
from enum import Enum

from bioviz import VtkModel, VtkWindow
import numpy as np


class NaturalVectorColors(Enum):
    """Colors for the vectors."""

    # light red
    U = (255, 20, 0)
    # light green
    V = (0, 255, 20)
    # light blue
    W = (20, 0, 255)


class VectorColors(Enum):
    """Colors for the vectors."""

    # red
    X = (255, 0, 0)
    # green
    Y = (0, 255, 0)
    # blue
    Z = (0, 0, 255)


class VtkFrameModel:
    def __init__(
        self,
        vtk_window: VtkWindow,
        normalized: bool = False,
    ):
        self.vtk_window = vtk_window
        self.three_vectors = [self.vtk_vector_model(color) for color in VectorColors]
        self.normalized = normalized

    def vtk_vector_model(self, color: NaturalVectorColors):
        vtkModelReal = VtkModel(
            self.vtk_window,
            force_color=color.value,
            force_opacity=1.0,
        )
        return vtkModelReal

    def update_frame(self, rt: np.ndarray):
        # in bioviz vectors are displayed through "forces"
        u_vector = np.concatenate((rt[:3, 3], rt[:3, 0] + rt[:3, 3]))
        v_vector = np.concatenate((rt[:3, 3], rt[:3, 1] + rt[:3, 3]))
        w_vector = np.concatenate((rt[:3, 3], rt[:3, 2] + rt[:3, 3]))

        self.three_vectors[0].update_force(
            segment_jcs=[np.identity(4)],
            all_forces=u_vector[np.newaxis, :, np.newaxis],
            max_forces=[1],
            normalization_ratio=0.22,
        )
        self.three_vectors[1].update_force(
            segment_jcs=[np.identity(4)],
            all_forces=v_vector[np.newaxis, :, np.newaxis],
            max_forces=[1],
            normalization_ratio=0.22,
        )
        self.three_vectors[2].update_force(
            segment_jcs=[np.identity(4)],
            all_forces=w_vector[np.newaxis, :, np.newaxis],
            max_forces=[1],
            normalization_ratio=0.22,
        )

