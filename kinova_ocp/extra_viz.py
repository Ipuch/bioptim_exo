"""
Example script for animating markers
"""
from enum import Enum

from bioviz import VtkModel, VtkWindow, Viz
import numpy as np
from pathlib import Path
from models.merge_biomod import merge_biomod


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


def custom_animate(model_kinova_path, model_path_upperlimb, q_k, q_upper_limb, all_rt=None):
    parent_path = Path(model_kinova_path).parent
    output_path = str(parent_path / "merged.bioMod")
    merge_biomod(model_kinova_path, model_path_upperlimb, output_path)
    q_tot = np.concatenate((q_k, q_upper_limb), axis=0)
    n_frames = q_tot.shape[1]

    viz = Viz(output_path, show_floor=False, show_global_ref_frame=False, show_muscles=False)
    if all_rt is not None:
        vtkObject = VtkFrameModel(viz.vtk_window, normalized=True)

    i = 1
    while viz.vtk_window.is_active:
        # Update the markers
        if all_rt is not None:
            vtkObject.update_frame(rt=all_rt[:, :, i])
        viz.set_q(q_tot[:, i])
        i = (i + 1) % n_frames
