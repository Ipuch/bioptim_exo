from ezc3d import c3d
import numpy as np
from scipy.interpolate import interp1d
import biorbd


class C3dData:
    """
    The base class for managing c3d file

    Attributes
    ----------
    c3d: ezc3d.c3d
        The c3d file
    marker_names: list
        The list of all marker names in the biorbd model.
    trajectories: ndarray
        The position of the markers
    Methods
    -------
    get_marker_trajectories(loaded_c3d: c3d, marker_names: list) -> np.ndarray
        Get markers trajectories
    get_indices(self)
        Get the indices of start and end
    get_final_time(self)
        Get the final time of c3d
    """

    def __init__(self, file_path: str, biorbd_model: biorbd.Model):
        self.c3d = c3d(file_path)
        self.marker_names = [biorbd_model.markerNames()[i].to_string() for i in range(len(biorbd_model.markerNames()))]
        self.trajectories = self.get_marker_trajectories(self.c3d, self.marker_names)

    @staticmethod
    def get_marker_trajectories(loaded_c3d: c3d, marker_names: list) -> np.ndarray:
        """
        get markers trajectories

        """

        # LOAD C3D FILE
        points = loaded_c3d["data"]["points"]
        labels_markers = loaded_c3d["parameters"]["POINT"]["LABELS"]["value"]

        # GET THE MARKERS POSITION (X, Y, Z) AT EACH POINT
        markers = np.zeros((3, len(marker_names), len(points[0, 0, :])))

        # pelvis markers
        for i, name in enumerate(marker_names):
            markers[:, i, :] = points[:3, labels_markers.index(name), :] * 1e-3
        return markers

    def get_indices(self):
        idx_start = 0 + 1
        idx_stop = len(self.trajectories[0, 0, :])
        return [idx_start, idx_stop]

    def get_final_time(self):
        """
        find phase duration
        """
        # todo: plz shrink the function
        freq = self.c3d["parameters"]["POINT"]["RATE"]["value"][0]
        index = self.get_indices()
        phase_time = []
        for i in range(len(index) - 1):
            phase_time.append((1 / freq * (index[i + 1] - index[i] + 1)))
        return phase_time[0]


class LoadData:
    """
    The base class for managing c3d data

    Attributes
    ----------
    model: biorbd.Model
        The biorbd Model
    nb_q: int
        The number of generalized coordinates in the model
    nb_qdot: int
        The number of generalized velocities in the model
    nb_markers: int
        The number of markers in the model
    c3d_data: C3dData
        The data from c3d
    q: numpy.ndarray
        The array of all values of the q
    qdot: numpy.ndarray
        The array of all values of the qdot
    Methods
    -------
    dispatch_data(self, data, nb_shooting: list, phase_time: list)
        divide and adjust data dimensions to match number of shooting point for each phase
    get_marker_ref(self, nb_shooting: list, phase_time: list, type: str) -> list:
        Give the position of the markers from the c3d
    get_experimental_data(self, number_shooting_points, phase_time, with_floating_base: bool)
        Give the values of q, qdot and the position of the markers from the c3d
        """
    def __init__(self, model: biorbd.Model, c3d_file: str, q_file: str, qdot_file: str):
        def load_txt_file(file_path: str):
            data_tp = np.loadtxt(file_path)
            return data_tp

        self.model = model
        self.nb_q = model.nbQ()
        self.nb_qdot = model.nbQdot()
        self.nb_markers = model.nbMarkers()

        # files path
        self.c3d_data = C3dData(c3d_file, model)
        self.q = load_txt_file(q_file)
        self.qdot = load_txt_file(qdot_file)

    def dispatch_data(self, data: np.ndarray, nb_shooting: list, phase_time: list):
        """
        divide and adjust data dimensions to match number of shooting point for each phase

        Parameters:
        data: np.ndarray
            The data we want to adjust
        nb_shooting: list
            The list of nb_shooting for each phases
        phase_time: list
            The list of duration for each phases4

        Returns:
            Data adjusted
        """

        index = self.c3d_data.get_indices()
        out = []
        for i in range(len(nb_shooting)):
            if len(data.shape) == 3:
                x = data[:, :, index[i] : index[i + 1] + 1]
            else:
                x = data[:, index[i] : index[i + 1] + 1]
            t_init = np.linspace(0, phase_time[i], (index[i + 1] - index[i]))
            t_node = np.linspace(0, phase_time[i], nb_shooting[i] + 1)
            f = interp1d(t_init, x, kind="linear")
            out.append(f(t_node))
        return out

    def get_marker_ref(self, nb_shooting: list, phase_time: list, type: str) -> list:
        """
        divide and adjust the dimensions to match number of shooting point for each phase

        Parameters:
        nb_shooting: list
            The list of nb_shooting for each phases
        phase_time: list
            The list of duration for each phases4
        type: str

        Returns:
            The array of marker's position adjusted
        """
        # todo: add an argument if "all" all markers and if "hand" only markers of hand if "MET5" only MET5

        return self.dispatch_data(self.c3d_data.trajectories, nb_shooting=nb_shooting, phase_time=phase_time)

    def get_experimental_data(self, number_shooting_points, phase_time, with_floating_base: bool):
        """
        Give all the data from c3d file

        Parameters:
        number_shooting_points: list
            The list of nb_shooting for each phases
        phase_time: list
            The list of duration for each phases4
        with_floating_base: bool
            True if there is a floating base in the biorbd model

        Returns:
            The values of q, qdot and the positions of the markers from the c3d
        """
        q_ref = self.dispatch_data(data=self.q, nb_shooting=number_shooting_points, phase_time=phase_time)
        qdot_ref = self.dispatch_data(data=self.qdot, nb_shooting=number_shooting_points, phase_time=phase_time)
        markers_ref = self.dispatch_data(
            data=self.c3d_data.trajectories, nb_shooting=number_shooting_points, phase_time=phase_time
        )
        q_ref[0] = q_ref[0][6:] if not with_floating_base else q_ref[0]
        qdot_ref[0] = qdot_ref[0][6:] if not with_floating_base else qdot_ref[0]

        return q_ref, qdot_ref, markers_ref
