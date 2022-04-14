from ezc3d import c3d
import numpy as np
from scipy.interpolate import interp1d
import biorbd


class C3dData:
    def __init__(self, file_path):
        self.c3d = c3d(file_path)
        self.marker_names = [
            "MAN",  # check 1
            "XYP",  # check 2
            "C7",  # check 3
            "T10",  # check 4
            "CLAV_SC",  # check 5
            "CLAV_AC",  # check 6
            # "SCAP_Cor",  # check 7
            "SCAP_IA",  # check 8
            # "SCAP_AA",  # check 9
            "SCAP_AC",  # check 10
            "SCAP_BACK",  # check 11
            "SCAP_FRONT",  # check 12
            "EPI_lat",  # check 13
            "EPI_med",  # check 14
            "DELT",  # check 15
            "ARM",  # check 16
            "ULNA",  # check 17
            "ELB",  # check 18
            "RADIUS",  # check 19
            "SEML",  # check 20
            "MET2",  # check 21
            "MET5",  # check 22
        ]

        self.trajectories = self.get_marker_trajectories(self.c3d, self.marker_names)

    @staticmethod
    def get_marker_trajectories(loaded_c3d, marker_names):
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
    def __init__(self, model, c3d_file, q_file, qdot_file):
        def load_txt_file(file_path, size):
            data_tp = np.loadtxt(file_path)
            # # nb_frame = int(len(data_tp) / size)
            # nb_frame = data_tp.shape[1]
            # out = np.zeros((size, nb_frame))
            # for n in range(nb_frame):
            #     out[:, n] = data_tp[n * size: n * size + size]
            return data_tp

        self.model = model
        self.nb_q = model.nbQ()
        self.nb_qdot = model.nbQdot()
        self.nb_markers = model.nbMarkers()

        # files path
        self.c3d_data = C3dData(c3d_file)
        self.q = load_txt_file(q_file, self.nb_q)
        self.qdot = load_txt_file(qdot_file, self.nb_qdot)

    def dispatch_data(self, data, nb_shooting: list, phase_time: list):
        """
        divide and adjust data dimensions to match number of shooting point for each phase
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
            f = interp1d(t_init, x, kind="cubic")
            out.append(f(t_node))
        return out

    def get_marker_ref(self, nb_shooting: list, phase_time: list, type: str) -> list:
        # todo: add an argument if "all" all markers and if "hand" only markers of hand if "MET5" only MET5

        return self.dispatch_data(self.c3d_data.trajectories, nb_shooting=nb_shooting, phase_time=phase_time)

    def get_experimental_data(self, number_shooting_points, phase_time):
        q_ref = self.dispatch_data(data=self.q, nb_shooting=number_shooting_points, phase_time=phase_time)
        qdot_ref = self.dispatch_data(data=self.qdot, nb_shooting=number_shooting_points, phase_time=phase_time)
        markers_ref = self.dispatch_data(
            data=self.c3d_data.trajectories, nb_shooting=number_shooting_points, phase_time=phase_time
        )
        return q_ref, qdot_ref, markers_ref
