import ezc3d
from pyomeca import Markers
import numpy as np
from .enums import Tasks


class LoadTask:
    def __init__(self,
                 task: Tasks,
                 model: "Models",
                 ):
        self.task = task
        self.c3d_path = task.value
        self.data_path = self.c3d_path.removesuffix(self.c3d_path.split("/")[-1])
        file_path = (
                self.data_path + model.name + "_" + self.c3d_path.split("/")[-1].removesuffix(
            ".c3d")
        )
        self.q_file_path = file_path + "_q.txt"
        self.qdot_file_path = file_path + "_qdot.txt"


class LoadEvent:
    def __init__(self,
                 task: Tasks,
                 marker_list: list[str],
                 ):
        self.task = task
        self.c3d_path = task.value
        self.c3d = ezc3d.c3d(self.c3d_path)
        self.marker_list = marker_list
        self.start_event_idx = 1 if self.task == Tasks.EAT else 0
        self.end_event_idx = 2 if self.task == Tasks.EAT else 1

    def get_time(self, idx: int) -> np.ndarray:
        """
        find the time corresponding to the event

        Parameters
        ---------
        idx: int
            index number of the event

        Returns
        --------
        event_values: ndarray
            array with the time value in seconds

        """
        event_time = self.c3d["parameters"]["EVENT"]["TIMES"]["value"][1][idx]
        return np.array(event_time)

    def get_frame(self, idx: int) -> np.ndarray:
        """
        find the frame corresponding to the event

        Parameters
        ---------
        idx: int
            index number of the event

        Returns
        --------
        event_values: ndarray
            array with the frame number

        """
        frame_rate = self.c3d["parameters"]["TRIAL"]["CAMERA_RATE"]["value"][0]
        frame = round(self.get_time(idx) * frame_rate)
        start_frame = self.c3d["parameters"]["TRIAL"]["ACTUAL_START_FIELD"]["value"][0]
        event_frame = frame - start_frame
        return np.array(event_frame)

    def get_markers(self, idx: int) -> np.ndarray:
        """
        find the position of each marker during an event

        Parameters
        ---------
        idx: int
            index number of the event

        Returns
        --------
        event_values: ndarray
            array with the position along three axes of each marker in millimeters

        """

        markers = Markers.from_c3d(self.c3d_path, usecols=self.marker_list, prefix_delimiter=":").to_numpy()
        event_markers = markers[:3, :, self.get_frame(idx)]

        return event_markers

    def get_event(self, idx: int) -> dict:
        """
        find the time, the frame and the position of each marker during an event

        Parameters
        ---------
        idx: int
            index number of the event

        Returns
        --------
        event_values: dict
            dictionary containing the time, the frame and the positions of each marker for the event corresponding to
            the given index

        """

        event_values = {"time": self.get_time(idx), "frame": self.get_frame(idx), "markers": self.get_markers(idx)}

        return event_values

    def start_frame(self):
        return self.get_frame(self.start_event_idx)

    def end_frame(self):
        return self.get_frame(self.end_event_idx)

    def get_start_end_time(self):
        """ Returns the start and end of the task """
        return self.get_time(self.start_event_idx), self.get_time(self.end_event_idx)

    def phase_time(self) -> np.ndarray:
        """ Returns the duration of the task """
        start, end = self.get_start_end_time()
        return np.float64(end - start)
