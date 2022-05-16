import ezc3d


def event(c3d_path: str, idx: int) -> dict:
    """
    find the time and the position of each marker during an event

    Parameters:
    ---------
    c3d_path: str
        path to the c3d file containing the generalized coordinates obtained via motion capture
    idx: int
        index number of the event

    Returns
    --------
    event_values : dict
        dictionary containing the time and the positions of each marker for the event corresponding to the given index

    """

    c3d = ezc3d.c3d(c3d_path)
    event_values = {}
    event_time = c3d["parameters"]["EVENT"]["TIMES"]["value"][1][idx]
    event_values["time"] = event_time
    frame_rate = c3d["parameters"]["TRIAL"]["CAMERA_RATE"]["value"][0]
    frame = round(event_time * frame_rate)
    points = c3d["data"]["points"]
    event_points = []
    for variable in range(len(points) - 1):
        marker_points = []
        for marker in range(len(points[variable])):
            marker_points.append(points[variable][marker][0])
        event_points.append(marker_points)
    event_values["points"] = event_points

    return event_values


path = "../event/F0_tete_05.c3d"
print(event(path, 0))
