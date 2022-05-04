import numpy as np
import biorbd
import ezc3d
import matplotlib.pyplot as plt

try:
    import bioviz

    biorbd_viz_found = True
except ModuleNotFoundError:
    biorbd_viz_found = False


def get_unit_division_factor(c3d_file: ezc3d) -> int:
    """
    Allow the users to get the length units of a c3d file

    Parameters
    ----------
    c3d_file: ezc3d
        c3d file converted into an ezc3d object

    Returns
    -------
    The division factor of length units
    """
    factor_str = c3d_file["parameters"]["POINT"]["UNITS"]["value"][0]
    if factor_str == "mm":
        factor = 1000
    elif factor_str == "m":
        factor = 1
    else:
        raise NotImplementedError("This is not implemented for this unit")

    return factor


def get_c3d_frequencies(c3d_file: ezc3d) -> float:
    """
    Allow the users to get the length units of c3d

    Parameters
    ----------
    c3d_file: ezc3d
        c3d file converted into an ezc3d object

    Returns
    -------
    The frequencies
    """
    return c3d_file["parameters"]["POINT"]["RATE"]["value"][0]


def plot_dof(q_old: np.ndarray, q: np.ndarray, biorbd_model: biorbd.Model):
    """
    Allow the users to get the length units of c3d

    Parameters
    ----------
    q_old: np.ndarray
        The list of q before applying the offset on some dof
    q: np.ndarray
        The list of q after applying the offset on some dof
    biorbd_model: biorbd.Model
        The biorbd model
    Returns
    -------
    The frequencies
    """
    count = 0
    range_max = []
    range_min = []
    for i in range(biorbd_model.nbSegment()):  # we excluded the 3 first dof which correspond to 3 translation
        print(i)
        if biorbd_model.segment(i).nbDof() > 0:
            print(biorbd_model.segment(i))
            for j in range(biorbd_model.segment(i).nbDof()):
                range_min.append(biorbd_model.segment(i).QRanges()[j].min())
                range_max.append(biorbd_model.segment(i).QRanges()[j].max())

    plt.figure()
    for h in range(biorbd_model.nbQdot()):
        plt.subplot(int(np.sqrt(biorbd_model.nbQdot())), int(np.sqrt(biorbd_model.nbQdot())) + 1, h + 1)
        plt.title(f"{biorbd_model.nameDof()[h].to_string()}")
        plt.plot(q_old[h, :], label=f"{str(h)} old")
        plt.plot(q[h, :], label=str(h))
        plt.plot([range_min[h]] * q.shape[1])
        plt.plot([range_max[h]] * q.shape[1])
        plt.legend()
    plt.show()


def get_segment_and_dof_id_from_global_dof(biorbd_model: biorbd.Model, global_dof: int):
    """
    Allow the users to get the segment id which correspond to a dof of the model and the id of this dof in the segment

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model
    global_dof: int
        The global id of the dof in the model

    Returns
    -------
    seg_id: int
        The id of the segment which correspond to the dof
    count_dof: int
         The dof id in this segment
    """
    for j, seg in enumerate(biorbd_model.segments()):
        complete_seg_name = biorbd_model.nameDof()[global_dof].to_string()  # We get "Segment_Name_DofName"
        seg_name = complete_seg_name.replace("_RotX", "")  # We remove "_DofName"
        seg_name = seg_name.replace("_RotY", "")
        seg_name = seg_name.replace("_RotZ", "")
        seg_name = seg_name.replace("_TransX", "")
        seg_name = seg_name.replace("_TransY", "")
        seg_name = seg_name.replace("_TransZ", "")

        if seg.name().to_string() == seg_name:
            seg_name_new = seg_name
            seg_id = j

    dof = biorbd_model.nameDof()[global_dof].to_string().replace(seg_name_new, "")
    dof = dof.replace("_", "")  # we remove the _ "_DofName"
    count_dof = 0
    while biorbd_model.segment(seg_id).nameDof(count_dof).to_string() != dof:
        count_dof += 1

    return seg_id, count_dof


def apply_offset(biorbd_model: biorbd.Model, q: np.ndarray, dof_list: list, offset: float):
    """
    Allow the users to get the generalized coordinate in between predefined ranges or closer.
    It fixes the output of the Kalman Filter.

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The biorbd model
    q: np.ndarray
        The array of q
    dof_list: list
        The list of global number of the dof
    offset: float
        The minimum offset that we wanted to apply on the dof

    Returns
    -------
    """
    q_with_offset = q.copy()
    for dof_nb in dof_list:
        seg_id, dof_id = get_segment_and_dof_id_from_global_dof(biorbd_model, dof_nb)
        range_min = biorbd_model.segment(seg_id).QRanges()[dof_id].min()
        range_max = biorbd_model.segment(seg_id).QRanges()[dof_id].max()
        value = q_with_offset[dof_nb, :].mean()
        delta_range_max = value - range_max
        delta_range_min = value - range_min

        compteur = 0

        if delta_range_min > 0 and delta_range_max > 0:  # if the value is above range max
            while (delta_range_min > 0 and delta_range_max > 0) and compteur < 100:
                compteur += 1
                q_with_offset[dof_nb, :] = q_with_offset[dof_nb, :] - offset
                value = q_with_offset[dof_nb, :].mean()
                delta_range_max = value - range_max
                delta_range_min = value - range_min

        elif delta_range_min < 0 and delta_range_max < 0:  # if the value is below range min
            while (delta_range_min < 0 and delta_range_max < 0) and compteur < 100:
                compteur += 1
                q_with_offset[dof_nb, :] = q_with_offset[dof_nb, :] + offset
                value = q_with_offset[dof_nb, :].mean()
                delta_range_max = value - range_max
                delta_range_min = value - range_min
        else:
            print("Nothing has to be done.")

    return q_with_offset
