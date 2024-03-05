import biorbd_casadi as biorbd

import numpy as np
from data.enums import Tasks, TasksKinova
from models.enums import Models

import data.load_events as load_events
from models.biorbd_model import NewModel, KinovaModel
import tracking.load_experimental_data as load_experimental_data


task = TasksKinova.DRINK
task_files = load_events.LoadTask(task=task, model=Models.WU_INVERSE_KINEMATICS)
model_path_upperlimb = Models.WU_INVERSE_KINEMATICS.value
model_mx_upperlimb = biorbd.Model(model_path_upperlimb)

data = load_experimental_data.LoadData(
    model=model_mx_upperlimb,
    c3d_file=task_files.c3d_path,
    q_file=task_files.q_file_path,
    qdot_file=task_files.qdot_file_path,
)

kinova_model = KinovaModel(model=Models.KINOVA_RIGHT_SLIDE_POLAR_BASE_WITH_VARIABLES)
marker_attachment_name = "Table:Table5"
index_marker_attachment = data.c3d_data.c3d["parameters"]["POINT"]["LABELS"]["value"].index(marker_attachment_name)
attachment_point_location = np.mean(data.c3d_data.c3d["data"]["points"][:, index_marker_attachment, :], axis=1)[0:3]
kinova_model.add_header(
    model_template=Models.KINOVA_RIGHT_SLIDE_POLAR_BASE_TEMPLATE,
    attachment_point_location=attachment_point_location / 1000,
    offset=np.array([0, -0.02, 0.]),
)
model_path_kinova = kinova_model.model_path
model_mx_kinova = biorbd.Model(model_path_kinova)
print(model_mx_kinova.marker(0, False).isAnatomical())

from biorbd_casadi import NodeSegment
NodeSegment(1,2,3)