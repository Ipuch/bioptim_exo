import bioviz
import numpy as np
import biorbd

# decimal option
np.set_printoptions(precision=10)

model_name = "models/KINOVA_merge_6dof.bioMod"

biorbd_model = biorbd.Model(model_name)

# Now the model is loaded as a biorbd object

# Q = np.ones(biorbd_model.nbQ())
Q = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.0398,
        -0.0752,
        0.0291,
        2.8037,
        -1.6032,
        -0.1205,
    ]
)
# Q is the vector that we get after manually placed the support into the simulation

print(biorbd_model.nbQ())

# In our configuration, ulna is segment 30 and support is segment 40 ( it can change if KINOVA_merge.bioMod is
# modified) So, we want to find the rototrans (aka homogeneous transform) of ulna and the support from local to global
# coordinates.
# As a reminder, here is the formula to transform a point from one frame (segment) to another (world).
# Position_in_world = Rototrans_matrix_world_segment * Position_in_segment

Rototrans_matrix_world_ulna = biorbd_model.globalJCS(Q, 30).to_array()
Rototrans_matrix_world_support = biorbd_model.globalJCS(Q, 40).to_array()

# We want to find the transformation matrix btw support and ulna
# Rototrans_matrix_ulna_support = Rototrans_matrix_ulna_world * Rototrans_matrix_world_support

# Rototrans_matrix_ulna_world is the inverse matrix of Rototrans_matrix_world_ulna, we use .transpose()
# .transpose() on the Rototrans object which actually inverses the rototranslation matrix.
Rototrans_matrix_ulna_world = biorbd_model.globalJCS(Q, 30).transpose().to_array()
# Be aware that Rototrans_matrix_world_ulna.transpose() would give the wrong matrix

# Finally
Rototrans_matrix_ulna_support = np.matmul(Rototrans_matrix_ulna_world, Rototrans_matrix_world_support)

print(Rototrans_matrix_ulna_support)

# We have Rototrans_matrix_ulna_support, so, we are now able to transform the coordinates of any point in the support
# frame to the ulna frame with the following formula:
# Position_in_ulna = Rototrans_matrix_ulna_support * Position_in_support
