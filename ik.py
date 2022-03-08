import biorbd
import bioviz
from bioptim import ObjectiveFcn, BiMapping
from scipy import optimize
import numpy as np

# model_name = "KINOVA_arm_deprecated.bioMod"
model_path = "/home/puchaud/Projets_Python/My_bioptim_examples/Kinova_arm/KINOVA_arm_reverse.bioMod"

m = biorbd.Model(model_path)
bound_min = []
bound_max = []
for i in range(m.nbSegment()):
    seg = m.segment(i)
    for r in seg.QRanges():
        bound_min.append(r.min())
        bound_max.append(r.max())
bounds = (bound_min, bound_max)


def objective_function(x, *args, **kwargs):
    markers = m.markers(x)
    out1 = np.linalg.norm(markers[0].to_array() - target)**2
    T = m.globalJCS(x, m.nbSegment()-1).to_array()
    out2 = T[2, 0]**2 + T[2, 1]**2 + T[0, 2]**2 + T[1, 2]**2 + (1 - T[2, 2])**2
    # print(out2)
    # print(out1)
    return out1+out2

target = np.zeros((1, 3))
# q0=np.zeros(m.nbQ())
q0=np.array((0.0, 0.0, 0.0, 0.0, -0.1709, 0.0515, -0.2892, 0.6695, 0.721, 0.0, 0.0, 0.0))
T = m.globalJCS(q0, m.nbSegment()-1).to_array()


pos = optimize.least_squares(objective_function, args=(m, target), x0=q0,
                             bounds=bounds, verbose=2, method='trf', jac='3-point', ftol=1e-15, gtol=1e-20)
print(pos)
print(f"Optimal q for the assistive arm at {target} is:\n{pos.x}\n"
      f"with cost function = {objective_function(pos.x)}")
print(m.globalJCS(q0, m.nbSegment()-1).to_array())
print(m.globalJCS(pos.x, m.nbSegment()-1).to_array())
# Verification
q = np.tile(pos.x, (10, 1)).T
# q = np.tile(q0, (10, 1)).T
import bioviz
biorbd_viz = bioviz.Viz(model_path)
biorbd_viz.load_movement(q)
biorbd_viz.exec()

# TODO: without six dof & add plausible bounds
