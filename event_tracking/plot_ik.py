import matplotlib.pyplot as plt
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import biorbd

q = np.loadtxt("F0_tete_05_q.txt")
m = biorbd.Model(
    "/home/lim/Documents/Stage_Thasaarah/bioptim_exo/models/wu_converted_definitif_inverse_kinematics.bioMod"
)
names = [i.to_string() for i in m.nameDof()]
fig = make_subplots(rows=5, cols=4, subplot_titles=names, shared_yaxes=True)
j = 0
for i in range(q.shape[0]):
    ii = i - j * 4
    fig.add_trace(go.Scatter(y=q[i]), row=j + 1, col=ii + 1)
    if ii == 3:
        j += 1

fig.show()

# plt.figure()
# plt.plot(q.T)
# plt.legend([f"{i}" for i in range(q.shape[0])])
# plt.show()
