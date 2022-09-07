import pickle
import os
from pathlib import Path
from bioptim import OptimalControlProgram
import numpy as np

def custom_load(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
        sol = dict(
            states=data['states'], controls=data['controls'], parameters=data['parameters'],
            iterations=data['iterations'],
            cost=data['cost'], detailed_cost=data['detailed_cost'],
            real_time_to_optimize=data['real_time_to_optimize'], param_scaling = ['param_scaling'],
            n_shooting = data["n_shooting"]
        )

        return sol



def from_bo_to_pickle(file):

    #  check if it.s .bo
    _, ext = os.path.splitext(file)
    if ext == ".bo":
        with open(file, 'rb') as f:
            a = pickle.load(f)
            ocp = OptimalControlProgram(**a["ocp_initializer"])
        _, sol = OptimalControlProgram.load(file)
        sol.print_cost()

        data = dict(
            states=a['sol'].states, controls=a['sol'].controls, parameters=a['sol'].parameters, iterations=a['sol'].iterations,
            cost=np.array(a['sol'].cost)[0][0], detailed_cost=sol.detailed_cost, real_time_to_optimize = a['sol'].real_time_to_optimize,
            param_scaling = [nlp.parameters.scaling for nlp in ocp.nlp], n_shooting=a["ocp_initializer"]["n_shooting"]
        )

    elif ext != ".bo":
        raise RuntimeError(f"Incorrect extension({ext}), it should be (.bo) or (.bob) if you use from_bo_to_pickle.")


    #fichier pickle :

    _, ext = os.path.splitext(file)

    pickle_file = _ + ".pckl"
    with open(pickle_file, "wb") as file:
        pickle.dump(data, file)
