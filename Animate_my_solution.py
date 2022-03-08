from bioptim import OptimalControlProgram

ocp, sol = OptimalControlProgram.load('Kinova.bo')
sol.print()
sol.animate()
# sol.graphs()