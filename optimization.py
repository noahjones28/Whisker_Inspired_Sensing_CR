import numpy as np
from pyslsqp import optimize
from pyslsqp.postprocessing import print_dict_as_table

class MyOptimization:
    def __init__(self, elasticaInstance=None, model=None):
        pass
    def objective(self, x):
        #Design Variables
        F = x[0]
        s = x[1]
        # Objective function
        Fx_sim, MB_sim = self.elastica.getProximalValues(F=F, s=s, tendon_displacement=self.tendon_displacement, surrogate_model=self.surrogate_model)
        obj = (Fx_sim - self.Fx_target)**2 + (MB_sim - self.MB_target)**2
        return obj

    def solve(self, axial_force, bending_moment_mag, tendon_displacement, x0, elastica, surrogate_model=None):
        # Initial Condition (F0, s0)
        self.x0 = x0
        # Target Values
        self.Fx_target = axial_force 
        self.MB_target = bending_moment_mag
        # Static Parameters
        self.tendon_displacement = tendon_displacement
        # Class Objects
        self.elastica = elastica
        self.surrogate_model = surrogate_model
        # optimize returns a dictionary that contains the results from optimization
        results = optimize(x0, obj=self.objective, acc=1e-6, finite_diff_abs_step=0.01)
        print_dict_as_table(results)