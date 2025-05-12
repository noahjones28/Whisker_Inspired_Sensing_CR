import numpy as np
from matplotlib import pyplot as plt
from smt.surrogate_models import KRG, RBF, KPLSK
from smt.sampling_methods import LHS
import elastica as ea
from elastica.modules import (
    BaseSystemCollection,
    Connections,
    Constraints,
    Forcing,
    CallBacks,
    Damping
)
from elastica.rod.cosserat_rod import CosseratRod
from elastica.boundary_conditions import OneEndFixedBC
from elastica.external_forces import EndpointForces
from elastica.dissipation import AnalyticalLinearDamper
from elastica.joint import FixedJoint
from elastica.callback_functions import CallBackBaseClass
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate
from elastica.external_forces import NoForces
from joblib import Parallel, delayed
from pyslsqp import optimize
from pyslsqp.postprocessing import print_dict_as_table
import glob, os
#from Animation.BeamAnimator import BeamAnimator

class ElasticaForceEstimation:
    def __init__(self, x, y, z, radii, youngs_modulus, poisson_ratio):
        self.x = x
        self.y = y
        self.z = z
        self.nodes =  np.column_stack([self.x, self.y, self.z])
        self.n_elem = 3 # numebr of elements that make up each rod
        self.n_nodes = self.nodes.shape[0]
        self.directions = (self.nodes[1:] - self.nodes[:-1]) / np.linalg.norm(self.nodes[1:] - self.nodes[:-1], axis=1)[:, None] # Compute unit direction vectors between consecutive nodes
        ref = np.array([0, 0, -1])
        self.normals = np.cross(self.directions, ref)
        diffs = self.nodes[1:] - self.nodes[:-1]             # Shape (n-1, 3)
        self.lengths = np.linalg.norm(diffs, axis=1)         # Shape (n-1,)
        self.total_length = np.sum(self.lengths)
        self.radii = radii
        self.areas = np.pi*self.radii**2
        self.density = 1240 # (kg m-3)
        self.nu = 6e-1 # Damping constant of the rod 0 to 1
        self.youngs_modulus = youngs_modulus
        self.poisson_ratio = poisson_ratio # 0.36
        self.shear_modulus = self.youngs_modulus / (2 * (1 + poisson_ratio))
        self.dx = self.total_length / self.n_nodes # Length of each rod segment (spatial resolution)
        self.dt = 2e-6 # should be very low like 1e-6 or you get instability
        self.timestepper = PositionVerlet() # Define time stepper
        self.final_time = 0.15 # time you want simualtion to cover
        self.total_steps = int(self.final_time / self.dt)
        self.rods = np.empty(self.n_nodes-1, dtype=object)
        self.rod_history = {} # Create an array to store a dictionary for each rod
        self.applied_force_index = None # By default
        for i in range(len(self.rods)):
            self.rod_history[i] = ea.defaultdict(list)
        # Options
        self.PLOT_FIGURE = True

    def run(self):
        self.sim = SystemSimulator() # Create new simulation each time
        for i in range(len(self.rods)):
            rod = self.createRod(i)
            self.addConstraints(rod)
            self.addDamping(rod)
            self.addForce(rod)
            self.addCallback(rod)
        # finalize system and run simulation
        self.sim.finalize()
        # Run Simulation
        integrate(self.timestepper, self.sim, self.final_time, self.total_steps)
        if self.PLOT_FIGURE:
            self.plotFig()
            
    def createRod(self, i):
        rod = CosseratRod.straight_rod(
            n_elements=self.n_elem, # number of elements
            start=self.nodes[i], # Starting position of first node in rod
            direction=self.directions[i], # Direction the rod extends
            normal=self.normals[i], # normal vector of rod
            base_length=self.lengths[i], # original length of rod (m)
            base_radius=self.radii[i], # original radius of rod (m)
            density=self.density, # density of rod (kg/m^3)
            youngs_modulus = self.youngs_modulus, # Elastic Modulus (Pa)
            shear_modulus = self.shear_modulus # Shear Modulus (Pa)
        )
        self.sim.append(rod)
        self.rods[i] = rod
        return rod

    def addConstraints(self, rod):
        index = np.where(self.rods == rod)[0][0]
        if index == 0:
            # Add fixed constraint if first rod 
            self.sim.constrain(rod).using(
                OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
            )
        elif index > 0:
            # Connect rods rigidly
            self.sim.connect(
                first_rod  = self.rods[index-1],
                second_rod = rod,
                first_connect_idx  = -1, # Connect to the last node of the first rod.
                second_connect_idx =  0  # Connect to first node of the second rod.
            ).using(
                FixedJoint,  # Type of connection between rods
                k  = 5e4,    # Spring constant of force holding rods together (F = k*x)
                nu = 5e-1,      # Energy dissipation of joint
                kt = 1e1     # Rotational stiffness of rod to avoid rods twisting
                )

    def addForce(self, rod):
        # Add tip force
        index = np.where(self.rods == rod)[0][0]
        cumsum = np.cumsum(self.lengths)
        # Find first index where cumulative sum exceeds n
        first_indices = np.where(cumsum >= self.s)[0]
        if len(first_indices) == 0:
            raise ValueError(f"Force location exceeds total length of beam!")
        else:
            first_index = first_indices[0]
        if index == first_index:
            self.applied_force_index = index
            self.sim.add_forcing_to(rod).using(
                DynamicPerpendicularEndpointForce,
                magnitude=self.F
            )

    def addCallback(self, rod):
        index = np.where(self.rods == rod)[0][0]
        # Add MyCallBack to SystemSimulator for each rod telling it how often to save data (step_skip)
        self.sim.collect_diagnostics(rod).using(
            MyCallBack, step_skip=1000, callback_params=self.rod_history[index]
)
    def addDamping(self, rod):
        self.sim.dampen(rod).using(
        AnalyticalLinearDamper,
        damping_constant = self.nu,
        time_step = self.dt,
)
    
    def getPosValues(self, time_index):
        x = np.zeros(self.n_nodes)
        y = np.zeros(self.n_nodes)
        z = np.zeros(self.n_nodes)
        for i in range(len(self.rods)):
            history = self.rod_history[i]
            x[i] = history["position"][time_index][0][0]
            y[i] = history["position"][time_index][1][0]
            z[i] = history["position"][time_index][2][0] 
            if i == len(self.rods)-1:
                x[i+1] = history["position"][time_index][0][-1]
                y[i+1] = history["position"][time_index][1][-1]
                z[i+1] = history["position"][time_index][2][-1]
        return x, y, z
    
    def getProximalValues(self, F, s, model=None):
        self.F = F
        self.s = s
        if model == None:
            # Run simulation
            self.run()
            # Obtain axial force
            f0 = self.rods[0].internal_forces[:, 0]   # 3D force vector at node 0
            t0 = self.rods[0].tangents[:, 0]          # 3D unit tangent vector at start
            axial_force = np.dot(f0, t0) # Scalar: component of force along the tangent 
            # Obtain torques
            torque_global = self.rods[0].internal_torques[:, 0]  # Get global torque vector at node 0 shape: (3,)
            My = torque_global[1] # Project global torque vector onto local y and z axes
            Mz = torque_global[0]
            bending_moment_mag = np.sqrt(My**2 + Mz**2) # Compute the bending moment magnitude (orthogonal to rod axis)
        elif model != None:
            x = np.array([[F, s]])
            axial_force, bending_moment_mag = model.predict(x)[0]
        return axial_force, bending_moment_mag
    
    def get_distal_values(self, Fx, MB, optimizer):
        x0 = np.array([0.2, self.total_length/2])
        optimizer.solve(Fx, MB, x0)

    def plotFig(self):
        x_initial, y_initial, z_initial = self.getPosValues(0)
        x_final, y_final, z_final = self.getPosValues(-1)
        fig = plt.figure(figsize=(16, 8))
        # Top-left: 3D Beam Deflection
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax1.plot3D(x_initial, y_initial, z_initial, label='Initial Beam', color='black', linestyle='--')
        ax1.plot3D(x_final, y_final, z_final, label='Deflected Beam', color='blue')
        # get last force element
        force = self.rod_history[self.applied_force_index]["applied_force"][-1]
        ax1.quiver(x_final[self.applied_force_index], y_final[self.applied_force_index], z_final[-1], force[0], force[1], force[2],
               length=0.1, color='red', normalize=False, label='Applied Force')
        ax1.set_title('Deflection at each node 3D')
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax1.set_zlabel('z (m)')
        ax1.set_ylim(-self.total_length*1.5/2, self.total_length*1.5/2)
        ax1.set_xlim(0, self.total_length*1.5)
        ax1.set_zlim(-self.total_length*1.5/2, self.total_length*1.5/2)
        ax1.legend()
        ax1.set_aspect('equal')
        # Top-right: 2D Beam Deflection
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(x_initial, y_initial, label='Initial Beam', color='black', linestyle='--')
        ax2.plot(x_final, y_final, label='Deflected Beam', color='blue')
        ax2.quiver(x_final[self.applied_force_index], y_final[self.applied_force_index], force[0], force[1],
            angles='xy',                
            scale_units='xy',           
            scale=10,                    # 10x smaller
            width=0.005,                # optional: make shaft thicker
            color='red',
            label='Applied Force')
        ax2.set_title('Deflection at each node 2D')
        ax2.set_xlabel('Length (m)')
        ax2.set_ylabel('y (m)')
        ax2.set_ylim(-self.total_length*1.5/2, self.total_length*1.5/2)
        ax2.set_xlim(0, self.total_length*1.5)
        ax2.legend()
        ax2.grid(True)
        ax2.set_aspect('equal')
        # Bottom-left: 2D scatter
        history = self.rod_history[len(self.rods) - 1]
        ax3 = fig.add_subplot(1, 3, 3)
        ax3.plot(history["step"][:], np.array(history["position"])[:, 1, -1], label='Deflected Beam over time', color='green')
        ax3.plot(history["step"][:], np.array(history["position"])[:, 1, -1], label='Deflected Beam over time', color='green')
        ax3.set_xlabel('# Steps')
        ax3.set_ylabel('y (m)')
        ax3.set_title('Deflection of distal node over time ')
        ax3.legend()
        ax3.grid(True)
        plt.tight_layout()
        plt.show()

class SystemSimulator(BaseSystemCollection, Connections, Constraints, Forcing, CallBacks, Damping): 
        pass 

class DynamicPerpendicularEndpointForce(NoForces):
    def __init__(self, magnitude: float, ref_direction: np.ndarray = np.array([0.0, 0.0, -1.0])):
        """
        Apply a force at the rod's endpoint that stays perpendicular to the current tangent.
        - magnitude: scalar value of the force
        - ref_direction: used to define the plane of perpendicularity (default: z-axis)
        """
        self.magnitude = magnitude
        self.ref_direction = ref_direction / np.linalg.norm(ref_direction)

    def apply_torques(self, system, time: float):
        pass  # No torque applied

    def apply_forces(self, system, time: float):
        # Get tangent vector at last element (near the tip)
        tangent = system.tangents[:, -1]
        tangent /= np.linalg.norm(tangent)

        # Compute a perpendicular direction using cross product
        perp = np.cross(tangent, self.ref_direction)

        # If ref is parallel to tangent, switch to another reference direction
        if np.linalg.norm(perp) < 1e-6:
            alt_ref = np.array([0.0, -1.0, 0.0])
            perp = np.cross(tangent, alt_ref)

        perp /= np.linalg.norm(perp)
        force_vector = self.magnitude * perp
        # Apply force only at the endpoint (last node)
        system.external_forces[:, -1] += force_vector

class MyCallBack(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            # Save time, step number, position, orientation and velocity
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position" ].append(system.position_collection.copy())
            self.callback_params["directors"].append(system.director_collection.copy())
            self.callback_params["velocity" ].append(system.velocity_collection.copy())
            # Save applied forces
            tip_force_vec = system.external_forces[:, -1]
            self.callback_params["applied_force"].append(tip_force_vec.copy())
            return

class MySurrogateModel:
    def __init__(self, method, elasticaInstance):
            self.ea = elasticaInstance
            # Initialize the RBF model
            if method == "RBF":
                self.model = RBF(d0=2, poly_degree=1, reg=1e-25)
            elif method == "KRG":
                self.model = KRG()
            elif method == "KPLSK":
                self.model = KPLSK(
                    n_comp=8,  # Improves speed & generalization
                    nugget=1e-4,
                    theta0=[1e-1],  # Better than 0.01
                    hyper_opt="Cobyla",  # Faster and more stable for high-D problems
                )
            else:
                 raise ValueError("Invalid surrogate model type!")
            
    def get_training_data(self, F_limits, s_limits, n_samples):
        limits = np.array([[F_limits[0], F_limits[-1]], [s_limits[0], s_limits[-1]]]) 
        sampling = LHS(xlimits=limits, criterion='maximin')
        samples = sampling(n_samples)  # Generate initial training points
        F_samples = np.array(samples[:, 0]) # get all sampeld F's
        s_samples = np.array(samples[:, 1]) # get all sampeld s's
        x_train = np.stack((F_samples, s_samples), axis=1)
        y_train = Parallel(n_jobs=-1)(
            delayed(self.ea.getProximalValues)(F, s) for F, s in x_train
        )
        # Save the scaled training and test sets as .npy
        os.makedirs("distal_to_proximal_training_data", exist_ok=True)
        np.save("distal_to_proximal_training_data/x_train.npy", x_train)
        np.save("distal_to_proximal_training_data/y_train.npy", y_train)
        raise ValueError("Sampling calculations complete! Please re-run code")
    
    def train_model(self):
        # Load training data
        x_train = np.load("distal_to_proximal_training_data/x_train.npy")
        y_train = np.load("distal_to_proximal_training_data/y_train.npy")
        # Train the model
        self.model.set_training_values(x_train, y_train)
        self.model.train()
        # Predict on the test set
        y_train_pred, y_train_std = self.model.predict_values(x_train), self.model.predict_variances(x_train)
        train_error = np.linalg.norm(y_train - y_train_pred) / np.linalg.norm(y_train)
        print("\ntrain error:", train_error)
    
    def predict(self, x):
        prediction = self.model.predict_values(x)
        return prediction

class MyOptimization:
    def __init__(self, elasticaInstance=None, model=None):
        self.ea = elasticaInstance
        self.model = model

    def objective(self, x):
        # the objective function
        F, s = x
        Fx_sim, MB_sim = self.ea.getProximalValues(F, s, self.model)
        obj = (Fx_sim - self.Fx_target)**2 + (MB_sim - self.MB_target)**2 
        return obj

    def solve(self, Fx, MB, x0):
        self.Fx_target = Fx
        self.MB_target = MB
        self.x0 = x0
        # optimize returns a dictionary that contains the results from optimization
        results = optimize(x0, obj=self.objective, acc=1e-6, finite_diff_abs_step=0.01)
        print_dict_as_table(results)

if __name__ == "__main__":
    ela = ElasticaForceEstimation(np.linspace(0, 0.2, 21), np.zeros(21), np.zeros(21), np.full(21, 1.5e-3), 2.12e9, 0.35)
    sur = MySurrogateModel("KRG", ela)
    #sur.train_model()
    fx, mb = ela.getProximalValues(0.55, 0.16)
    print(fx)
    print(mb)
    #fx, mb = ela.getProximalValues(0.2, 0.2)
    #print(fx)
    #print(mb)
    #opt = MyOptimization(ela, sur)
    #ela.get_distal_values(fx, mb, opt)


