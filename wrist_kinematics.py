import numpy as np
import matplotlib.pyplot as plt

class AsymmetricWristKinematics:
    def __init__(self,
                 length=0.2,
                 ri=0.020,
                 ro=0.040,
                 h=0.015732):
        """
        Geometry parameters for constant-curvature wrist.
        length: total arc length [m]
        ri: inner tube radius [m]
        ro: outer tube radius [m]
        h: cut height [m]
        """
        self.length = length
        self.ri = ri
        self.ro = ro
        self.h = h
        # Neutral-axis offset
        self.y_bar = 0.5 * (ri + ro)

    def curvature(self, delta_l):
        """
        Compute curvature κ from tendon displacement Δl [m].
        Uses first-order approximation: κ = Δl / (h (ri + y_bar) - Δl y_bar)
        """
        num = delta_l
        den = self.h * (self.ri + self.y_bar) - delta_l * self.y_bar
        return num / den

    def shape(self, delta_l, n_points=61):
        """
        Return (x, y) arrays of the wrist body for given Δl [m].
        Approximates a constant-curvature arc of length self.length.
        """
        kappa = self.curvature(delta_l)
        s = self.length
        if abs(kappa) < 1e-8:
            x = np.linspace(0, s, n_points)
            y = np.zeros_like(x)
            return x, y
        ss = np.linspace(0, s, n_points)
        x = (1/kappa) * np.sin(kappa * ss)
        y = (1/kappa) * (1 - np.cos(kappa * ss))
        return x, y

    def delta_l_from_shape(self, x, y):
        """
        Compute tendon displacement Δl [m] given x, y arrays of a constant-curvature arc.
        Inverts the mapping: curvature → Δl.
        """
        x_end = x[-1]
        y_end = y[-1]
        # Total bend angle θ satisfies tan(θ/2) = y_end / x_end
        theta = 2 * np.arctan2(y_end, x_end)
        kappa = theta / self.length
        # Invert curvature-to-displacement relation
        delta_l = (kappa * self.h * (self.ri + self.y_bar)) / (1 + kappa * self.y_bar)
        return delta_l

    def plot(self, delta_l):
        """
        Plot the wrist outline for a given Δl [m].
        """
        x, y = self.shape(delta_l)
        plt.figure(figsize=(5,5))
        plt.plot(x, y, '-o', markersize=4)
        plt.axis('equal')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title(f'Wrist Shape for Δl = {delta_l*1e3:.1f} mm')
        plt.grid(True)
        plt.show()
