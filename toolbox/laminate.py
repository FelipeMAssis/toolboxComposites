import numpy as np
from toolbox.material import Material
from toolbox.lamina import Lamina

class Laminate:
    def __init__(self, layup):
        """
        Initialize the Laminate class with a given layup.

        Parameters:
        - layup: List of Lamina objects representing each layer of the laminate
        """
        self.layup = layup      
        self.A, self.B, self.D = self.compute_ABD_matrices()

    def compute_ABD_matrices(self):
        """
        Compute the A, B, and D matrices for the laminate.

        Returns:
        - A: Extensional stiffness matrix
        - B: Coupling stiffness matrix
        - D: Bending stiffness matrix
        """
        thicknesses = [lam.t for lam in self.layup]
        z_coords = np.cumsum([0] + thicknesses) - np.sum(thicknesses) / 2
        Qbars = [lam.Qbar for lam in self.layup]

        # Initialize A, B, and D matrices as zero matrices
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))

        for i, Qbar in enumerate(Qbars):
            delta_z = z_coords[i+1] - z_coords[i]
            A += delta_z * Qbar
            B += (z_coords[i+1]**2 - z_coords[i]**2) / 2 * Qbar
            D += (z_coords[i+1]**3 - z_coords[i]**3) / 3 * Qbar

        return A, B, D
    
    def thermal_forces_and_moments(self, deltaT):
        """
        Compute thermal forces (N) and moments (M) due to a temperature change.

        Parameters:
        - deltaT: Temperature difference

        Returns:
        - Nt: Thermal force vector
        - Mt: Thermal moment vector
        """
        thicknesses = [lam.t for lam in self.layup]
        z_coords = np.cumsum([0] + thicknesses) - np.sum(thicknesses) / 2
        Qbars = [lam.Qbar for lam in self.layup]
        alphabars = [lam.alphabar for lam in self.layup]

        Nt = np.zeros((3, 1))
        Mt = np.zeros((3, 1))

        for i, Qbar in enumerate(Qbars):
            delta_z = z_coords[i+1] - z_coords[i]
            Nt += delta_z * (Qbar @ (deltaT * alphabars[i]))
            Mt += (z_coords[i+1]**2 - z_coords[i]**2) / 2 * (Qbar @ (deltaT * alphabars[i]))

        return Nt, Mt
    
    def def2forces(self, eps0, kappa, deltaT=0):
        """
        Compute the resultant forces (N) and moments (M) based on strains, curvatures, and temperature.

        Parameters:
        - eps0: Mid-plane strain vector
        - kappa: Curvature vector
        - deltaT: Temperature difference (default: 0)

        Returns:
        - N: Resultant force vector
        - M: Resultant moment vector
        """
        Nt, Mt = self.thermal_forces_and_moments(deltaT)

        N = self.A @ eps0 + self.B @ kappa - Nt
        M = self.B @ eps0 + self.D @ kappa - Mt

        return N, M
    
    def forces2def(self, Nm, Mm, deltaT=0):
        """
        Compute the mid-plane strains (eps0) and curvatures (kappa) from given forces and moments.

        Parameters:
        - Nm: Mechanical force vector
        - Mm: Mechanical moment vector
        - deltaT: Temperature difference (default: 0)

        Returns:
        - eps0: Mid-plane strain vector
        - kappa: Curvature vector
        """
        Nt, Mt = self.thermal_forces_and_moments(deltaT)

        # Adjust forces and moments to account for thermal effects
        N = Nm + Nt
        M = Mm + Mt

        # Construct the full ABD matrix and solve for strains and curvatures
        ABD_matrix = np.block([[self.A, self.B], [self.B, self.D]])
        NM_vector = np.vstack((N, M))

        eps_kappa = np.linalg.inv(ABD_matrix) @ NM_vector
        eps0 = eps_kappa[:3]
        kappa = eps_kappa[3:]

        return eps0, kappa