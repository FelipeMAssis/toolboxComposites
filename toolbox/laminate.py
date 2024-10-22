import numpy as np
from toolbox.material import Material
from toolbox.lamina import Lamina
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utils.angular_hatch import AngularHatch
matplotlib.hatch._hatch_types.append(AngularHatch)

class Laminate:
    """
    A class representing a composite laminate, composed of multiple lamina layers.

    Attributes:
    - layup: List of Lamina objects representing the layers of the laminate
    - thickness: Total thickness of the laminate
    - A: Extensional stiffness matrix of the laminate
    - B: Coupling stiffness matrix of the laminate
    - D: Bending stiffness matrix of the laminate

    Methods:
    - __init__: Initializes the laminate with the layup of lamina layers.
    - calc_thickness: Calculates the total thickness of the laminate.
    - assign_lamina_z: Assigns z-coordinates (top and bottom) to each lamina based on thickness.
    - compute_ABD_matrices: Computes the A, B, and D stiffness matrices.
    - calc_thermal_forces: Computes thermal forces and moments due to a temperature change.
    - calc_forces: Computes resultant forces and moments due to strain, curvature, and temperature effects.
    - calc_def: Calculates mid-plane strains and curvatures from applied forces and moments.
    - lamina_def: Computes strain distribution across individual laminae.
    """

    def __init__(self, layup):
        """
        Initialize the Laminate class with a given layup.

        Parameters:
        - layup: List of Lamina objects representing each layer of the laminate
        """
        self.layup = layup
        self.thickness = self.calc_thickness()
        self.assign_lamina_z()    
        self.A, self.B, self.D = self.compute_ABD_matrices()

    def calc_thickness(self):
        """
        Calculate the total thickness of the laminate.

        Returns:
        - thickness: Total thickness of the laminate
        """
        return sum([lam.t for lam in self.layup])

    def assign_lamina_z(self):
        """
        Assign the z-coordinates (top and bottom) to each lamina in the laminate.
        This method calculates the top and bottom coordinates for each lamina 
        based on the laminate thickness distribution.
        """
        thicknesses = [lam.t for lam in self.layup]
        z_coords = np.cumsum([0] + thicknesses) - np.sum(thicknesses) / 2
        for i,lam in enumerate(self.layup):
            lam.zbot = z_coords[i]
            lam.ztop = z_coords[i+1]

    def compute_ABD_matrices(self):
        """
        Compute the A, B, and D stiffness matrices for the laminate.
        These matrices describe the extensional, coupling, and bending stiffness of the laminate.

        Returns:
        - A: Extensional stiffness matrix
        - B: Coupling stiffness matrix
        - D: Bending stiffness matrix
        """
        # Initialize A, B, and D matrices as zero matrices
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))

        for lam in self.layup:
            A += (lam.ztop-lam.zbot) * lam.Qbar
            B += (lam.ztop**2 - lam.zbot**2) / 2 * lam.Qbar
            D += (lam.ztop**3 - lam.zbot**3) / 3 * lam.Qbar

        return A, B, D
    
    def calc_thermal_forces(self, deltaT):
        """
        Compute thermal forces and moments due to a temperature change.

        Parameters:
        - deltaT: Temperature difference

        Returns:
        - Nt: Thermal force vector (3x1)
        - Mt: Thermal moment vector (3x1)
        """
        Nt = np.zeros((3, 1))
        Mt = np.zeros((3, 1))

        for lam in self.layup:
            Nt += (lam.ztop-lam.zbot) * (lam.Qbar @ (deltaT * lam.alphabar))
            Mt += (lam.ztop**2 - lam.zbot**2) / 2 * (lam.Qbar @ (deltaT * lam.alphabar))

        return Nt, Mt
    
    def calc_forces(self, eps0, kappa, deltaT=0):
        """
        Compute the resultant forces and moments based on mid-plane strains, curvatures, and temperature.

        Parameters:
        - eps0: Mid-plane strain vector (3x1)
        - kappa: Curvature vector (3x1)
        - deltaT: Temperature difference (default: 0)

        Returns:
        - N: Resultant force vector (3x1)
        - M: Resultant moment vector (3x1)
        - Nt: Thermal force vector (3x1)
        - Mt: Thermal moment vector (3x1)
        """
        Nt, Mt = self.calc_thermal_forces(deltaT)

        N = self.A @ eps0 + self.B @ kappa - Nt
        M = self.B @ eps0 + self.D @ kappa - Mt

        return N, M, Nt, Mt
    
    def calc_def(self, Nm, Mm, deltaT=0):
        """
        Compute the mid-plane strains (eps0) and curvatures (kappa) from applied forces and moments.

        Parameters:
        - Nm: Mechanical force vector (3x1)
        - Mm: Mechanical moment vector (3x1)
        - deltaT: Temperature difference (default: 0)

        Returns:
        - eps0: Mid-plane strain vector (3x1)
        - kappa: Curvature vector (3x1)
        - Nt: Thermal force vector (3x1)
        - Mt: Thermal moment vector (3x1)
        """
        Nt, Mt = self.calc_thermal_forces(deltaT)

        # Adjust forces and moments to account for thermal effects
        N = Nm + Nt
        M = Mm + Mt

        # Construct the full ABD matrix and solve for strains and curvatures
        ABD_matrix = np.block([[self.A, self.B], [self.B, self.D]])
        NM_vector = np.vstack((N, M))

        eps_kappa = np.linalg.inv(ABD_matrix) @ NM_vector
        eps0 = eps_kappa[:3]
        kappa = eps_kappa[3:]

        return eps0, kappa, Nt, Mt

    def lamina_def(self, eps0, kappa):
        """
        Compute strain distribution across individual laminae in the laminate.

        Parameters:
        - eps0: Mid-plane strain vector (3x1)
        - kappa: Curvature vector (3x1)

        Returns:
        - epsbot: Strain at the bottom surface of each lamina
        - epstop: Strain at the top surface of each lamina
        """
        epsbot = []
        epstop = []
        for lam in self.layup:
            epsbot.append(eps0 + lam.zbot*kappa)
            epstop.append(eps0 + lam.ztop*kappa)
        return epsbot, epstop

