import numpy as np
from utils.tensors import vector, tensor, rotate

class Lamina:
    """
    A class to represent a lamina in a composite laminate, defined by its material properties, thickness, and orientation.

    Attributes:
    - material: Material object containing properties like E11, E22, G12, v12, etc.
    - t: Thickness of the lamina
    - Q: Stiffness matrix for the lamina in the principal material coordinate system
    - S: Compliance matrix (inverse of Q)
    - theta: Orientation angle of the lamina (in degrees)
    - T: Transformation matrix based on the orientation angle
    - Qbar: Transformed stiffness matrix (Q) in global coordinates
    - Sbar: Transformed compliance matrix (S) in global coordinates
    - alphabar: Transformed thermal expansion coefficients in global coordinates
    - zbot: Bottom coordinate of the lamina (optional, can be assigned later)
    - ztop: Top coordinate of the lamina (optional, can be assigned later)

    Methods:
    - __init__: Initializes the lamina with material, thickness, and orientation angle.
    - calc_Q: Calculates and returns the stiffness matrix Q for the lamina.
    - calc_T: Calculates and returns the transformation matrix T based on the lamina's angle.
    - calc_Qbar: Calculates and returns the transformed stiffness matrix Qbar in global coordinates.
    - calc_alphabar: Calculates and returns the transformed thermal expansion coefficients.
    - calc_sigma: Calculates the stress based on strain and thermal effects.
    - calc_eps: Calculates the strain based on stress and thermal effects.
    """

    def __init__(self, material, t, theta=0, deg=True):
        """
        Initialize the Lamina class with material properties, thickness, and angle.

        Parameters:
        - material: Material object with properties like E11, E22, G12, etc.
        - t: Thickness of the lamina
        - theta: Lamina orientation angle in degrees (default: 0)
        - deg: Boolean flag to determine if theta is in degrees (default: True)
        """
        self.material = material
        self.t = t
        self.Q = self.calc_Q()
        self.S = np.linalg.inv(self.Q)
        self.theta = theta if deg else np.degrees(theta)
        self.T = self.calc_T()
        self.Qbar = self.calc_Qbar()
        self.Sbar = np.linalg.inv(self.Qbar)
        self.alphabar = self.calc_alphabar()
        self.zbot = None
        self.ztop = None
    
    def calc_Q(self):
        """
        Calculate the stiffness matrix Q using the material properties.

        Returns:
        - Q: Stiffness matrix for the lamina in the principal material coordinate system.
        """
        E11 = self.material.E11
        E22 = self.material.E22
        G12 = self.material.G12
        v12 = self.material.v12
        v21 = self.material.v21
        
        denom = 1 - v12 * v21
        return np.array(
            [[E11 / denom, v12 * E22 / denom, 0],
             [v12 * E22 / denom, E22 / denom, 0],
             [0, 0, G12]]
        )

    def calc_T(self):
        """
        Calculate the transformation matrix T based on the lamina angle theta.

        Returns:
        - T: Transformation matrix, which is used to transform stiffness and strain components between coordinate systems.
        """
        theta_rad = np.radians(self.theta)
        c = np.cos(theta_rad)
        s = np.sin(theta_rad)
        return np.array(
            [[c**2, s**2, c * s],
             [s**2, c**2, -c * s],
             [-2 * c * s, 2 * c * s, c**2 - s**2]]
        )
    
    def calc_Qbar(self):
        """
        Calculate the transformed stiffness matrix Qbar using the transformation matrix T.

        Returns:
        - Qbar: Transformed stiffness matrix in global coordinates.
        """
        return self.T.T @ self.Q @ self.T

    def calc_alphabar(self):
        """
        Calculate the transformed thermal expansion coefficients.

        Returns:
        - alphabar: Transformed thermal expansion coefficients in global coordinates.
        """
        return self.T @ self.material.alpha
    

    def calc_sigma(self, eps, deltaT=0):
        """
        Calculate the stress from strain and thermal effects.

        Parameters:
        - eps: Strain vector in global coordinates
        - deltaT: Temperature difference (default: 0)

        Returns:
        - sigma: Stress vector in global coordinates
        - sigmap: Stress vector in principal material coordinates
        - epst: Thermal strain due to temperature variation
        """
        eps_vec = vector(eps)
        epst = deltaT * self.alphabar
        sigma = self.Qbar @ (eps_vec - epst)
        sigmap = rotate(self.theta, sigma)
        return sigma, sigmap, epst
    

    def calc_eps(self, sigma, deltaT=0):
        """
        Calculate the strain from stress and thermal effects.

        Parameters:
        - sigma: Stress vector in global coordinates
        - deltaT: Temperature difference (default: 0)

        Returns:
        - eps: Strain vector in global coordinates
        - epsp: Strain vector in principal material coordinates
        - epst: Thermal strain due to temperature variation
        """
        sigma_vec = vector(sigma)
        epst = deltaT * self.alphabar
        eps = self.Sbar @ sigma_vec + epst
        epsp = rotate(self.theta, eps)
        return eps, epsp, epst
