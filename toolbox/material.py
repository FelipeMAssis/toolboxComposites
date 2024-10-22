import numpy as np

class Material:
    """
    A class to represent the material properties of a composite lamina.

    Attributes:
    - E11: Elastic modulus in the fiber direction [MPa]
    - E22: Elastic modulus in the transverse direction [MPa]
    - G12: Shear modulus [MPa]
    - v12: Poisson's ratio in the 1-2 plane
    - v21: Poisson's ratio in the 2-1 plane (calculated)
    - alpha: Thermal expansion coefficients as a column vector (alpha11, alpha22, alpha12)
    - rho: Material density [kg/m^3] (optional)
    
    Methods:
    - __init__: Initializes the material properties, including mechanical and thermal coefficients.
    - calculate_v21: Calculates and returns Poisson's ratio in the 2-1 plane based on E11, E22, and v12.
    """

    def __init__(self, E11, E22, G12, v12, alpha11=0, alpha22=0, alpha12=0, rho=None):
        """
        Initialize the Material class with mechanical and thermal properties.

        Parameters:
        - E11: Elastic modulus in the fiber direction [MPa]
        - E22: Elastic modulus in the transverse direction [MPa]
        - G12: Shear modulus [MPa]
        - v12: Poisson's ratio in the 1-2 plane
        - alpha11: Thermal expansion coefficient in the fiber direction [default: 0]
        - alpha22: Thermal expansion coefficient in the transverse direction [default: 0]
        - alpha12: Shear thermal expansion coefficient [default: 0]
        - rho: Material density [kg/m^3] (optional)
        """
        self.E11 = E11  # Elastic modulus in the fiber direction [MPa]
        self.E22 = E22  # Elastic modulus in the transverse direction [MPa]
        self.G12 = G12  # Shear modulus [MPa]
        self.v12 = v12  # Poisson's ratio in the 1-2 plane

        # Poisson's ratio in the 2-1 plane, calculated from v12 and the modulus ratio
        self.v21 = self.calculate_v21()

        # Thermal expansion coefficients stored in a column vector
        self.alpha = np.array([[alpha11], [alpha22], [alpha12]])
        
        self.rho = rho

    def calculate_v21(self):
        """
        Calculate Poisson's ratio in the transverse (2-1) direction.
        Formula: v21 = (E22 * v12) / E11

        Returns:
        - v21: Poisson's ratio in the 2-1 plane
        """
        return (self.E22 * self.v12) / self.E11
