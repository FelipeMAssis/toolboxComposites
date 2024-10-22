# import numpy as np

class Plate:
    """
    A class representing a composite plate constructed from a laminate.

    Attributes:
    - laminate: The Laminate object that contains stiffness matrices (ABD), particularly the D matrix for bending stiffness
    - length: Length of the plate (denoted as 'a')
    - width: Width of the plate (denoted as 'b')

    Methods:
    - __init__: Initializes the plate with a laminate, length, and width.
    """
    def __init__(self, laminate, length, width):
        """
        Initialize the Plate class with a laminate, length, and width.

        Parameters:
        - laminate: Laminate object containing stiffness matrix D
        - length: Length of the plate (a)
        - width: Width of the plate (b)
        """
        self.laminate = laminate
        self.length = length
        self.width = width

    # Solução de Navier não validada

    '''def solve_navier(self, amn):
        """
        Solve the Navier solution for the plate based on input coefficients amn.

        Parameters:
        - amn: Coefficient matrix (m x n) for Fourier series terms

        Returns:
        - Amn: Solved coefficient matrix (m x n)
        """
        a = self.length
        b = self.width
        D11 = self.laminate.D[0, 0]
        D22 = self.laminate.D[1, 1]
        D12 = self.laminate.D[0, 1]
        D66 = self.laminate.D[2, 2]

        m, n = amn.shape
        Amn = np.zeros((m, n))

        # Iterate through the amn matrix and solve for each term in Amn
        for i in range(m):
            i_term = (i * 2 - 1) / a  # Precompute to avoid recalculating in inner loop
            for j in range(n):
                j_term = (j * 2 - 1) / b  # Precompute j_term
                
                denominator = (D11 * i_term**4 +
                               2 * (D12 + 2 * D66) * i_term**2 * j_term**2 +
                               D22 * j_term**4)
                Amn[i, j] = amn[i, j] / denominator if denominator != 0 else 0

        return Amn'''
    

