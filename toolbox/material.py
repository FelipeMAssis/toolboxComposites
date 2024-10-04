import numpy as np

class Material:
    def __init__(self, E11, E22, G12, v12, alpha11=0, alpha22=0, alpha12=0):
        self.E11 = E11 # Elasticity modulus in the fiber direction [MPa]
        self.E22 = E22 # Elasticity modulus in the transverse direction [MPa]
        self.G12 = G12 # Shear modulus [MPa]
        self.v12 = v12 # Poisson's ration 12
        self.v21 = E22*v12/E11 # Poisson's ration 21
        self.alpha = np.array(
            [[alpha11],
             [alpha22],
             [alpha12]]
        )