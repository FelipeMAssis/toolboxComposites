#import numpy as np
#import matplotlib.pyplot as plt

class Load:
    """
    A class representing a load applied to a plate.

    Attributes:
    - q: Magnitude of the applied load.
    - x_bounds: Tuple representing the x-coordinate bounds of the load (e.g., (x_min, x_max)).
    - y_bounds: Tuple representing the y-coordinate bounds of the load (e.g., (y_min, y_max)).
    - direction: Direction in which the load is applied (e.g., 'x', 'y', or 'z').
    - type: Type of load (default is 0, which can represent a uniform load, but may be extended to represent other types).

    Methods:
    - __init__: Initializes the load with magnitude, bounds, direction, and type.
    """

    def __init__(self,q,x_bounds,y_bounds,direction,type=0):
        """
        Initialize the Load class.

        Parameters:
        - q: Magnitude of the load applied.
        - x_bounds: Tuple representing the x-coordinate bounds where the load is applied (x_min, x_max) or (x) for concentrated loads.
        - y_bounds: Tuple representing the y-coordinate bounds where the load is applied (y_min, y_max) or (y) for concentrated loads.
        - direction: Direction of the applied load (e.g., 'x', 'y', 'z').
        - type: Integer representing the type of load (default is 0 for a uniform load).
        """
        self.q = q
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.direction = direction
        self.type = type


# Solução de Navier não validada

'''class UniformLoadNavier:
    def __init__(self,q0):
        self.q0 = q0
        
    def calc_amn(self,m,n):
        amn = np.zeros([m,n])
        for i in range(m):
            for j in range(n):
                amn[i,j] = 16*self.q0/(np.pi**2*(i*2-1)*(j*2-1))
        return amn

class PointLoadNavier:
    def __init__(self,q,x,y,a,b):
        self.q = q
        self.x = x
        self.y = y
        self.a = a
        self.b = b

    def calc_amn(self,m,n):
        a = self.a
        b = self.b
        x = self.x
        y = self.y
        q = self.q
        amn = np.zeros([m,n])
        for i in range(m):
            for j in range(n):
                amn[i,j] = q*np.sin((i*2-1)*np.pi*x/a)*np.sin((j*2-1)*np.pi*y/b)
        amn = amn*4/(self.a*self.b)
        return amn'''