import numpy as np

class Spring:
    def __init__(self,stiffness,x_bounds,y_bounds,direction,type=0):
        self.stiffness = stiffness
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.direction = direction
        self.type = type
