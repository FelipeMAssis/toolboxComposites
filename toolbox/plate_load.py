import numpy as np
import matplotlib.pyplot as plt

class UniformLoadNavier:
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
        return amn

class Load:
    def __init__(self,q,x_bounds,y_bounds,direction,type=0):
        self.q = q
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.direction = direction
        self.type = type