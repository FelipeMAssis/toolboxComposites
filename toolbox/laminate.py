import numpy as np
from toolbox.material import Material
from toolbox.lamina import Lamina

class Laminate:
    def __init__(self, layup):
        self.layup = layup      
        self.A, self.B, self.D = self.ABD()

    def ABD(self):
        t = [lam.t for lam in self.layup]
        z = (np.cumsum([0]+t)-sum(t)/2)
        Qbar = [lam.Q for lam in self.layup]
        A = np.zeros(3)
        B = np.zeros(3)
        D = np.zeros(3)
        for i,Qbarl in enumerate(Qbar):
            A = A + (z[i+1]-z[i])*Qbarl
            B = B + ((z[i+1]**2-z[i]**2)/2)*Qbarl
            D = D + ((z[i+1]**3-z[i]**3)/3)*Qbarl
        return A,B,D

    def def2forces(self, eps0, kappa, deltaT):
        t = [lam.t for lam in self.layup]
        z = (np.cumsum([0]+t)-sum(t)/2)
        Qbar = [lam.Qbar for lam in self.layup]
        alphabar = [lam.alphabar for lam in self.layup]
        Nt = np.zeros([3,1])
        Mt = np.zeros([3,1])
        for i,Qbarl in enumerate(Qbar):
            Nt = Nt + (z[i+1]-z[i])*(Qbarl @ (deltaT*alphabar[i]))
            Mt = Mt + ((z[i+1]**2-z[i]**2)/2)*(Qbarl @ (deltaT*alphabar[i]))
        N = self.A @ eps0 + self.B @ kappa - Nt
        M = self.B @ eps0 + self.D @ kappa - Mt
        return N,M
    
    def forces2def(self, Nm, Mm, deltaT):
        t = [lam.t for lam in self.layup]
        z = (np.cumsum([0]+t)-sum(t)/2)
        Qbar = [lam.Qbar for lam in self.layup]
        alphabar = [lam.alphabar for lam in self.layup]
        Nt = np.zeros([3,1])
        Mt = np.zeros([3,1])
        for i,Qbarl in enumerate(Qbar):
            Nt = Nt + (z[i+1]-z[i])*(Qbarl @ (deltaT*alphabar[i]))
            Mt = Mt + ((z[i+1]**2-z[i]**2)/2)*(Qbarl @ (deltaT*alphabar[i]))
        N = Nm + Nt
        M = Mm + Mt
        ABD = np.block([[self.A,self.B],[self.B,self.D]])
        NM = np.block([[N],[M]])
        epskappa = np.linalg.inv(ABD) @ NM
        eps0 = np.matrix(epskappa[:3,0]).transpose()
        kappa = np.matrix(epskappa[3:,0]).transpose()
        return eps0, kappa