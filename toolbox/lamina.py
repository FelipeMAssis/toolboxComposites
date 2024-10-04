import numpy as np
import matplotlib.pyplot as plt
from toolbox.utils import tensor2vec, vec2tensor, rotate

class Lamina:
    def __init__(self, material, t, theta=0, deg=True):
        self.material = material
        self.t = t
        self.Q = self.calc_Q()
        if deg:
            self.theta = theta
        else:
            self.theta = 180*theta/np.pi
        self.T = self.calc_T()
        self.Qbar = self.T.transpose() @ self.Q @ self.T
        self.Sbar = np.linalg.inv(self.Qbar)
        self.alphabar = self.T @ self.material.alpha

    def calc_Q(self):
        E11 = self.material.E11
        E22 = self.material.E22
        G12 = self.material.G12
        v12 = self.material.v12
        v21 = self.material.v21

        v21 = E22*v12/E11
        return np.array(
            [[E11/(1-v12*v21),           v12*E22/(1-v12*v21),    0],
            [v12*E22/(1-v12*v21),       E22/(1-v12*v21),        0],
            [0,                         0,                      G12]]
            )

    def calc_T(self):
        theta = np.pi*self.theta/180
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array(
            [[c**2,      s**2,        c*s],
            [s**2,       c**2,        -c*s],
            [-2*c*s,     2*c*s,       c**2-s**2]]
            )
    
    def calc_Qbar(self):
        return self.T.transpose() @ self.Q @ self.T
    
    def calc_sigma(self, eps, deltaT=0):
        if eps.shape[1]>1:
            eps = tensor2vec(eps)
        sigma = self.Qbar @ (eps-deltaT*self.alphabar)
        sigmap = rotate(self.theta, sigma)
        return sigma, sigmap
    
    def calc_eps(self, sigma, deltaT=0):
        if sigma.shape[1]>1:
            sigma = tensor2vec(sigma)
        eps = (self.Sbar @ sigma) + deltaT*self.alphabar
        epsp = rotate(self.theta, eps)
        return eps, epsp
    
    def max_stress(self, sigma, Xt, Yt, Xc, Yc, plot=False):
        if sigma.shape[1]==1:
            sigma = vec2tensor(sigma)
        sigmap = rotate(self.theta, sigma)
        sigma11 = sigmap[0,0]
        sigma22 = sigmap[1,1]

        if sigma11<Xt and sigma11>-Xc and sigma22<Yt and sigma22>-Yc:
            fail=False
        else:
            fail=True
            
        if plot:
            fig = plt.figure()
            plt.plot([Xt,Xt,-Xc,-Xc,Xt],[-Yc,Yt,Yt,-Yc,-Yc], color='r')
            plt.scatter(sigma11,sigma22)
            plt.hlines(0, 1.2*min(sigma11,-Xc), 1.2*max(sigma11,Xt), color='k')
            plt.vlines(0, 1.2*min(sigma22,-Yc), 1.2*max(sigma22,Yt), color='k')
            plt.xlim([1.2*min(sigma11,-Xc), 1.2*max(sigma11,Xt)])
            plt.ylim([1.2*min(sigma22,-Yc), 1.2*max(sigma22,Yt)])
            return fail,fig
        return fail
    
    def max_strain(self, sigma, Xt, Yt, Xc, Yc, plot=False):
        v12 = self.material.v12
        v21 = self.material.v21
        if sigma.shape[1]==1:
            sigma = vec2tensor(sigma)
        sigmap = rotate(self.theta, sigma)
        sigma11 = sigmap[0,0]
        sigma22 = sigmap[1,1]
        
        if sigma11-v12*sigma22<Xt and sigma11-v12*sigma22>-Xc and sigma22-v21*sigma11<Yt and sigma22-v21*sigma11>-Yc:
            fail=False
        else:
            fail=True
            
        if plot:
            s1c1 = (-Xc + v12*Yt)/(1-v12*v21)
            s2c1 = (Yt - v21*Xc)/(1-v12*v21)
            s1c2 = (Xt + v12*Yt)/(1-v12*v21)
            s2c2 = (Yt + v21*Xt)/(1-v12*v21)
            s1c3 = (Xt - v12*Yc)/(1-v12*v21)
            s2c3 = (-Yc + v21*Xt)/(1-v12*v21)
            s1c4 = (-Xc - v12*Yc)/(1-v12*v21)
            s2c4 = (-Yc - v21*Xc)/(1-v12*v21)
            fig = plt.figure()
            plt.plot(
                [s1c1,s1c2,s1c3,s1c4,s1c1],
                [s2c1,s2c2,s2c3,s2c4,s2c1],
                color='r'
                )
            plt.scatter(sigma11,sigma22)
            plt.hlines(0, 1.5*min(sigma11,-Xc), 1.5*max(sigma11,Xt), color='k')
            plt.vlines(0, 1.5*min(sigma22,-Yc), 1.5*max(sigma22,Yt), color='k')
            plt.xlim([1.5*min(sigma11,-Xc), 1.5*max(sigma11,Xt)])
            plt.ylim([1.5*min(sigma22,-Yc), 1.5*max(sigma22,Yt)])
            return fail,fig
        return fail

    def tsai_hill(self,sigma, Xt, Yt, Xc, Yc, S, plot=False):
        if sigma.shape[1]==1:
            sigma = vec2tensor(sigma)
        sigmap = rotate(self.theta, sigma)
        sigma11 = sigmap[0,0]
        sigma22 = sigmap[1,1]
        tau12 = sigmap[0,1]
        X = Xt if sigma11>=0 else Xc
        Y = Yt if sigma11>=0 else Yc
        crit = sigma11**2/X**2 - sigma11*sigma22/X**2 + sigma22**2/Y**2 + tau12**2/S**2
        fail = crit >= 1

        # TODO: Plot

        return fail, crit

    def hoffman(self, sigma, Xt, Yt, Xc, Yc, S, plot=False):
        if sigma.shape[1]==1:
            sigma = vec2tensor(sigma)
        sigmap = rotate(self.theta, sigma)
        sigma11 = sigmap[0,0]
        sigma22 = sigmap[1,1]
        tau12 = sigmap[0,1]
        crit = sigma11**2/(Xc*Xt) - sigma11*sigma22/(Xc*Xt) + sigma22**2/(Yc*Yt) - (Xt-Xc)*sigma11/(Xc*Xt) - (Yt-Yc)*sigma22/(Yc*Yt) + tau12**2/S**2
        fail = crit >= 1

        # TODO: Plot

        return fail, crit

    def tsai_wu(self, sigma, F1, F2, F11, F22, F66, F12, plot=False):
        if sigma.shape[1]==1:
            sigma = vec2tensor(sigma)
        sigmap = rotate(self.theta, sigma)
        sigma11 = sigmap[0,0]
        sigma22 = sigmap[1,1]
        tau12 = sigmap[0,1]
        crit = F1*sigma11 + F2*sigma22 + F11*sigma11**2 + F22*sigma22**2 + F66*tau12**2 + 2*F12*sigma11*sigma22
        fail = crit >= 1

        # TODO: Plot

        return fail, crit