import numpy as np
from utils.tensors import tensor
from toolbox.results import LaminaResults
from scipy.optimize import fsolve

class CommomCriterion:
    def __init__(self, Xt, Yt, Xc, Yc, S):
        self.Xt = Xt
        self.Yt = Yt
        self.Xc = Xc
        self.Yc = Yc
        self.S = S  

class MaxStressCriterion(CommomCriterion):
    def __init__(self, Xt, Yt, Xc, Yc, S):
        self.name = 'Max Stress'
        super().__init__(Xt, Yt, Xc, Yc, S)
    
    def evaluate(self, lamina_results):
        sigmap = lamina_results.sigmap
        sigmap = tensor(sigmap)
        sigma11, sigma22, sigma12 = sigmap[0, 0], sigmap[1, 1], sigmap[0,1]
        crit1t = sigma11/self.Xt
        crit1c = -sigma11/self.Xc
        crit2t = sigma22/self.Yt
        crit2c = -sigma22/self.Yc
        crit3 = abs(sigma12)/self.S
        crit = max(crit1t, crit1c, crit2t, crit2c, crit3)
        fail =  crit >= 1
        strength = 1/crit if np.sqrt(crit)!=0 else 0
        return fail, crit, strength

class MaxStrainCriterion(CommomCriterion):
    def __init__(self, Xt, Yt, Xc, Yc, S):
        self.name = 'Max Strain'
        super().__init__(Xt, Yt, Xc, Yc, S)

    def evaluate(self, lamina_results):
        sigmap = lamina_results.sigmap
        sigmap = tensor(sigmap)
        v12 = lamina_results.lamina.material.v12
        v21 = lamina_results.lamina.material.v21
        sigma11, sigma22, sigma12 = sigmap[0, 0], sigmap[1, 1], sigmap[0,1]
        crit1t = (sigma11 - v12 * sigma22)/self.Xt
        crit1c = -(sigma11 - v12 * sigma22)/self.Xc
        crit2t = (sigma22 - v21 * sigma11)/self.Yt
        crit2c = -(sigma22 - v21 * sigma11)/self.Yc
        crit3 = abs(sigma12)/self.S
        crit = max(crit1t, crit1c, crit2t, crit2c, crit3)
        fail =  crit >= 1
        strength = 1/crit if np.sqrt(crit)!=0 else 0
        return fail, crit, strength

class TsaiHillCriterion(CommomCriterion):
    def __init__(self, Xt, Yt, Xc, Yc, S):
        self.name = 'Tsai Hill'
        super().__init__(Xt, Yt, Xc, Yc, S)

    def evaluate(self, lamina_results):
        sigmap = lamina_results.sigmap
        sigmap = tensor(sigmap)
        sigma11, sigma22, tau12 = sigmap[0, 0], sigmap[1, 1], sigmap[0, 1]
        X = self.Xt if sigma11 >= 0 else self.Xc
        Y = self.Yt if sigma22 >= 0 else self.Yc
        crit = sigma11**2 / X**2 - sigma11 * sigma22 / X**2 + sigma22**2 / Y**2 + tau12**2 / self.S**2
        fail = crit >= 1
        strength = 1/np.sqrt(crit) if np.sqrt(crit)!=0 else 0
        return fail, crit, strength

class HoffmanCriterion(CommomCriterion):
    def __init__(self, Xt, Yt, Xc, Yc, S):
        self.name = 'Hoffman'
        super().__init__(Xt, Yt, Xc, Yc, S)

    def evaluate(self, lamina_results):
        sigmap = lamina_results.sigmap
        crit = self.evaluate_crit(sigmap)
        fail = crit >= 1
        strength = abs(float(fsolve(lambda s: (self.evaluate_crit(abs(s)*sigmap)-1)**2, 1)))
        return fail, crit, strength
        
    def evaluate_crit(self,sigmap):
        
        sigmap = tensor(sigmap)
        sigma11, sigma22, tau12 = sigmap[0, 0], sigmap[1, 1], sigmap[0, 1]
        crit = (
            sigma11**2 / (self.Xc * self.Xt)
            - sigma11 * sigma22 / (self.Xc * self.Xt)
            + sigma22**2 / (self.Yc * self.Yt)
            - (self.Xt - self.Xc) * sigma11 / (self.Xc * self.Xt)
            - (self.Yt - self.Yc) * sigma22 / (self.Yc * self.Yt)
            + tau12**2 / self.S**2
        )
        return crit

class TsaiWuCriterion(CommomCriterion):
    def __init__(self, Xt, Yt, Xc, Yc, S):
        self.name = 'Tsai Wu'
        super().__init__(Xt, Yt, Xc, Yc, S)

    def evaluate(self, lamina_results):
        sigmap = lamina_results.sigmap
        crit = self.evaluate_crit(sigmap)
        fail = crit >= 1
        strength = abs(float(fsolve(lambda s: (self.evaluate_crit(abs(s)*sigmap)-1)**2, 1)))
        return fail, crit, strength
        
    
    def evaluate_crit(self, sigmap):
        F1 = 1/self.Xt-1/self.Xc
        F2 = 1/self.Yt-1/self.Yc
        F11 = 1/(self.Xt*self.Xc)
        F22 = 1/(self.Yt*self.Yc)
        F66 = 1/self.S**2
        F12 = -((F11*F22)**(1/2))/2
        sigmap = tensor(sigmap)
        sigma11, sigma22, tau12 = sigmap[0, 0], sigmap[1, 1], sigmap[0, 1]
        crit = F1 * sigma11 + F2 * sigma22 + F11 * sigma11**2 + F22 * sigma22**2 + F66 * tau12**2 + 2 * F12 * sigma11 * sigma22
        return crit