import numpy as np
import matplotlib.pyplot as plt
from toolbox.utils import tensor2vec, vec2tensor, rotate

class Lamina:
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
        self.theta = theta if deg else np.degrees(theta)
        self.T = self.calc_T()
        self.Qbar = self.calc_Qbar()
        self.Sbar = np.linalg.inv(self.Qbar)
        self.alphabar = self.calc_alphabar()
    
    def calc_Q(self):
        """
        Calculate the stiffness matrix Q using the material properties.
        
        Returns:
        - Q: Stiffness matrix for the lamina
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
        - T: Transformation matrix
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
        Calculate the transformed stiffness matrix Qbar.

        Returns:
        - Qbar: Transformed stiffness matrix
        """
        return self.T.T @ self.Q @ self.T

    def calc_alphabar(self):
        """
        Calculate the transformed thermal expansion coefficients.

        Returns:
        - alphabar: Transformed thermal expansion coefficients
        """
        return self.T @ self.material.alpha
    

    def calc_sigma(self, eps, deltaT=0):
        """
        Calculate the stress from strain and thermal effects.

        Parameters:
        - eps: Strain vector
        - deltaT: Temperature difference (default: 0)

        Returns:
        - sigma: Stress in global coordinates
        - sigmap: Stress in principal material coordinates
        """
        eps_vec = tensor2vec(eps) if eps.shape[1] > 1 else eps
        sigma = self.Qbar @ (eps_vec - deltaT * self.alphabar)
        sigmap = rotate(self.theta, sigma)
        return sigma, sigmap
    

    def calc_eps(self, sigma, deltaT=0):
        """
        Calculate the strain from stress and thermal effects.

        Parameters:
        - sigma: Stress vector
        - deltaT: Temperature difference (default: 0)

        Returns:
        - eps: Strain in global coordinates
        - epsp: Strain in principal material coordinates
        """
        sigma_vec = tensor2vec(sigma) if sigma.shape[1] > 1 else sigma
        eps = self.Sbar @ sigma_vec + deltaT * self.alphabar
        epsp = rotate(self.theta, eps)
        return eps, epsp
    
    def max_stress(self, sigma, Xt, Yt, Xc, Yc, plot=False):
        """
        Evaluate the maximum stress failure criterion.

        Parameters:
        - sigma: Stress tensor
        - Xt, Yt: Tensile strengths in fiber and transverse directions
        - Xc, Yc: Compressive strengths in fiber and transverse directions
        - plot: Boolean flag to enable plotting (default: False)

        Returns:
        - fail: Boolean flag indicating failure (True/False)
        """
        sigma_tens = vec2tensor(sigma) if sigma.shape[1] == 1 else sigma
        sigmap = rotate(self.theta, sigma_tens)
        sigma11, sigma22 = sigmap[0, 0], sigmap[1, 1]

        fail = not (Xc < sigma11 < Xt and Yc < sigma22 < Yt)

        if plot:
            self.plot_failure_criteria(sigma11, sigma22, Xt, Yt, Xc, Yc)
        return fail

    def plot_failure_criteria(self, sigma11, sigma22, Xt, Yt, Xc, Yc):
        """
        Plot the maximum stress failure envelope and the current stress state.
        """
        plt.figure()
        plt.plot([Xt, Xt, -Xc, -Xc, Xt], [-Yc, Yt, Yt, -Yc, -Yc], color='r')
        plt.scatter(sigma11, sigma22)
        plt.hlines(0, 1.2 * min(sigma11, -Xc), 1.2 * max(sigma11, Xt), color='k')
        plt.vlines(0, 1.2 * min(sigma22, -Yc), 1.2 * max(sigma22, Yt), color='k')
        plt.xlim([1.2 * min(sigma11, -Xc), 1.2 * max(sigma11, Xt)])
        plt.ylim([1.2 * min(sigma22, -Yc), 1.2 * max(sigma22, Yt)])
        plt.show()

    def max_strain(self, sigma, Xt, Yt, Xc, Yc, plot=False):
        """
        Evaluate the maximum strain failure criterion.

        Parameters:
        - sigma: Stress tensor
        - Xt, Yt: Tensile strengths in fiber and transverse directions
        - Xc, Yc: Compressive strengths in fiber and transverse directions
        - plot: Boolean flag to enable plotting (default: False)

        Returns:
        - fail: Boolean flag indicating failure (True/False)
        """
        v12 = self.material.v12
        v21 = self.material.v21
        sigma_tens = vec2tensor(sigma) if sigma.shape[1] == 1 else sigma
        sigmap = rotate(self.theta, sigma_tens)
        sigma11, sigma22 = sigmap[0, 0], sigmap[1, 1]

        fail = not (Xc < sigma11 - v12 * sigma22 < Xt and Yc < sigma22 - v21 * sigma11 < Yt)

        if plot:
            self.plot_strain_failure_criteria(Xt, Yt, Xc, Yc, v12, v21, sigma11, sigma22)
        return fail

    def plot_strain_failure_criteria(self, Xt, Yt, Xc, Yc, v12, v21, sigma11, sigma22):
        """
        Plot the maximum strain failure envelope and the current stress state.
        """
        s1c1, s2c1 = (-Xc + v12 * Yt) / (1 - v12 * v21), (Yt - v21 * Xc) / (1 - v12 * v21)
        s1c2, s2c2 = (Xt + v12 * Yt) / (1 - v12 * v21), (Yt + v21 * Xt) / (1 - v12 * v21)
        s1c3, s2c3 = (Xt - v12 * Yc) / (1 - v12 * v21), (-Yc + v21 * Xt) / (1 - v12 * v21)
        s1c4, s2c4 = (-Xc - v12 * Yc) / (1 - v12 * v21), (-Yc - v21 * Xc) / (1 - v12 * v21)

        plt.figure()
        plt.plot([s1c1, s1c2, s1c3, s1c4, s1c1], [s2c1, s2c2, s2c3, s2c4, s2c1], color='r')
        plt.scatter(sigma11, sigma22)
        plt.hlines(0, 1.5 * min(sigma11, -Xc), 1.5 * max(sigma11, Xt), color='k')
        plt.vlines(0, 1.5 * min(sigma22, -Yc), 1.5 * max(sigma22, Yt), color='k')
        plt.xlim([1.5 * min(sigma11, -Xc), 1.5 * max(sigma11, Xt)])
        plt.ylim([1.5 * min(sigma22, -Yc), 1.5 * max(sigma22, Yt)])
        plt.show()
    

    def tsai_hill(self, sigma, Xt, Yt, Xc, Yc, S, plot=False):
        """
        Evaluate the Tsai-Hill failure criterion.

        Parameters:
        - sigma: Stress tensor
        - Xt, Yt: Tensile strengths in fiber and transverse directions
        - Xc, Yc: Compressive strengths in fiber and transverse directions
        - S: Shear strength
        - plot: Boolean flag to enable plotting (default: False)

        Returns:
        - fail: Boolean flag indicating failure (True/False)
        - crit: Tsai-Hill criterion value
        """
        sigma_tens = vec2tensor(sigma) if sigma.shape[1] == 1 else sigma
        sigmap = rotate(self.theta, sigma_tens)
        sigma11, sigma22, tau12 = sigmap[0, 0], sigmap[1, 1], sigmap[0, 1]
        X = Xt if sigma11 >= 0 else Xc
        Y = Yt if sigma22 >= 0 else Yc

        crit = sigma11**2 / X**2 - sigma11 * sigma22 / X**2 + sigma22**2 / Y**2 + tau12**2 / S**2
        fail = crit >= 1

        # TODO: Add plotting if required
        return fail, crit

    def hoffman(self, sigma, Xt, Yt, Xc, Yc, S, plot=False):
        """
        Evaluate the Hoffman failure criterion.

        Parameters:
        - sigma: Stress tensor
        - Xt, Yt: Tensile strengths in fiber and transverse directions
        - Xc, Yc: Compressive strengths in fiber and transverse directions
        - S: Shear strength
        - plot: Boolean flag to enable plotting (default: False)

        Returns:
        - fail: Boolean flag indicating failure (True/False)
        - crit: Hoffman criterion value
        """
        sigma_tens = vec2tensor(sigma) if sigma.shape[1] == 1 else sigma
        sigmap = rotate(self.theta, sigma_tens)
        sigma11, sigma22, tau12 = sigmap[0, 0], sigmap[1, 1], sigmap[0, 1]

        crit = (
            sigma11**2 / (Xc * Xt)
            - sigma11 * sigma22 / (Xc * Xt)
            + sigma22**2 / (Yc * Yt)
            - (Xt - Xc) * sigma11 / (Xc * Xt)
            - (Yt - Yc) * sigma22 / (Yc * Yt)
            + tau12**2 / S**2
        )
        fail = crit >= 1

        # TODO: Add plotting if required
        return fail, crit

    def tsai_wu(self, sigma, F1, F2, F11, F22, F66, F12, plot=False):
        """
        Evaluate the Tsai-Wu failure criterion.

        Parameters:
        - sigma: Stress tensor
        - F1, F2, F11, F22, F66, F12: Tsai-Wu interaction coefficients
        - plot: Boolean flag to enable plotting (default: False)

        Returns:
        - fail: Boolean flag indicating failure (True/False)
        - crit: Tsai-Wu criterion value
        """
        sigma_tens = vec2tensor(sigma) if sigma.shape[1] == 1 else sigma
        sigmap = rotate(self.theta, sigma_tens)
        sigma11, sigma22, tau12 = sigmap[0, 0], sigmap[1, 1], sigmap[0, 1]

        crit = F1 * sigma11 + F2 * sigma22 + F11 * sigma11**2 + F22 * sigma22**2 + F66 * tau12**2 + 2 * F12 * sigma11 * sigma22
        fail = crit >= 1

        # TODO: Add plotting if required
        return fail, crit
