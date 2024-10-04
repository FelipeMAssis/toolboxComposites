import numpy as np
from toolbox.material import Material
from toolbox.lamina import Lamina
from toolbox.laminate import Laminate

def test_lamina():
    mat = Material(
        129500,
        9370,
        5240,
        0.38,
        1e-6,
        1e-5,
        0
    )

    lam = Lamina(
        mat,
        0.19,
        45
    )

    sigma = np.array(
        [[100],
        [0],
        [0]]
    )

    eps, epsp = lam.calc_eps(sigma, 0)
    sigmar, sigmapr = lam.calc_sigma(eps, 0)

    print(mat.alpha)
    print(lam.alphabar)

def test_laminate():
    mat = Material(
        129500,
        9370,
        5240,
        0.38,
        1e-6,
        1e-5,
        0
    )
    theta = [0,90,0,90,90,0,90,0]
    t = [0.19]*len(theta)
    layup = [Lamina(mat, ti, thetai) for ti,thetai in zip(t,theta)]
    laminate = Laminate(layup)
    
    N = np.array([
        [1000],
        [1000],
        [0]
    ])
    M = np.array([
        [10],
        [0],
        [10]
    ])
    eps0, kappa = laminate.forces2def(N,M,10)

    print(eps0)
    print(kappa)

    N, M = laminate.def2forces(eps0, kappa, 10)

    print(N)
    print(M)


test_laminate()
#test_lamina()

