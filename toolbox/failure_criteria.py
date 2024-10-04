import numpy as np
import matplotlib.pyplot as plt

def rotate_plane_stress(sigma, theta, deg=True):
    if deg:
        theta = np.pi*theta/180
    c = np.cos(theta)
    s = np.sin(theta)
    T = np.array(
        [[c, -s],
         [s, c]]
    )
    return T.transpose() @ sigma @ T

def max_stress(sigma, theta, Xt, Yt, Xc, Yc, deg=True, plot=False):
    sigmap = rotate_plane_stress(sigma, theta)
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
    
def max_strain(sigma, E11, E22, v12, theta, Xt, Yt, Xc, Yc, deg=True, plot=False):
    v21 = E22*v12/E11
    sigmap = rotate_plane_stress(sigma, theta)
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

def tsai_hill(sigma, theta, Xt, Yt, Xc, Yc, S, deg=True, plot=False):
    sigmap = rotate_plane_stress(sigma, theta)
    sigma11 = sigmap[0,0]
    sigma22 = sigmap[1,1]
    tau12 = sigmap[0,1]
    X = Xt if sigma11>=0 else Xc
    Y = Yt if sigma11>=0 else Yc
    crit = sigma11**2/X**2 - sigma11*sigma22/X**2 + sigma22**2/Y**2 + tau12**2/S**2
    fail = crit >= 1
    return fail, crit

def hoffman(sigma, theta, Xt, Yt, Xc, Yc, S, deg=True, plot=False):
    sigmap = rotate_plane_stress(sigma, theta)
    sigma11 = sigmap[0,0]
    sigma22 = sigmap[1,1]
    tau12 = sigmap[0,1]
    crit = sigma11**2/(Xc*Xt) - sigma11*sigma22/(Xc*Xt) + sigma22**2/(Yc*Yt) - (Xt-Xc)*sigma11/(Xc*Xt) - (Yt-Yc)*sigma22/(Yc*Yt) + tau12**2/S**2
    fail = crit >= 1
    return fail, crit

def tsai_wu(sigma, theta, F1, F2, F11, F22, F66, F12, deg=True, plot=False):
    sigmap = rotate_plane_stress(sigma, theta)
    sigma11 = sigmap[0,0]
    sigma22 = sigmap[1,1]
    tau12 = sigmap[0,1]
    crit = F1*sigma11 + F2*sigma22 + F11*sigma11**2 + F22*sigma22**2 + F66*tau12**2 + 2*F12*sigma11*sigma22
    fail = crit >= 1
    return fail, crit