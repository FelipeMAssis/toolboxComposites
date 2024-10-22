import numpy as np

class SSPolyShapeFun:
    def __init__(self,length,width,m,n):
        """
        Initializes the SSPolyShapeFun class for shape functions using polynomial basis.

        Args:
            length: Length of the plate.
            width: Width of the plate.
            m: Number of modes in the x-direction.
            n: Number of modes in the y-direction.
        """
        self.length = length
        self.width = width
        self.n = n
        self.m = m

    def Nk(self,x,y,i,j,a,b,dx=0,dy=0):
        """
        Computes the shape function or its derivatives.

        Args:
            x: X-coordinate.
            y: Y-coordinate.
            i: Mode number in the x-direction.
            j: Mode number in the y-direction.
            a: Length of the plate.
            b: Width of the plate.
            dx: Derivative order in the x-direction (default 0).
            dy: Derivative order in the y-direction (default 0).

        Returns:
            Value of the shape function or its derivative.
        """
        if dx==0 and dy==0:
            return x*(x-a)*y*(y-b)*x**i*y**j
        elif dx==1 and dy==0:
            return y*(y-b)*y**j*((i+2)*x**(i+1)-a*(i+1)*x**i)
        elif dx==0 and dy==1:
            return x*(x-a)*x**i*((j+2)*y**(j+1)-b*(j+1)*y**j)
        elif dx==2 and dy==0:
            return y*(y-b)*y**j*((i+2)*(i+1)*x**(i)-a*(i+1)*i*x**(i-1))
        elif dx==0 and dy==2:
            return x*(x-a)*x**i*((j+2)*(j+1)*y**(j)-b*(j+1)*j*y**(j-1))
        elif dx==1 and dy==1:
            return ((i+2)*x**(i+1)-a*(i+1)*x**i)*((j+2)*y**(j+1)-b*(j+1)*y**j)
        else:
            raise ValueError('Cannot compute derivative')

    def shape_fun(self,x,y,dx=0,dy=0):
        mvec = np.arange(1,self.m+1,1)
        nvec = np.arange(1,self.n+1,1)
        sf = np.zeros((1,len(mvec)*len(nvec)))
        k = 0
        a = self.length
        b = self.width
        for i in mvec:
            for j in nvec:
                sf[0,k] = self.Nk(x,y,i,j,a,b,dx,dy)
                k += 1
        return sf
    
    def N_ww(self,x,y):
        N_xx = self.shape_fun(x,y,2,0)
        N_yy = self.shape_fun(x,y,0,2)
        N_xy = self.shape_fun(x,y,1,1)
        return np.block(
            [[N_xx],
            [N_yy],
            [N_xy]]
        )


class SineShapeFun:
    def __init__(self,length,width,m,n):
        self.length = length
        self.width = width
        self.n = n
        self.m = m
    
    def Nk(self,x,y,i,j,a,b,dx=0,dy=0):
        if dx==0 and dy==0:
            return np.sin(i*np.pi*x/a)*np.sin(j*np.pi*y/b)
        elif dx==1 and dy==0:
            return (i*np.pi*np.cos((np.pi*i*x)/a)*np.sin((np.pi*j*y)/b))/a
        elif dx==0 and dy==1:
            return (j*np.pi*np.cos((np.pi*j*y)/b)*np.sin((np.pi*i*x)/a))/b
        elif dx==2 and dy==0:
            return -(i**2*np.pi**2*np.sin((np.pi*i*x)/a)*np.sin((np.pi*j*y)/b))/a**2
        elif dx==0 and dy==2:
            return -(j**2*np.pi**2*np.sin((np.pi*i*x)/a)*np.sin((np.pi*j*y)/b))/b**2
        elif dx==1 and dy==1:
            return (i*j*np.pi**2*np.cos((np.pi*i*x)/a)*np.cos((np.pi*j*y)/b))/(a*b)
        else:
            raise ValueError('Cannot compute derivative')
        
    def N_ww(self,x,y):
        N_xx = self.shape_fun(x,y,2,0)
        N_yy = self.shape_fun(x,y,0,2)
        N_xy = self.shape_fun(x,y,1,1)
        return np.block(
            [[N_xx],
            [N_yy],
            [N_xy]]
        )

class CompleteSine(SineShapeFun):
    def __init__(self,length,width,m,n):
        super().__init__(length,width,m,n)
    
    def shape_fun(self,x,y,dx=0,dy=0):
        mvec = np.arange(1,self.m+1,1)
        nvec = np.arange(1,self.n+1,1)
        sf = np.zeros((1,len(mvec)*len(nvec)))
        k = 0
        a = self.length
        b = self.width
        for i in mvec:
            for j in nvec:
                sf[0,k] = self.Nk(x,y,i,j,a,b,dx,dy)
                k += 1
        return sf

class IncompleteSine(SineShapeFun):
    def __init__(self,length,width,m,n):
        super().__init__(length,width,m,n)
        
    def shape_fun(self,x,y,dx=0,dy=0):
        mvec = np.arange(1,self.m+1,2)
        nvec = np.arange(1,self.n+1,2)
        sf = np.zeros((1,len(mvec)*len(nvec)))
        k = 0
        a = self.length
        b = self.width
        for i in mvec:
            for j in nvec:
                sf[0,k] = self.Nk(x,y,i,j,a,b,dx,dy)
                k += 1
        return sf