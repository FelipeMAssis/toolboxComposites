import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from utils.tensors import vector, tensor, rotate
from utils.angular_hatch import AngularHatch
matplotlib.hatch._hatch_types.append(AngularHatch)
from toolbox.lamina import Lamina
from toolbox.laminate import Laminate
from toolbox.plate import Plate
from toolbox.plate_load import UniformLoadNavier, PointLoadNavier
from toolbox.shape_function import IncompleteSine, CompleteSine, SSPolyShapeFun
from scipy.integrate import quad, dblquad, quad_vec
import scipy as sp

class LaminaResults:
    def __init__(self, lamina, deltaT=0, eps=None, sigma=None, failure_criteria=[]):
        """
        Initialize the LaminaResults class, calculating stress or strain based on the provided values.

        Parameters:
        - lamina: Lamina object
        - deltaT: Temperature difference
        - eps: Strain vector (optional)
        - sigma: Stress vector (optional)
        """
        self.lamina = lamina
        self.deltaT = deltaT
        self.eps = eps
        self.epsp = None
        self.epst = None
        self.sigma = sigma
        self.sigmap = None

        # Calculate missing values
        if self.eps is not None and self.sigma is None:
            self.sigma, self.sigmap, self.epst = lamina.calc_sigma(self.eps, self.deltaT)
            self.epsp = rotate(self.lamina.theta, self.eps)
        elif self.sigma is not None and self.eps is None:
            self.eps, self.epsp, self.epst = lamina.calc_eps(self.sigma, self.deltaT)
            self.sigmap = rotate(self.lamina.theta, self.sigma)
        elif (self.eps is None and self.sigma is None) or (self.eps is not None and self.sigma is not None):
            raise ValueError("Either 'eps' or 'sigma' must be provided, but not both.")

        self.rdict = {
            'eps':self.eps,
            'epsp':self.epsp,
            'epst':self.epst,
            'sigma':self.sigma,
            'sigmap':self.sigmap,
        }

        self.failure_criteria = failure_criteria

        for crit in failure_criteria:
            self.rdict[crit.name] = crit.evaluate(self)[1:]

    def evaluate_failure(self):
        for crit in self.failure_criteria:
            fail,value,strength = crit.evaluate(self)
            print(crit.name)
            print(f'Fail occurs: {fail}')
            print(f'Criterion value: {value:.2E}')
            print(f'Strength (multiplier): {strength:.2f}\n')
    
    def plot_vecs(self, ax, vec, pos1=0, pos2 = 0.1, color='blue'):
        """
        Plot vectors (strain or stress) on the figure.

        Parameters:
        - ax: Matplotlib axis
        - vec: Vector to plot (strain or stress)
        - pos1: Position offset for the arrows in the x direction
        - pos2: Position offset for the arrows in the y direction
        - color: Arrow color
        """
        scalevec = max([abs(e) for e in vec])/2
        sgx = vec[0]!=abs(vec[0])
        sgy = vec[1]!=abs(vec[1])
        plt.arrow(
            1 + float(sgx*abs(float(vec[0]/scalevec))),
            pos1,
            float(vec[0]/scalevec),
            0,
            color=color,
            head_width=0.1,
            head_length=0.2,
            length_includes_head=True
        )
        plt.arrow(
            pos1,
            1 + float(sgy*abs(float(vec[1]/scalevec))),
            0,
            float(vec[1]/scalevec),
            color=color,
            head_width=0.1,
            head_length=0.2,
            length_includes_head = True
        )
        plt.arrow(
            -1 - float(sgx*abs(float(vec[0]/scalevec))),
            pos1,
            -float(vec[0]/scalevec),
            0,
            color=color,
            head_width=0.1,
            head_length=0.2,
            length_includes_head=True
        )
        plt.arrow(
            pos1,
            -1 - float(sgy*abs(float(vec[1]/scalevec))),
            0,
            -float(vec[1]/scalevec),
            color=color,
            head_width=0.1,
            head_length=0.2,
            length_includes_head = True
        )
        plt.arrow(
            1+pos2,
            - float(vec[2]/scalevec)/2,
            0,
            float(vec[2]/scalevec),
            color=color,
            head_width=0.1,
            head_length=0.2,
            length_includes_head = True
        )
        plt.arrow(
            -1-pos2,
            + float(vec[2]/scalevec)/2,
            0,
            -float(vec[2]/scalevec),
            color=color,
            head_width=0.1,
            head_length=0.2,
            length_includes_head = True
        )
        plt.arrow(
            - float(vec[2]/scalevec)/2,
            1+pos2,
            float(vec[2]/scalevec),
            0,
            color=color,
            head_width=0.1,
            head_length=0.2,
            length_includes_head = True
        )
        plt.arrow(
            + float(vec[2]/scalevec)/2,
            -1-pos2,
            -float(vec[2]/scalevec),
            0,
            color=color,
            head_width=0.1,
            head_length=0.2,
            length_includes_head = True
        )
        
    def plot_state(self, alligned=False):
        """
        Plot the state of strain and stress on the lamina, considering its alignment.

        Parameters:
        - aligned: If True, plot in the principal material axes. Otherwise, plot in global axes.
        """
        if alligned:
            eps = self.epsp
            sigma = self.sigmap
            theta = 0
        else:
            theta = self.lamina.theta
            eps = self.eps
            sigma = self.sigma
        fig, ax = plt.subplots()
        hatch_pat = '{'+f'{90-theta}'+'}{4}'
        ax.add_patch(Rectangle((-1, -1), 2, 2, fill=False, hatch = hatch_pat, color='grey'))
        self.plot_vecs(ax,eps,0.1,0.1,'blue')
        self.plot_vecs(ax,sigma,-0.1,0.2,'green')
        handles = [Patch(color='blue',label='eps'),
                   Patch(color='green',label='sigma')]
        fig.legend(handles=handles)
        plt.axis('off')
        plt.gca().set_aspect('equal')
        

class LaminateResults:
    def __init__(self, laminate, deltaT=0, eps0=None, kappa=None, N=None, M=None, failure_criteria=[]):
        """
        Initialize the LaminateResults class to calculate and store results for a composite laminate.

        Parameters:
        - laminate (Laminate): An instance of the Laminate class representing the laminate configuration.
        - deltaT (float, optional): Temperature change applied to the laminate. Default is 0.
        - eps0 (numpy array, optional): Mid-plane strain vector (ε₀). Should be provided if N and M are not specified.
        - kappa (numpy array, optional): Curvature vector (κ). Should be provided if N and M are not specified.
        - N (numpy array, optional): Resultant force vector (N). Should be provided if eps0 and kappa are not specified.
        - M (numpy array, optional): Resultant moment vector (M). Should be provided if eps0 and kappa are not specified.

        Either (eps0, kappa) or (N, M) must be provided, but not both.

        Raises:
        - ValueError: If neither (eps0, kappa) nor (N, M) are provided, or if both are provided simultaneously.

        Attributes:
        - eps0 (numpy array): Mid-plane strain vector calculated or provided.
        - kappa (numpy array): Curvature vector calculated or provided.
        - N (numpy array): Resultant force vector calculated or provided.
        - M (numpy array): Resultant moment vector calculated or provided.
        - Nt (numpy array): Thermal force vector due to temperature change.
        - Mt (numpy array): Thermal moment vector due to temperature change.
        - bot_lamina_results (list of LaminaResults): Results for each lamina at the bottom surface.
        - top_lamina_results (list of LaminaResults): Results for each lamina at the top surface.
        - rdict (dict): Dictionary containing all the calculated results for easy access.
        """
        self.laminate = laminate
        self.deltaT = deltaT
        self.eps0 = eps0
        self.kappa = kappa
        self.N = N
        self.M = M
        self.Nt = None
        self.Mt = None
        self.failure_criteria = failure_criteria

        # Calculate missing values
        if (self.eps0 is not None and self.kappa is not None) and (self.N is None and self.M is None):
            self.N, self.M, self.Nt, self.Mt = laminate.calc_forces(self.eps0, self.kappa, self.deltaT)
        elif (self.eps0 is None and self.kappa is None) and (self.N is not None and self.M is not None):
            self.eps0, self.kappa, self.Nt, self.Mt = laminate.calc_def(self.N, self.M, self.deltaT)
        else:
            raise ValueError("'eps0' and 'kappa' or 'N' and 'M' must be provided.")
        
        epsbot, epstop = self.laminate.lamina_def(self.eps0, self.kappa)
        self.bot_lamina_results = [LaminaResults(lam,self.deltaT,eps,failure_criteria=failure_criteria) for eps,lam in zip(epsbot,self.laminate.layup)]
        self.top_lamina_results = [LaminaResults(lam,self.deltaT,eps,failure_criteria=failure_criteria) for eps,lam in zip(epstop,self.laminate.layup)]

        self.rdict = {
            'eps0':self.eps0,
            'kappa':self.kappa,
            'N':self.N,
            'M':self.M,
            'Nt':self.Nt,
            'Mt':self.Mt,
            'LaminaTop':self.top_lamina_results,
            'LaminaBot':self.bot_lamina_results,
        }

        self.evaluate_failure()
    
    def evaluate_failure(self,show=False):
        for crit in self.failure_criteria:
            max_value = -np.inf
            for i,result in enumerate(self.bot_lamina_results+self.top_lamina_results):
                fail,value,strength = crit.evaluate(result)
                if value > max_value:
                    laminate_fail = fail
                    laminate_strength = strength
                    max_value = value
                    crit_lamina = i if i<=len(self.laminate.layup) else i-len(self.laminate.layup)
                    top_bot = 'top' if i<=len(self.laminate.layup) else 'bot'
            self.rdict[crit.name] = [max_value,1*laminate_fail,crit_lamina,top_bot=='top']
            if show:
                print(crit.name)
                print(f'Fail occurs: {laminate_fail}')
                print(f'Critical lamina: {crit_lamina} @ {top_bot}')
                print(f'Criterion value: {max_value:.2E}')
                print(f'Strength (multiplier): {laminate_strength:.2f}\n')

    def plot_laminate(self):
        """
        Plot a visual representation of the laminate cross-section, showing the layup
        of individual lamina layers, their orientations, and their thicknesses.
        """
        t = self.laminate.thickness
        fig, ax = plt.subplots()
        plt.plot([-t/2,0,0,-t/2],[t/2,t/2,-t/2,-t/2],color='k')
        for lam in self.laminate.layup:
            hatch_pat = '{'+f'{90-lam.theta}'+'}{4}'
            ax.add_patch(Rectangle((-t/2, lam.zbot), t/2, lam.t, fill=False, hatch = hatch_pat, color='grey'))
            ax.add_patch(Rectangle((-t/2, lam.zbot), t/2, lam.t, fill=False, color='k'))
            plt.text(t/20, lam.zbot, f'{lam.zbot:.4f}',verticalalignment='center')
            plt.text(t/20, lam.ztop, f'{lam.ztop:.4f}',verticalalignment='center')
            plt.text(t/20, (lam.zbot + lam.ztop)/2, f'{lam.theta:.0f}°',verticalalignment='center')
        plt.ylim([-t/2,t/2])
        plt.axis('off')
        plt.gca().set_aspect('equal')
        #plt.show()
    
    def plot_state(self):
        """
        Plot the distribution of strains and stresses across the thickness of the laminate,
        visualizing the top and bottom surface values for each lamina.
        
        Two plots are generated:
        - Strain distribution (in blue, red, and green for εₓₓ, εᵧᵧ, and γₓᵧ respectively)
        - Stress distribution (in blue, red, and green for σₓₓ, σᵧᵧ, and τₓᵧ respectively)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2)
        t = self.laminate.thickness
        for i,lam in enumerate(self.laminate.layup):
            ax1.hlines(lam.ztop,0,1, linestyles='-', color='k', linewidths=0.5, transform=ax1.get_yaxis_transform())
            ax1.hlines(lam.zbot,0,1, linestyles='-', color='k', linewidths=0.5, transform=ax1.get_yaxis_transform())
            b, = ax1.plot([self.top_lamina_results[i].eps[0], self.bot_lamina_results[i].eps[0]],[lam.ztop, lam.zbot], color='blue',label='XX')
            r, = ax1.plot([self.top_lamina_results[i].eps[1], self.bot_lamina_results[i].eps[1]],[lam.ztop, lam.zbot], color='red',label='YY')
            g, = ax1.plot([self.top_lamina_results[i].eps[2], self.bot_lamina_results[i].eps[2]],[lam.ztop, lam.zbot], color='green',label='XY')
            ax2.hlines(lam.ztop,0,1, linestyles='-', color='k', linewidths=0.5, transform=ax2.get_yaxis_transform())
            ax2.hlines(lam.zbot,0,1, linestyles='-', color='k', linewidths=0.5, transform=ax2.get_yaxis_transform())
            ax2.plot([self.top_lamina_results[i].sigma[0], self.bot_lamina_results[i].sigma[0]],[lam.ztop, lam.zbot], color='blue')
            ax2.plot([self.top_lamina_results[i].sigma[1], self.bot_lamina_results[i].sigma[1]],[lam.ztop, lam.zbot], color='red')
            ax2.plot([self.top_lamina_results[i].sigma[2], self.bot_lamina_results[i].sigma[2]],[lam.ztop, lam.zbot], color='green')
        ax1.set_ylim([-t/2,t/2])
        ax1.set_yticks([lam.zbot for lam in self.laminate.layup]+[t/2])
        ax2.set_yticks([lam.zbot for lam in self.laminate.layup]+[t/2])
        ax2.set_ylim([-t/2,t/2])
        ax1.vlines(0,0,1, linestyles='-', color='k', linewidths=0.5, transform=ax1.get_xaxis_transform())
        ax2.vlines(0,0,1, linestyles='-', color='k', linewidths=0.5, transform=ax2.get_xaxis_transform())
        ax1.set_xlabel('Strain')
        ax1.set_ylabel('z')
        ax2.set_xlabel('Stress')
        ax2.set_ylabel('z')
        fig.legend(handles=[b,r,g], loc='outside lower center', ncols=3)
        fig.tight_layout()
        plt.ylim([-t/2,t/2])
        #plt.show()

class PlateResults:
    def __init__(self, plate, loads=[], springs=[], holes=[], divx=50, divy=50, failure_criteria=[]):
        self.plate = plate
        self.loads = loads
        self.springs = springs
        self.holes = holes
        self.divx = divx
        self.divy = divy

        self.failure_criteria = failure_criteria

    def plot_deformed(self,xvec,yvec,w,title=None,show_max=False,show_min=False):
        # Plot the deformed shape
        plt.figure()
        plt.contourf(xvec, yvec, w)
        if show_min:
            wmin = float(np.min(w))
            xmin = xvec[np.argmin(w,axis=0)][-1]
            ymin = yvec[np.argmin(w,axis=1)][-1]
            plt.scatter([xmin],[ymin],marker='x',color='magenta')
            plt.text(xmin,ymin,f'{wmin:.2f}',color='magenta')
        if show_max:
            wmax = float(np.max(w))
            xmax = xvec[np.argmax(w,axis=0)][-1]
            ymax = yvec[np.argmax(w,axis=1)][-1]
            plt.scatter([xmax],[ymax],marker='x',color='cyan')
            plt.text(xmax,ymax,f'{wmax:.2f}',color='cyan')
        for hole in self.holes:
            plt.fill(
                [hole.x_bounds[0],hole.x_bounds[1],hole.x_bounds[1],hole.x_bounds[0],hole.x_bounds[0]],
                [hole.y_bounds[0],hole.y_bounds[0],hole.y_bounds[1],hole.y_bounds[1],hole.y_bounds[0]],
                color='white'
                )
        plt.gca().set_aspect('equal')
        plt.colorbar()
        if title is None:
            plt.title('Deformed Shape of the Plate')
        else:
            plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')

    def plot_state(self,laminate_var=None,dir=None,lamina=None,lamina_var=None,mode=0,title=None,show_min=False,show_max=False):
        a, b = self.plate.length, self.plate.width
        xvec = np.linspace(0, a, self.divx)
        yvec = np.linspace(0, b, self.divy)
        field = np.zeros((self.divx, self.divy))
        for i, x in enumerate(xvec):
            for j, y in enumerate(yvec):
                laminate_results = self.get_laminate_results(x,y,mode)
                if lamina_var is None:
                    field[i,j] = laminate_results.rdict[laminate_var][dir]
                else:
                    field[i,j] = laminate_results.rdict[laminate_var][lamina].rdict[lamina_var][dir]
        plt.figure()
        plt.contourf(xvec, yvec, field)
        if show_min:
            fieldmin = float(np.min(field))
            xmin = xvec[np.argmin(field,axis=0)][-1]
            ymin = yvec[np.argmin(field,axis=1)][-1]
            plt.scatter([xmin],[ymin],marker='x',color='magenta')
            plt.text(xmin,ymin,f'{fieldmin:.2e}',color='magenta')
        if show_max:
            fieldmax = float(np.max(field))
            xmax = xvec[np.argmax(field,axis=0)][-1]
            ymax = yvec[np.argmax(field,axis=1)][-1]
            plt.scatter([xmax],[ymax],marker='x',color='cyan')
            plt.text(xmax,ymax,f'{fieldmax:.2e}',color='cyan')
        for hole in self.holes:
            plt.fill(
                [hole.x_bounds[0],hole.x_bounds[1],hole.x_bounds[1],hole.x_bounds[0],hole.x_bounds[0]],
                [hole.y_bounds[0],hole.y_bounds[0],hole.y_bounds[1],hole.y_bounds[1],hole.y_bounds[0]],
                color='white'
                )
        plt.gca().set_aspect('equal')
        plt.colorbar()
        if title is None:
            title = laminate_var + f' {str(dir)}' if lamina_var is None else laminate_var + f' {str(lamina)}' + ' ' + lamina_var + f' {str(dir)}'
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')


class PlateNavierResults(PlateResults):
    """
    Class for analyzing and plotting results of plate under loads using Navier's solution.

    Attributes:
    -----------
    plate : Plate
        Plate object containing the physical and material properties of the plate.
    loads : list
        List of load objects (UniformLoadNavier, PointLoadNavier) acting on the plate.
    m, n : int
        Number of terms in the Fourier series for x and y directions.
    divx, divy : int
        Number of divisions in x and y directions for plotting the load and deformation.

    Methods:
    --------
    plot_load():
        Plots the load distribution on the plate using the Fourier coefficients.
    plot_deformed():
        Plots the deformed shape of the plate.
    """
    def __init__(self, plate, loads=[], springs=[], holes=[], m=10, n=10, divx=50, divy=50, failure_criteria=[]):
        """
        Initializes the PlateNavierResults with the plate and load objects, 
        and computes the Fourier coefficients for the load and displacement.

        Parameters:
        -----------
        plate : Plate
            Plate object with dimensions and material properties.
        loads : list
            List of load objects applied to the plate.
        m, n : int, optional
            Number of Fourier terms in the x and y directions (default is 10).
        divx, divy : int, optional
            Number of divisions in the x and y directions for plotting (default is 100).
        """
        super().__init__(plate, loads, springs, holes, divx, divy, failure_criteria)
        self.m = m
        self.n = n
        # Calculate the Fourier coefficients for the load
        self.amn = sum(load.calc_amn(self.m, self.n) for load in self.loads)
        self.Amn = self.plate.solve_navier(self.amn)
        
    def get_laminate_results(self,x,y,_):
        a, b = self.plate.length, self.plate.width
        kappax = sum(
            - ((im*2-1) * np.pi / a)**2 * self.Amn[im, jn] * np.sin((im*2-1) * np.pi * x / a) * np.sin((jn*2-1) * np.pi * y / b)
            for im in range(self.m)
            for jn in range(self.n)
        )
        kappay = sum(
            - ((jn*2-1) * np.pi / b)**2 * self.Amn[im, jn] * np.sin((im*2-1) * np.pi * x / a) * np.sin((jn*2-1) * np.pi * y / b)
            for im in range(self.m)
            for jn in range(self.n)
        )
        kappaxy = sum(
            2 * ((jn*2-1) * np.pi / b) * ((im*2-1) * np.pi / a) * self.Amn[im, jn] * np.cos((im*2-1) * np.pi * x / a) * np.cos((jn*2-1) * np.pi * y / b)
            for im in range(self.m)
            for jn in range(self.n)
        )
        kappa = np.array(
            [[kappax],
             [kappay],
             [kappaxy]]
        )
        eps0 = np.zeros([3,1])
        return LaminateResults(self.plate.laminate, deltaT=0, eps0=eps0, kappa = kappa, failure_criteria=self.failure_criteria)

    def plot_load(self):
        """
        Plots the load distribution q(x, y) on the plate based on the Fourier coefficients (amn).
        """
        a, b = self.plate.length, self.plate.width
        xvec = np.linspace(0, a, self.divx)
        yvec = np.linspace(0, b, self.divy)
        q = np.zeros((self.divx, self.divy))

        # Compute load distribution using Fourier series
        for i, x in enumerate(xvec):
            for j, y in enumerate(yvec):
                q[i, j] = sum(
                    self.amn[im, jn] * np.sin((im*2-1) * np.pi * x / a) * np.sin((jn*2-1) * np.pi * y / b)
                    for im in range(self.m)
                    for jn in range(self.n)
                )

        qt = np.sum(q)*(xvec[1]-xvec[0])*(yvec[1]-yvec[0])

        # Plot the load distribution
        plt.figure()
        plt.contourf(xvec, yvec, q)
        plt.colorbar()
        plt.title(f'Load Distribution on Plate (total force = {qt:.2f})')
        plt.xlabel('x')
        plt.ylabel('y')
        #plt.show()

    def plot_deformed(self):
        """
        Plots the deformed shape w(x, y) of the plate based on the Fourier coefficients (Amn).
        """
        a, b = self.plate.length, self.plate.width
        xvec = np.linspace(0, a, self.divx)
        yvec = np.linspace(0, b, self.divy)
        w = np.zeros((self.divx, self.divy))

        # Compute deformation using Fourier series
        for i, x in enumerate(xvec):
            for j, y in enumerate(yvec):
                w[i, j] = sum(
                    self.Amn[im, jn] * np.sin((im*2-1) * np.pi * x / a) * np.sin((jn*2-1) * np.pi * y / b)
                    for im in range(self.m)
                    for jn in range(self.n)
                )

        w /= np.pi**4

        super().plot_deformed(xvec,yvec,w)
        #plt.show()

class PlateStaticBendingRR(PlateResults):
    def __init__(self, plate, loads=[], springs=[], holes=[], w_shape_fun='incomplete_sine', m=10, n=10, divx=50, divy=50, failure_criteria=[]):
        super().__init__(plate, loads, springs, holes, divx, divy, failure_criteria)
        self.m = m
        self.n = n
        if w_shape_fun == 'incomplete_sine':
            self.w_shape_fun = IncompleteSine(
                self.plate.length,
                self.plate.width,
                self.m,
                self.n
            )
        if w_shape_fun == 'complete_sine':
            self.w_shape_fun = CompleteSine(
                self.plate.length,
                self.plate.width,
                self.m,
                self.n
            )
        if w_shape_fun == 'poly':
            self.w_shape_fun = SSPolyShapeFun(
                self.plate.length,
                self.plate.width,
                self.m,
                self.n
            )
        self.K = self.calculate_K()+self.calculate_K_springs()
        self.F = self.calculate_F()
        self.q = np.linalg.solve(self.K,self.F)

    def calculate_K(self):
        a = self.plate.length
        b = self.plate.width
        xvec = np.linspace(0,a,self.divx)
        dx = xvec[1]-xvec[0]
        yvec = np.linspace(0,b,self.divy)
        dy = yvec[1]-yvec[0]
        C = self.plate.laminate.D
        Htest = self.w_shape_fun.N_ww(0,0)
        test_mat = (Htest.T @ C @ Htest)
        K = np.zeros(test_mat.shape)
        for i,x in enumerate(xvec[:-1]):
            x = (x+xvec[i+1])/2
            for j,y in enumerate(yvec[:-1]):
                y = (y+yvec[j+1])/2
                H = self.w_shape_fun.N_ww(x,y)
                K += (H.T @ C @ H)*dx*dy
        return K
    
    def calculate_K_springs(self):
        C = self.plate.laminate.D
        Htest = self.w_shape_fun.N_ww(0,0)
        test_mat = (Htest.T @ C @ Htest)
        K = np.zeros(test_mat.shape)
        for spring in self.springs:
            if len(spring.x_bounds)==2:
                xvec = np.linspace(spring.x_bounds[0],spring.x_bounds[1],self.divx)
                dx = xvec[1]-xvec[0]
            else:
                xvec = 2*spring.x_bounds
                dx = 1
            for i,x in enumerate(xvec[:-1]):
                x = (x+xvec[i+1])/2
                if len(spring.y_bounds)==2:
                    yvec = np.linspace(spring.y_bounds[0],spring.y_bounds[1],self.divy)
                    dy = yvec[1]-yvec[0]
                else:
                    yvec = 2*spring.y_bounds
                    dy = 1
                for j,y in enumerate(yvec[:-1]):
                    y = (y+yvec[j+1])/2
                    if spring.direction == 0 and spring.type == 1:
                        N = self.w_shape_fun.shape_fun(x,y,1,0)
                    elif spring.direction == 2 and spring.type == 0:
                        N = self.w_shape_fun.shape_fun(x,y,0,0)
                    else:
                        raise ValueError('Spring type not supported')
                    K += spring.stiffness*(N.T @ N)*dx*dy
        return K
    
    def calculate_F(self):
        Ntest = self.w_shape_fun.shape_fun(0,0)
        F = np.zeros(Ntest.T.shape)
        for load in self.loads:
            if len(load.x_bounds)==2:
                xvec = np.linspace(load.x_bounds[0],load.x_bounds[1],self.divx)
                dx = xvec[1]-xvec[0]
            else:
                xvec = 2*load.x_bounds
                dx = 1
            for i,x in enumerate(xvec[:-1]):
                x = (x+xvec[i+1])/2
                if len(load.y_bounds)==2:
                    yvec = np.linspace(load.y_bounds[0],load.y_bounds[1],self.divy)
                    dy = yvec[1]-yvec[0]
                else:
                    yvec = 2*load.y_bounds
                    dy = 1
                for j,y in enumerate(yvec[:-1]):
                    y = (y+yvec[j+1])/2
                    if load.direction == 2 and load.type == 0:
                        N = self.w_shape_fun.shape_fun(x,y,0,0)
                    else:
                        raise ValueError('Load type not supported')
                    F += load.q*(N.T)*dy*dx
        return F
    
    def get_laminate_results(self,x,y,_):
        kappax = float(self.w_shape_fun.shape_fun(x,y,2,0) @ self.q)
        kappay = float(self.w_shape_fun.shape_fun(x,y,0,2) @ self.q)
        kappaxy = float(self.w_shape_fun.shape_fun(x,y,1,1) @ self.q)
        kappa = np.array(
            [[kappax],
             [kappay],
             [kappaxy]]
        )
        eps0 = np.zeros([3,1])
        return LaminateResults(self.plate.laminate, deltaT=0, eps0=eps0, kappa = kappa, failure_criteria=self.failure_criteria)
    
    def plot_deformed(self,title=None,show_max=False,show_min=False):
        a, b = self.plate.length, self.plate.width
        xvec = np.linspace(0, a, self.divx)
        yvec = np.linspace(0, b, self.divy)
        w = np.zeros((self.divx, self.divy))
        for i,x in enumerate(xvec):
            for j,y in enumerate(yvec):
                w[i,j] = self.w_shape_fun.shape_fun(x,y) @ self.q 
        super().plot_deformed(xvec,yvec,w,title,show_max,show_min)

class PlateRR(PlateResults):
    def __init__(self, plate, holes=[], shape_fun=None, divx=50, divy=50, failure_criteria=[]):
        super().__init__(plate, divx=divx, divy=divy, failure_criteria=failure_criteria)
        self.holes = holes
        if shape_fun is None:
            shape_fun = [
                CompleteSine(self.plate.length, self.plate.width, 2, 2),
                CompleteSine(self.plate.length, self.plate.width, 2, 2),
                CompleteSine(self.plate.length, self.plate.width, 2, 2)
            ]
            self.shape_fun_u = shape_fun[0]
        self.shape_fun_v = shape_fun[1]
        self.shape_fun_w = shape_fun[2]
        self.K = self.calculate_K() - self.calculate_K_holes()
    
    def calculate_K(self):
        a = self.plate.length
        b = self.plate.width
        xvec = np.linspace(0,a,self.divx)
        dx = xvec[1]-xvec[0]
        yvec = np.linspace(0,b,self.divy)
        dy = yvec[1]-yvec[0]
        C = np.block(
                [[self.plate.laminate.A, self.plate.laminate.B],
                 [self.plate.laminate.B, self.plate.laminate.D]]
        )
        len_K = np.sum(
            [m*n for m,n in [
                (self.shape_fun_u.m, self.shape_fun_u.n),
                (self.shape_fun_v.m, self.shape_fun_v.n),
                (self.shape_fun_w.m, self.shape_fun_w.n)
                ]]
        )
        K = np.zeros(2*[len_K])
        shape_Nu = self.shape_fun_u.shape_fun(0,0).shape
        shape_Nv = self.shape_fun_v.shape_fun(0,0).shape
        for i,x in enumerate(xvec[:-1]):
            x = (x+xvec[i+1])/2
            for j,y in enumerate(yvec[:-1]):
                y = (y+yvec[j+1])/2
                N_uv = np.block(
                    [[self.shape_fun_u.shape_fun(x,y,1,0), np.zeros(shape_Nv)],
                     [np.zeros(shape_Nu), self.shape_fun_v.shape_fun(x,y,0,1)],
                     [self.shape_fun_u.shape_fun(x,y,0,1), self.shape_fun_v.shape_fun(x,y,1,0)]]
                )
                N_ww = self.shape_fun_w.N_ww(x,y)
                H = np.block(
                    [[N_uv, np.zeros((N_uv.shape[0],N_ww.shape[1]))],
                     [np.zeros((N_ww.shape[0],N_uv.shape[1])), N_ww]]
                )
                K += (H.T @ C @ H)*dx*dy
        return K
    
    def calculate_K_holes(self):
        C = np.block(
                [[self.plate.laminate.A, self.plate.laminate.B],
                 [self.plate.laminate.B, self.plate.laminate.D]]
        )
        len_K = np.sum(
            [m*n for m,n in [
                (self.shape_fun_u.m, self.shape_fun_u.n),
                (self.shape_fun_v.m, self.shape_fun_v.n),
                (self.shape_fun_w.m, self.shape_fun_w.n)
                ]]
        )
        K = np.zeros(2*[len_K])
        shape_Nu = self.shape_fun_u.shape_fun(0,0).shape
        shape_Nv = self.shape_fun_v.shape_fun(0,0).shape
        for hole in self.holes:
            xvec = np.linspace(hole.x_bounds[0],hole.x_bounds[1],self.divx)
            dx = xvec[1]-xvec[0]
            yvec = np.linspace(hole.y_bounds[0],hole.y_bounds[1],self.divy)
            dy = yvec[1]-yvec[0]
            for i,x in enumerate(xvec[:-1]):
                x = (x+xvec[i+1])/2
                for j,y in enumerate(yvec[:-1]):
                    y = (y+yvec[j+1])/2
                    N_uv = np.block(
                        [[self.shape_fun_u.shape_fun(x,y,1,0), np.zeros(shape_Nv)],
                        [np.zeros(shape_Nu), self.shape_fun_v.shape_fun(x,y,0,1)],
                        [self.shape_fun_u.shape_fun(x,y,0,1), self.shape_fun_v.shape_fun(x,y,1,0)]]
                    )
                    N_ww = self.shape_fun_w.N_ww(x,y)
                    H = np.block(
                        [[N_uv, np.zeros((N_uv.shape[0],N_ww.shape[1]))],
                        [np.zeros((N_ww.shape[0],N_uv.shape[1])), N_ww]]
                    )
                    K += (H.T @ C @ H)*dx*dy
        return K
    
    def get_laminate_results(self,x,y,mode=0):
        shape_Nu = self.shape_fun_u.shape_fun(0,0).shape
        shape_Nv = self.shape_fun_v.shape_fun(0,0).shape
        N_uv = np.block(
            [[self.shape_fun_u.shape_fun(x,y,1,0), np.zeros(shape_Nv)],
                [np.zeros(shape_Nu), self.shape_fun_v.shape_fun(x,y,0,1)],
                [self.shape_fun_u.shape_fun(x,y,0,1), self.shape_fun_v.shape_fun(x,y,1,0)]]
        )
        N_ww = self.shape_fun_w.N_ww(x,y)
        H = np.block(
            [[N_uv, np.zeros((N_uv.shape[0],N_ww.shape[1]))],
                [np.zeros((N_ww.shape[0],N_uv.shape[1])), N_ww]]
        )
        eps0kappa = H @ self.q[:,mode]
        eps0 = np.array([eps0kappa[:3]]).T
        kappa = np.array([eps0kappa[3:]]).T
        return LaminateResults(self.plate.laminate, deltaT=0, eps0=eps0, kappa = kappa, failure_criteria=self.failure_criteria)
    
    def plot_deformed(self, mode, direction,title=None,show_max=False,show_min=False):
        a, b = self.plate.length, self.plate.width
        xvec = np.linspace(0, a, self.divx)
        yvec = np.linspace(0, b, self.divy)
        u = np.zeros((self.divx, self.divy))
        v = np.zeros((self.divx, self.divy))
        w = np.zeros((self.divx, self.divy))
        shape_Nu = self.shape_fun_u.shape_fun(0,0).shape
        shape_Nv = self.shape_fun_v.shape_fun(0,0).shape
        shape_Nw = self.shape_fun_w.shape_fun(0,0).shape
        for i,x in enumerate(xvec):
            for j,y in enumerate(yvec):
                N = np.block(
                    [[self.shape_fun_u.shape_fun(x,y), np.zeros(shape_Nv), np.zeros(shape_Nw)],
                     [np.zeros(shape_Nu), self.shape_fun_v.shape_fun(x,y), np.zeros(shape_Nw)],
                     [np.zeros(shape_Nu), np.zeros(shape_Nv), self.shape_fun_w.shape_fun(x,y)]]
                )
                uvec = N @ self.q[:,mode]
                u[i,j] = uvec[0]
                v[i,j] = uvec[1]
                w[i,j] = uvec[2]
        if direction==0:
            super().plot_deformed(xvec,yvec,u,title,show_max,show_min)
        if direction==1:
            super().plot_deformed(xvec,yvec,v,title,show_max,show_min)
        if direction==2:
            super().plot_deformed(xvec,yvec,w,title,show_max,show_min)
     
class PlateBucklingRR(PlateRR):
    def __init__(self, plate, loads=np.array([1,1,1]).T, holes=[], shape_fun=None, divx=50, divy=50, failure_criteria=[]):
        super().__init__(plate, holes, shape_fun, divx, divy, failure_criteria)
        self.loads = loads
        self.Kg = self.calculate_Kg()
        self.Ncrit, self.q = sp.linalg.eig(self.K, self.Kg)
        p = np.abs(self.Ncrit.argsort())
        self.Ncrit = self.Ncrit[p]
        self.q = self.q[:,p]
    
    def calculate_Kg(self):
        a = self.plate.length
        b = self.plate.width
        xvec = np.linspace(0,a,self.divx)
        dx = xvec[1]-xvec[0]
        yvec = np.linspace(0,b,self.divy)
        dy = yvec[1]-yvec[0]
        shape_Nu = self.shape_fun_u.shape_fun(0,0).shape
        shape_Nv = self.shape_fun_v.shape_fun(0,0).shape
        shape_Nw = self.shape_fun_w.shape_fun(0,0).shape
        len_K = np.sum(
            [m*n for m,n in [
                (self.shape_fun_u.m, self.shape_fun_u.n),
                (self.shape_fun_v.m, self.shape_fun_v.n),
                (self.shape_fun_w.m, self.shape_fun_w.n)
                ]]
        )
        Kg = np.zeros(2*[len_K])
        for i,x in enumerate(xvec[:-1]):
            x = (x+xvec[i+1])/2
            for j,y in enumerate(yvec[:-1]):
                y = (y+yvec[j+1])/2
                Nw_x = self.shape_fun_w.shape_fun(x,y,1,0)
                Nnx = np.block(
                    [[np.zeros(shape_Nu), np.zeros(shape_Nv), np.zeros(shape_Nw)],
                     [np.zeros(shape_Nu), np.zeros(shape_Nv), np.zeros(shape_Nw)],
                     [np.zeros(shape_Nu), np.zeros(shape_Nv), Nw_x]]
                )
                Kgx = Nnx.T @ Nnx
                Nw_y = self.shape_fun_w.shape_fun(x,y,0,1)
                Nny = np.block(
                    [[np.zeros(shape_Nu), np.zeros(shape_Nv), np.zeros(shape_Nw)],
                     [np.zeros(shape_Nu), np.zeros(shape_Nv), np.zeros(shape_Nw)],
                     [np.zeros(shape_Nu), np.zeros(shape_Nv), Nw_y]]
                )
                Kgy = Nny.T @ Nny
                Nw_xy = self.shape_fun_w.shape_fun(x,y,1,1)
                Nnxy = np.block(
                    [[np.zeros(shape_Nu), np.zeros(shape_Nv), np.zeros(shape_Nw)],
                     [np.zeros(shape_Nu), np.zeros(shape_Nv), np.zeros(shape_Nw)],
                     [np.zeros(shape_Nu), np.zeros(shape_Nv), Nw_xy]]
                )
                Kgxy = Nnxy.T @ Nnxy
                Kg += (self.loads[0]*Kgx + self.loads[1]*Kgy + 2*self.loads[2]*Kgxy)*dx*dy
        return Kg
                
class PlateModalRR(PlateRR):
    def __init__(self, plate, holes=[], shape_fun=None, divx=50, divy=50, failure_criteria=[]):
        super().__init__(plate, holes, shape_fun, divx, divy, failure_criteria)
        self.M = self.calculate_M() - self.calculate_M_holes()
        w2, self.q = sp.linalg.eig(self.K, self.M)
        self.omega = np.sqrt(w2)
        p = np.abs(self.omega.argsort())
        self.omega = self.omega[p]
        self.q = self.q[:,p]


    def calculate_M(self):
        rho = self.plate.laminate.layup[0].material.rho
        h = self.plate.laminate.thickness
        a = self.plate.length
        b = self.plate.width
        xvec = np.linspace(0,a,self.divx)
        dx = xvec[1]-xvec[0]
        yvec = np.linspace(0,b,self.divy)
        dy = yvec[1]-yvec[0]
        I0 = np.zeros([5,5])
        I0[0,0] = rho*h
        I0[1,1] = rho*h
        I0[2,2] = rho*h
        I0[3,3] = rho*h**3/12
        I0[4,4] = rho*h**3/12
        len_M = np.sum(
            [m*n for m,n in [
                (self.shape_fun_u.m, self.shape_fun_u.n),
                (self.shape_fun_v.m, self.shape_fun_v.n),
                (self.shape_fun_w.m, self.shape_fun_w.n)
                ]]
        )
        M = np.zeros(2*[len_M])
        shape_Nu = self.shape_fun_u.shape_fun(0,0).shape
        shape_Nv = self.shape_fun_v.shape_fun(0,0).shape
        shape_Nw = self.shape_fun_w.shape_fun(0,0).shape
        for i,x in enumerate(xvec[:-1]):
            x = (x+xvec[i+1])/2
            for j,y in enumerate(yvec[:-1]):
                y = (y+yvec[j+1])/2
                B = np.block(
                    [[self.shape_fun_u.shape_fun(x,y), np.zeros(shape_Nv), np.zeros(shape_Nw)],
                     [np.zeros(shape_Nu), self.shape_fun_v.shape_fun(x,y), np.zeros(shape_Nw)],
                     [np.zeros(shape_Nu), np.zeros(shape_Nv), self.shape_fun_w.shape_fun(x,y)],
                     [np.zeros(shape_Nu), np.zeros(shape_Nv), self.shape_fun_w.shape_fun(x,y,1,0)],
                     [np.zeros(shape_Nu), np.zeros(shape_Nv), self.shape_fun_w.shape_fun(x,y,0,1)]]
                )
                M += (B.T @ I0 @ B)*dx*dy
        return M
    
    def calculate_M_holes(self):
        rho = self.plate.laminate.layup[0].material.rho
        h = self.plate.laminate.thickness
        I0 = np.zeros([5,5])
        I0[0,0] = rho*h
        I0[1,1] = rho*h
        I0[2,2] = rho*h
        I0[3,3] = rho*h**3/12
        I0[4,4] = rho*h**3/12
        len_M = np.sum(
            [m*n for m,n in [
                (self.shape_fun_u.m, self.shape_fun_u.n),
                (self.shape_fun_v.m, self.shape_fun_v.n),
                (self.shape_fun_w.m, self.shape_fun_w.n)
                ]]
        )
        M = np.zeros(2*[len_M])
        shape_Nu = self.shape_fun_u.shape_fun(0,0).shape
        shape_Nv = self.shape_fun_v.shape_fun(0,0).shape
        shape_Nw = self.shape_fun_w.shape_fun(0,0).shape
        for hole in self.holes:
            xvec = np.linspace(hole.x_bounds[0],hole.x_bounds[1],self.divx)
            dx = xvec[1]-xvec[0]
            yvec = np.linspace(hole.y_bounds[0],hole.y_bounds[1],self.divy)
            dy = yvec[1]-yvec[0]
            for i,x in enumerate(xvec[:-1]):
                x = (x+xvec[i+1])/2
                for j,y in enumerate(yvec[:-1]):
                    y = (y+yvec[j+1])/2
                    B = np.block(
                        [[self.shape_fun_u.shape_fun(x,y), np.zeros(shape_Nv), np.zeros(shape_Nw)],
                        [np.zeros(shape_Nu), self.shape_fun_v.shape_fun(x,y), np.zeros(shape_Nw)],
                        [np.zeros(shape_Nu), np.zeros(shape_Nv), self.shape_fun_w.shape_fun(x,y)],
                        [np.zeros(shape_Nu), np.zeros(shape_Nv), self.shape_fun_w.shape_fun(x,y,1,0)],
                        [np.zeros(shape_Nu), np.zeros(shape_Nv), self.shape_fun_w.shape_fun(x,y,0,1)]]
                    )
                    M += (B.T @ I0 @ B)*dx*dy
        return M


