
# Solução de Navier não validada

'''import matplotlib.pyplot as plt
from toolbox.results import PlateNavierResults

class PlateNavierAnalysis:
    def __init__(self, plate, loads, m_range=range(5,10), n_range=range(5,10), divx=50, divy=50):
        self.plate = plate
        self.loads = loads
        self.m_range = m_range
        self.n_range = n_range
        self.divx = divx
        self.divy = divy
        self.results = []
        for m in m_range:
            results_line = []
            for n in n_range:
                results_line.append(PlateNavierResults(plate,loads,m,n,divx,divy))
            self.results.append(results_line)
        
    def plot_convergence(self,x,y,laminate_var=None,lamina_var=None,lamina=None,top=None):
        plt.figure()'''

