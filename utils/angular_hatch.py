import numpy as np
import matplotlib

class AngularHatch(matplotlib.hatch.HatchPatternBase):
    def __init__(self, hatch, density):
        self.num_lines=0
        self.num_vertices=0
        if hatch[0] == "{":
            h = hatch.strip("{}").split("}{")
            angle = np.deg2rad(float(h[0])-45)
            d = float(h[1])
            self.num_lines = int(density*d)
            self.num_vertices = (self.num_lines + 1) * 2
            self.R = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])

    def set_vertices_and_codes(self, vertices, codes):

        steps = np.linspace(-0.5, 0.5, self.num_lines + 1, True)

        vertices[0::2, 0] = 0.0 + steps
        vertices[0::2, 1] = 0.0 - steps
        vertices[1::2, 0] = 1.0 + steps
        vertices[1::2, 1] = 1.0 - steps
        codes[0::2] = matplotlib.path.Path.MOVETO
        codes[1::2] = matplotlib.path.Path.LINETO
        vertices[:,:] = np.dot((vertices-0.5),self.R)+0.5