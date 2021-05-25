import numpy as np
import matplotlib.pyplot as plt

class SemiInfiniteArea:
    def __init__(self,
                D,
                dD,
                d2D,
                qRange,
                tRange):

        self.D          = D

        self.q_a        = qRange[0]
        self.q_b        = qRange[1]
        self.t_a        = tRange[0]
        self.t_b        = tRange[1]

        self.Gamma      = lambda t: D(self.q_b, t)
        self.dGamma     = lambda t: dD(self.q_b, t)
        self.d2Gamma    = lambda t: d2D(self.q_b, t)

        self.x_inf      = lambda t: [t, 0]

        self.x1         = lambda t: (self.Gamma(t))[0]
        self.x2         = lambda t: (self.Gamma(t))[1]
        self.dx1        = lambda t: (self.dGamma(t))[0]
        self.dx2        = lambda t: (self.dGamma(t))[1]
        self.d2x1       = lambda t: (self.d2Gamma(t))[0]
        self.d2x2       = lambda t: (self.d2Gamma(t))[1]

    def plot_boundary(self, show_plot=True):
        ''' Plots a curve defined by boundary functions '''
        tau = np.linspace(self.t_a, self.t_b)

        x = list(map(self.x1, tau))
        y = list(map(self.x2, tau))

        plt.plot(x, y, label='Г0')

        x = np.linspace(-2, 2)
        y = np.zeros(len(x))

        plt.plot(x, y, label='Г')

        plt.axis('scaled')
        plt.legend()
        
        if show_plot:
            plt.show()

    def normal(self, t):
        ''' Calculates a normal vector from curve Gamma in point t '''
        return np.array([
            self.dx2(t) / np.linalg.norm(self.dGamma(t), 2),
            -self.dx1(t) / np.linalg.norm(self.dGamma(t), 2)
        ])

    def normal_inf(self, t):
        ''' Calculates a normal vector from the infinite line x_inf in point t '''
        return np.array([
            0,
            -1
        ])

    def get_random_points(self, n, qRange = None, tRange = None):
        q_a = self.q_a
        q_b = self.q_b
        if qRange != None:
            q_a = qRange[0]
            q_b = qRange[1]

        t_a = self.t_a
        t_b = self.t_b
        if tRange != None:
            t_a = tRange[0]
            t_b = tRange[1]

        pts = []
        for i in range(n):
            q = q_a + (q_b - q_a)*np.random.random()
            t = t_a + (t_b - t_a)*np.random.random()

            pts.append(self.D(q, t))

        return pts
        
        