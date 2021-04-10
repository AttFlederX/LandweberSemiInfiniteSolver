import numpy as np

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

        self.x1         = lambda t: (self.Gamma(t))[0]
        self.x2         = lambda t: (self.Gamma(t))[1]
        self.dx1        = lambda t: (self.dGamma(t))[0]
        self.dx2        = lambda t: (self.dGamma(t))[1]
        self.d2x1       = lambda t: (self.d2Gamma(t))[0]
        self.d2x2       = lambda t: (self.d2Gamma(t))[1]
        