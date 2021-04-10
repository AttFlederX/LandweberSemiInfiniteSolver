from semi_infinite_area import SemiInfiniteArea
import numpy as np

class CauchyProblem:
    def __init__(self,
                f1,
                f2,
                area: SemiInfiniteArea,
                u_ex = None):
        self.f1 = f1
        self.f2 = f2
        self.area = area

        self.u_ex = u_ex