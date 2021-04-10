from typing import Callable, Sequence
from semi_infinite_area import SemiInfiniteArea
import numpy as np

class DirichletNeumannSolver:
    def __init__(self,
                h,
                f,
                area: SemiInfiniteArea,
                u_ex = None):
        self.h = h
        self.f = f
        self.area = area

        self.u_ex = u_ex