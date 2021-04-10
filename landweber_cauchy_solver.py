from cauchy_problem import CauchyProblem
from semi_infinite_area import SemiInfiniteArea
from typing import Callable, Sequence
import numpy as np

class LandweberCauchySolver:
    def __init__(self, 
                problem: CauchyProblem):
        self.problem = problem