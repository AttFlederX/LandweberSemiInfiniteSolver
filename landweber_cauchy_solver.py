from cauchy_problem import CauchyProblem
from semi_infinite_area import SemiInfiniteArea
from dirichlet_neumann_solver import DirichletNeumannSolver
from typing import Callable, Sequence
import numpy as np

class LandweberCauchySolver:
    def __init__(self, 
                problem: CauchyProblem):
        self.problem = problem
        
    ### Runtime functions
    def solve(self, beta=0.2, M = 4, maxiter = 5, verbose_mode = False):
        '''
            Solves the Cauchy problem
 
            beta: iterative coeficient > 0, in the article it is lowercase gamma
            M: number of quadrature nodes
            maxiter: desired number of iterations
            
        '''
        #Step 1: init
        hk = lambda x: np.exp(x[0]-x[1]) #random initial function h0
        for i in range(maxiter):
            #Step2: solve well-posed problem A (3.1-3.2)
            solver = DirichletNeumannSolver(self.problem.f2, hk, self.problem.area)
            mu, alpha = solver.solve(M)
            #Step3: gk = uk - f1
            gk_x = lambda x: solver.get_u_approx(x, mu, alpha) - self.problem.f1(x)
            gk = lambda t: gk_x(self.problem.area.Gamma(t))
            #Step4: solve well-posed problem B (3.3-3.4)
            solver = DirichletNeumannSolver(gk, lambda x: 0, self.problem.area)
            mu, alpha = solver.solve(M)
            #Step5: reassign hk
            #TODO
            
        
