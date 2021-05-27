from cauchy_problem import CauchyProblem
from semi_infinite_area import SemiInfiniteArea
from dirichlet_neumann_solver import DirichletNeumannSolver
#from dirichlet_neumann_solver import load_or_generate_exact_data
#from typing import Callable, Sequence
import numpy as np
from os import path

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
        hk_x = lambda x: x[0]+x[1]#np.exp(x[0]-x[1]) #random initial function h0
        hk0 = lambda t: hk_x(self.problem.area.Gamma(t))
        hk = lambda t: hk_x(self.problem.area.Gamma(t))
        for i in range(1, maxiter+1):
            if (verbose_mode): 
                print("\tIteration", i, "started")
            #Step2: solve well-posed problem A (3.1-3.2)
            solver1 = DirichletNeumannSolver(self.problem.f2, hk, self.problem.area)
            mu, alpha = solver1.solve(M)
            print("first", mu, alpha)
            #Step3: gk = uk - f1
            gk_x = lambda x: solver1.get_u_approx(x, mu, alpha)
            gk = lambda t: gk_x(self.problem.area.Gamma(t)) - self.problem.f1(t)
            #Step4: solve well-posed problem B (3.3-3.4)
            solver2 = DirichletNeumannSolver(gk, lambda t: 0, self.problem.area)
            mu, alpha = solver2.solve(M, False)
            print("second", mu, alpha)
            #Step5: reassign hk
            dVnu = lambda t: solver2.get_du_normal_approx(t, mu, alpha)
            hk = lambda t: hk0(t) - i * beta * dVnu(t)
            
        #Final approximation
        solver = DirichletNeumannSolver(self.problem.f2, hk, self.problem.area)
        mu, alpha = solver.solve(M)
        
        return mu, alpha
    
    
def load_or_generate_exact_data_cauchy(g, f, area, M_ex=128):
    # if path.exists('exactCauchy.npy'):
    #     data = np.array([])
    #     with open('exactCauchy.npy', 'rb') as f:
    #         data = np.load(f)
    #         data = data.tolist()

    #     return data[0][:-1], data[0][-1]
    # else:        
        problem = CauchyProblem(f, g, area)
        solver_ex = LandweberCauchySolver(problem)
        mu_ex, alpha_ex = solver_ex.solve(M=M_ex, maxiter=3, verbose_mode=False)

        mu_ex_data = list(mu_ex)
        mu_ex_data.append(alpha_ex)

        # with open('exactCauchy.npy', 'wb') as f:
        #     np.save(f, np.array([mu_ex_data]))

        return mu_ex, alpha_ex

###Testing
def testCauchy():
    r = lambda t: np.math.sqrt(np.cos(t)**2 + 0.25*(np.sin(t)**2))
    area = SemiInfiniteArea(
        D   = lambda q, t: np.array([   r(t)*np.cos(t),   r(t)*np.sin(t) + 1.5     ]),
        dD  = lambda q, t: np.array([   -q*np.sin(t),  q*np.cos(t)     ]),
        d2D = lambda q, t: np.array([   -q*np.cos(t),  -q*np.sin(t)    ]),
    
        qRange = [0.0, 1.0],
        tRange = [0.0, 2*np.pi]
    )
    
    g_x = lambda x: x[0] - 0.1*(x[1] - 1.5)
    g = lambda t: g_x(area.Gamma(t))
    
    f_x = lambda x: x[0]*np.math.exp(-(x[0]**2)) #Why this function?
    f = lambda t: f_x(area.x_inf(t))
    
    #area.plot_boundary()
    
    problem = CauchyProblem(f, g, area)
    solver = LandweberCauchySolver(problem)
    dirichlet_solver = DirichletNeumannSolver(g, f, area)
    
    # generate 'exact' data
    # mu_ex, alpha_ex = load_or_generate_exact_data_cauchy(g, f, area)
    # V = lambda x: dirichlet_solver.get_u_approx(x, mu_ex, alpha_ex, 1, 64)
    # dVnu = lambda t: dirichlet_solver.get_du_normal_approx(t, mu_ex, alpha_ex, 1, 64)
    
    
    x_test = np.array([1.1, 1.5])
    t_test = np.pi / 2
    
    # print(f'\nV({x_test}) = {V(x_test)}\n')
    # print(f'\ndVnu({x_test}) = {dVnu(t_test)}\n')
    
    for i in [4, 8, 16, 32, 64]:
        mu, alpha = solver.solve(M=i, maxiter=1, verbose_mode=True)
        print(f' >>Cauchy> [M={i}] U({x_test}) = {dirichlet_solver.get_u_approx(x_test, mu, alpha, 1, 64)}')
        print(f' >>Cauchy> [M={i}] dUnu({t_test}) = {dirichlet_solver.get_du_normal_approx(t_test, mu, alpha, 1, 64)}')
        
testCauchy()