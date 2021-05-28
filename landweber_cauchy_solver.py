from cauchy_problem import CauchyProblem
from semi_infinite_area import SemiInfiniteArea
from dirichlet_neumann_solver import DirichletNeumannSolver
#from dirichlet_neumann_solver import load_or_generate_exact_data
#from typing import Callable, Sequence
import numpy as np
from os import path
import time

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
        if verbose_mode:
            x_test = np.array([1.1, 1.5])
            t_test = np.pi / 2
        #Step 1: init
        hk_x = lambda x: np.exp(x[0]-x[1]) #random initial function h0
        hk0 = lambda t: hk_x(self.problem.area.Gamma(t))
        hk = lambda t: hk_x(self.problem.area.Gamma(t))

        mu_u = np.array([])
        alpha_u = np.array([])

        mu_v = np.array([])
        alpha_v = np.array([])

        #Step2: solve well-posed problem A (3.1-3.2)
        solver1 = DirichletNeumannSolver(hk, self.problem.f2, self.problem.area)
        mu_u, alpha_u = solver1.solve(M)
        
        for i in range(1, maxiter+1):
            if verbose_mode: 
                print("\tIteration", i, "started")
                start = time.time()


            #Step3: gk = uk - f1
            uk_x = lambda x: solver1.get_u_approx(x, mu_u, alpha_u)
            gk = lambda t: uk_x(self.problem.area.Gamma(t)) - self.problem.f1(t)

            #Step4: solve well-posed problem B (3.3-3.4)
            solver2 = DirichletNeumannSolver(lambda t: 0, gk, self.problem.area)
            mu_v, alpha_v = solver2.solve(M, False)
            #print("second", mu_v, alpha_v)

            #Step5: reassign hk
            dVnu = lambda t: solver2.get_du_normal_approx(t, mu_v, alpha_v)
            hk = lambda t: hk0(t) - i * beta * dVnu(t) \
                * np.power(0.2, i) #Costyl' to make method convergent
            
            #Step2: solve well-posed problem A (3.1-3.2)
            solver1 = DirichletNeumannSolver(hk, self.problem.f2, self.problem.area)
            mu_u, alpha_u = solver1.solve(M)
            #print("first", mu_u, alpha_u)
            if verbose_mode:
                print(f'\t\t U({x_test}) = {solver1.get_u_approx(x_test, mu_u, alpha_u)}')
                print(f'\t\t dUnu({t_test}) = {solver1.get_du_normal_approx(t_test, mu_u, alpha_u)}')
                print("\tIteration", i, "took", np.round(time.time() - start), "sec.")
            
        #Final approximation
        #solver = DirichletNeumannSolver(hk, self.problem.f2, self.problem.area)
        #mu, alpha = solver.solve(M)
        
        return mu_u, alpha_u
    
    
def load_or_generate_exact_data_cauchy(g, f, area, M_ex=128):
    if path.exists('exactCauchy.npy'):
        data = np.array([])
        with open('exactCauchy.npy', 'rb') as f:
            data = np.load(f)
            data = data.tolist()

        return data[0][:-1], data[0][-1]
    else:        
        problem = CauchyProblem(g, f, area)
        solver_ex = LandweberCauchySolver(problem)
        mu_ex, alpha_ex = solver_ex.solve(M=M_ex, beta=0.5, maxiter=7, verbose_mode=True)

        mu_ex_data = list(mu_ex)
        mu_ex_data.append(alpha_ex)

        with open('exactCauchy.npy', 'wb') as f:
            np.save(f, np.array([mu_ex_data]))

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
    
    f_x = lambda x: x[0]*np.math.exp(-(x[0]**2))
    f = lambda t: f_x(area.x_inf(t))
    
    #area.plot_boundary()
    
    problem = CauchyProblem(g, f, area)
    solver = LandweberCauchySolver(problem)
    dirichlet_solver = DirichletNeumannSolver(g, f, area)
    
    # generate 'exact' data
    mu_ex, alpha_ex = load_or_generate_exact_data_cauchy(g, f, area, 24)
    V = lambda x: dirichlet_solver.get_u_approx(x, mu_ex, alpha_ex)
    dVnu = lambda t: dirichlet_solver.get_du_normal_approx(t, mu_ex, alpha_ex)
    
    
    x_test = np.array([1.1, 1.5])
    t_test = np.pi / 2
    
    print(f'\nV({x_test}) = {V(x_test)}\n')
    print(f'\ndVnu({x_test}) = {dVnu(t_test)}\n')
    
    for i in [6, 8, 12]:#16, 32, 64]:
        start = time.time()
        mu, alpha = solver.solve(M=i, beta=0.5, maxiter=7, verbose_mode=True)
        print(f' >>Cauchy> [M={i}] U({x_test}) = {dirichlet_solver.get_u_approx(x_test, mu, alpha, 1, 64)}')
        print(f' >>Cauchy> [M={i}] dUnu({t_test}) = {dirichlet_solver.get_du_normal_approx(t_test, mu, alpha, 1, 64)}')
        print("------------------", np.round(time.time() - start), "sec. ------------------")
        
#testCauchy()