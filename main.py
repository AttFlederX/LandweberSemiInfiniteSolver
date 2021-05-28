from cauchy_problem import CauchyProblem
from semi_infinite_area import SemiInfiniteArea
from dirichlet_neumann_solver import DirichletNeumannSolver
from landweber_cauchy_solver import LandweberCauchySolver
import numpy as np
from os import path
import time
import matplotlib.pyplot as plt
    
    
def load_or_generate_exact_data2(g, f, area, M_ex=128):
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

def testCauchy2():
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
    
    area.plot_boundary()
    
    problem = CauchyProblem(g, f, area)
    solver = LandweberCauchySolver(problem)
    dirichlet_solver = DirichletNeumannSolver(g, f, area)
    
    # generate 'exact' data
    mu_ex, alpha_ex = load_or_generate_exact_data2(g, f, area, 24)
    V = lambda x: dirichlet_solver.get_u_approx(x, mu_ex, alpha_ex)
    dVnu = lambda t: dirichlet_solver.get_du_normal_approx(t, mu_ex, alpha_ex)
    
    x_test = np.array([1.1, 1.5])
    t_test = np.pi / 2
    
    print(f'\nV({x_test}) = {V(x_test)}\n')
    print(f'\ndVnu({x_test}) = {dVnu(t_test)}\n')
    
    for i in [8]:#[6, 8, 12]:
        start = time.time()
        mu, alpha = solver.solve(M=i, beta=0.51, maxiter=6, verbose_mode=True)
        print(f' >>Cauchy> [M={i}] U({x_test}) = {dirichlet_solver.get_u_approx(x_test, mu, alpha)}')
        print(f' >>Cauchy> [M={i}] dUnu({t_test}) = {dirichlet_solver.get_du_normal_approx(t_test, mu, alpha)}')
        print("------------------", np.round(time.time() - start), "sec. ------------------")

    pts = dirichlet_solver.area.get_random_points(100, qRange = [0.0, 0.15], tRange = [0, 2*np.pi])
    #pts2 = dirichlet_solver.area.get_random_points(50, qRange = [0.0, 0.15], tRange = [(5 * np.pi) / 6, (13 * np.pi) / 6])
    #print(pts)
    res = []
    expected = []
    #mu, alpha = solver.solve(M=4, maxiter=2, verbose_mode=True)
    for pt in pts:
        res.append(dirichlet_solver.get_u_approx(pt, mu, alpha))
        expected.append(dirichlet_solver.get_u_approx(pt, mu_ex, alpha_ex))
   
    return res, pts, expected
#-------------------------------------------
z, pts_res, expec = testCauchy2()

x = []
y = []
for pt in pts_res:
    x.append(pt[0])
    y.append(pt[1])
#print(len(x))
#print(len(y))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x, y, z, marker='o')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Наближені значення')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x, y, expec, marker='o')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Точні значення')
plt.show()