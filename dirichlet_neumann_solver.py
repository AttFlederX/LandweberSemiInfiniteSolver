from semi_infinite_area import SemiInfiniteArea
from util import print_matrix

from os import path
import numpy as np
import matplotlib.pyplot as plt

class DirichletNeumannSolver:
    def __init__(self,
                g,
                f,
                area: SemiInfiniteArea,
                u_ex = None):
        self.g = g
        self.f = f
        self.area = area

        self.u_ex = u_ex

    def __N(self, x, y):
        y_star = [y[0], -y[1]]
        return np.math.log(1 / (np.linalg.norm(x - y) * np.linalg.norm(x - y_star)))

    def __R_j(self, i, j, t, M):
        weigh_sum = 0
        for m in range(1, M):
            weigh_sum += (1/m) * np.cos(m * (t[i] - t[j]))
            
        return -(1/(2*M)) * (1 + 2*weigh_sum + (np.cos(t[i] - t[j]) / M))

    def __H_1(self, t, tau):
        if abs(t - tau) > 10 ** -9:
            return self.__N(self.area.Gamma(t), self.area.Gamma(tau)) + \
                (1/2) * np.math.log((4/np.math.e) * (np.sin((t - tau) / 2) ** 2))
        else:
            return -np.math.log(2 * np.linalg.norm(self.area.dGamma(t)) * self.area.x2(t)) - 0.5

    def __get_mtx_coef(self, i, j, t, M):
        return -(1/2) * self.__R_j(i, j, t, M) + (1/(2*M)) * self.__H_1(t[i], t[j])

    def __w(self, t_j, c, M_1):
        inf_sum = 0
        h_inf = c / np.math.sqrt(M_1)

        for i in range(-M_1, M_1+1):
            inf_sum += self.f(i * h_inf) * self.__N(self.area.Gamma(t_j), self.area.x_inf(i * h_inf))

        return self.g(t_j) - h_inf * inf_sum

    def __H_2(self, t, tau):
        x = self.area.Gamma(t)
        y = self.area.Gamma(tau)
        y_star = [y[0], -y[1]]

        if abs(t - tau) > 10 ** -9:
            return (np.dot(y - x, self.area.normal(t)) / (np.linalg.norm(x - y)**2)) - \
                (np.dot(x - y_star, self.area.normal(t)) / (np.linalg.norm(x - y_star)**2))
        else:
            return (np.dot(self.area.d2Gamma(t), self.area.normal(t)) / (2*(np.linalg.norm(self.area.dGamma(t))**2))) + \
                (self.area.dx1(t) / (2*self.area.x2(t)*np.linalg.norm(self.area.dGamma(t))))

    def __H_3(self, t, tau):
        x = self.area.Gamma(t)
        y_inf = self.area.x_inf(tau)

        return -2 * (np.dot(x - y_inf, self.area.normal(t)) / (np.linalg.norm(x - y_inf)**2))

    ### Runtime functions
    def solve(self, M = 4, verbose_mode = False):
        '''
            Solves the Dirichlet-Neumann problem with the instance's parameter values
 
            n: number of quadrature nodes
        '''
        mu_a = np.zeros(2*M + 1)
        t = np.linspace(self.area.t_a, self.area.t_b, 2*M, False)
        c = 1
        M_1 = 64

        # setup linear system
        mtx_A = np.zeros((2*M + 1, 2*M + 1))
        vct_w = np.zeros(2*M + 1)

        for i in range(2*M + 1):
            if i == 2*M: # boundary condition
                for j in range(2*M):
                    mtx_A[i, j] = 1
                mtx_A[i, 2*M] = 0 # alpha

                continue

            for j in range(2*M):
                mtx_A[i, j] = self.__get_mtx_coef(i, j, t, M)

            mtx_A[i, 2*M] = 1 # alpha
            vct_w[i] = self.__w(t[i], c, M_1)

        if verbose_mode:
            print("\nmtx_A=\n")
            print_matrix(mtx_A)
            print(f"vct_w={vct_w}")

        mu_a = np.linalg.solve(mtx_A, vct_w)

        alpha = mu_a[-1]
        mu = mu_a[:2*M]

        if verbose_mode:
            print(f"\nmu_approx={mu}")
            print(f"alpha={alpha}\n")

        return mu, alpha


    ## Testing functions

    def get_u_approx(self, x, mu, alpha, c = 1, M_1 = 64):
        '''
            Calculates the approximate value of the function using given densities
        '''
        M = int(len(mu) / 2)
        t = np.linspace(self.area.t_a, self.area.t_b, 2*M, False)

        quad_sum = 0
        for j in range(2*M):
            quad_sum += mu[j] * self.__N(x, self.area.Gamma(t[j]))

        inf_sum = 0
        h_inf = c / np.math.sqrt(M_1)
        for i in range(-M_1, M_1+1):
            inf_sum += self.f(i * h_inf) * self.__N(x, self.area.x_inf(i * h_inf))

        return (quad_sum / (2*M)) + (h_inf * inf_sum) + alpha

    def get_du_normal_approx(self, t, mu, alpha, c, M_1):
        M = int(len(mu) / 2)
        T = np.linspace(self.area.t_a, self.area.t_b, 2*M, False).tolist()

        i = T.index(t)

        quad_sum = 0
        for j in range(2*M):
            quad_sum += mu[j] * self.__H_2(t, T[j])

        inf_sum = 0
        h_inf = c / np.math.sqrt(M_1)
        for k in range(-M_1, M_1+1):
            inf_sum += self.f(k * h_inf) * self.__H_3(t, k * h_inf)

        return (-mu[i] / (2*(np.linalg.norm(self.area.dGamma(t))**2))) + \
            (quad_sum / (2*M)) + \
            (h_inf * inf_sum)


    def compute_error(self, mu_approx, n_pts, verbose_mode = False, latex_mode = False):
        '''
            Computes the approximation error relative to KDE solution given the approximate density values in the specified set of points

            mu_approx: approximated density values from the solve method, 
            q_max: number of area sample points on the radius, 
            t_max: number of area sample points on the angle
        '''
        
        error_vct = np.zeros(n_pts)
        rel_error_vct = np.zeros(n_pts)
        idx = 0

        pts = self.area.get_random_points(n_pts, qRange = [0.0, 0.15], tRange = [(5 * np.pi) / 6, (13 * np.pi) / 6])
        self.area.plot_boundary(show_plot=False)

        for pt in pts:
            if verbose_mode: 
                plt.plot(pt[0], pt[1], '.')

            ex = np.sqrt(self.a_ex(pt[0], pt[1])) * self.V_ex(pt[0], pt[1])
            approx = self.get_u_approx(pt, mu_approx)

            if not np.isnan(approx) and ex != 0:
                error_vct[idx] = abs(ex - approx)
                rel_error_vct[idx] = abs(error_vct[idx] / ex) * 100

                if verbose_mode:
                    if latex_mode:
                        print(f"{pt[0]:<05f}, {pt[1]:<05f}\t& {ex:<05f} & {approx:<05f} & {rel_error_vct[idx]:<05f}")
                    else:
                        print(f"In point {pt}: exact: {ex}, approx.: {approx}, error: {error_vct[idx]}, rel. error {rel_error_vct[idx]} %")

                idx += 1

        if verbose_mode:                
            print(f"error_vcr={error_vct}")

            plt.axis('scaled')
            plt.show()

        return error_vct, rel_error_vct

def load_or_generate_exact_data(g, f, area, M_ex=128):
    if path.exists('exact.npy'):
        data = np.array([])
        with open('exact.npy', 'rb') as f:
            data = np.load(f)
            data = data.tolist()

        return data[0][:-1], data[0][-1]
    else:        
        solver_ex = DirichletNeumannSolver(g, f, area)
        mu_ex, alpha_ex = solver_ex.solve(M=M_ex, verbose_mode=False)

        mu_ex_data = list(mu_ex)
        mu_ex_data.append(alpha_ex)

        with open('exact.npy', 'wb') as f:
            np.save(f, np.array([mu_ex_data]))

        return mu_ex, alpha_ex

#V = lambda x, y: x**2 - y**2
#gradV = lambda x, y: [2*x, -2*y]

r = lambda t: np.math.sqrt(np.cos(t)**2 + 0.25*(np.sin(t)**2))
area = SemiInfiniteArea(
    D   = lambda q, t: np.array([   r(t)*np.cos(t),   r(t)*np.sin(t) + 1.5     ]),
    dD  = lambda q, t: np.array([   -q*np.sin(t),  q*np.cos(t)     ]),
    d2D = lambda q, t: np.array([   -q*np.cos(t),  -q*np.sin(t)    ]),

    qRange = [0.0, 1.0],
    tRange = [0.0, 2*np.pi]
)

# g = lambda t: V(
#     area.Gamma(t)[0], 
#     area.Gamma(t)[1]
# )

# f = lambda t: np.dot(gradV(
#     area.x_inf(t)[0], 
#     area.x_inf(t)[1]
# ), area.normal_inf(t))

g_x = lambda x: x[0] - 0.1*(x[1] - 1.5)
g = lambda t: g_x(area.Gamma(t))

f_x = lambda x: x[0]*np.math.exp(-(x[0]**2))
f = lambda t: f_x(area.x_inf(t))

area.plot_boundary()


solver = DirichletNeumannSolver(g, f, area)

# generate 'exact' data
mu_ex, alpha_ex = load_or_generate_exact_data(g, f, area)
V = lambda x: solver.get_u_approx(x, mu_ex, alpha_ex, 1, 64)
dVnu = lambda t: solver.get_du_normal_approx(t, mu_ex, alpha_ex, 1, 64)


x_test = np.array([1.1, 1.5])
t_test = np.pi / 2

print(f'\nV({x_test}) = {V(x_test)}\n')
print(f'\ndVnu({x_test}) = {dVnu(t_test)}\n')

for i in [4, 8, 16, 32, 64]:
    mu, alpha = solver.solve(M=i, verbose_mode=False)
    print(f' >>> [M={i}] U({x_test}) = {solver.get_u_approx(x_test, mu, alpha, 1, 64)}')
    print(f' >>> [M={i}] dUnu({t_test}) = {solver.get_du_normal_approx(t_test, mu, alpha, 1, 64)}')