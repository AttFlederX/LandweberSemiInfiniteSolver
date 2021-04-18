from semi_infinite_area import SemiInfiniteArea
import numpy as np
import matplotlib.pyplot as plt
from util import print_matrix

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

    def get_mtx_coef(self, i, j, t, M):
        # TODO
        #return (self.R_j(t[i], t[j], n) * self.K_1(t[i], t[j])) + ((1 / (2*n)) * self.K_2(t[i], t[j]))
        pass


    ### Runtime functions
    def solve(self, M = 4, verbose_mode = False):
        '''
            Solves the KGE with the instance's parameter values
 
            n: number of quadrature nodes
        '''
        mu = np.zeros(2*M)
        t = np.linspace(self.area.t_a, self.area.t_b, 2*M, False)

        # setup linear system
        mtx_A = np.zeros((2*M, 2*M))
        vct_h = np.zeros(2*M)

        for i in range(2*M):
            for j in range(2*M):
                mtx_A[i, j] = self.get_mtx_coef(i, j, t, M)

                if i == j:
                    mtx_A[i, j] -= 0.5

            vct_h[i] = self.h(t[i])

        if verbose_mode:
            print("\nmtx_A=\n")
            print_matrix(mtx_A)
            print(f"vct_h={vct_h}")

        mu = np.linalg.solve(mtx_A, vct_h)

        if verbose_mode:
            print(f"\nmu_approx={mu}\n")

        return mu


    ## Testing functions

    def get_u_approx(self, x, mu):
        '''
            Calculates the approximate value of the function using given densities
        '''
        n = int(len(mu) / 2)

        quad_sum = 0
        for j in range(2*n):
            quad_sum += mu[j] * self.__K(x, self.__t(j, n))

        return quad_sum / (2*n)

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