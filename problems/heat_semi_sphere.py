from src import *
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import sympy as sp
import numpy as np
import os
import dill

def generate_semi_sphere(N, R=1.0):
    theta, phi = sp.symbols('theta phi', real=True)

    # parametrization
    x_sym = R * sp.sin(theta) * sp.cos(phi)
    y_sym = R * sp.sin(theta) * sp.sin(phi)
    z_sym = R * sp.cos(theta)

    manifold = Manifold([theta, phi], [x_sym, y_sym, z_sym])
    manifold.compute()

    theta_range = (0, np.pi/2)
    phi_range = (0, 2*np.pi)

    theta_max = theta_range[1]

    num_boundary = int(np.round(np.sqrt(2 * np.pi * N)))
    num_interior = N - num_boundary

    manifold.sample([theta_range, phi_range], num_interior)

    # boundary
    x_sym_bdry = R * sp.cos(phi)
    y_sym_bdry = R * sp.sin(phi)
    z_sym_bdry = sp.S(0)

    boundary = Manifold([phi], [x_sym_bdry, y_sym_bdry, z_sym_bdry])
    boundary.sample([phi_range], num_boundary)

    manifold.params = np.vstack([
        manifold.params,
        np.insert(boundary.params, 0, values=theta_max, axis=1)
    ])

    manifold.points = np.vstack([
        manifold.points,
        boundary.points
    ])

    id_interior = np.arange(num_interior)
    id_boundary = np.arange(num_interior, N)

    n_vecs = np.zeros((N, manifold.n)) 
    n_vecs[id_boundary] = [0.0, 0.0, -1.0]

    manifold.id_interior = id_interior
    manifold.id_boundary = id_boundary

    manifold.build_tree()

    return manifold, id_interior, id_boundary, n_vecs


def compute_mms_semi_sphere(manifold, n_vecs):
    theta, phi = manifold.xi
    x_sym, y_sym, z_sym = manifold.x

    u_sym = sp.sin(x_sym) * sp.cos(y_sym)

    u_lap_sym = manifold.get_laplacian(u_sym)
    u_grad_sym = manifold.get_gradient(u_sym)
    f_sym = -u_lap_sym

    u_func = sp.lambdify((theta, phi), u_sym, 'numpy')
    f_func = sp.lambdify((theta, phi), f_sym, 'numpy')
    u_lap_func = sp.lambdify((theta, phi), u_lap_sym, 'numpy')
    u_grad_func = sp.lambdify((theta, phi), u_grad_sym, 'numpy')

    tt = manifold.params[:, 0]
    pp = manifold.params[:, 1]

    u_vals = u_func(tt, pp)
    f_vals = f_func(tt, pp)
    u_lap_vals = u_lap_func(tt, pp)

    u_grad_vals = u_grad_func(tt, pp).squeeze().T # shape: (N, n)
    g_vals = u_vals + np.sum(n_vecs * u_grad_vals, axis=1) # shape: (N,)

    return u_vals, f_vals, u_lap_vals, u_grad_vals, g_vals


def precompute_schur_solver_bdf4(L, D_n, id_interior, id_boundary, N, dt, nu):
    gamma = 12.0 / 25.0  # BDF4 step scale coeff

    L_csr = L.tocsr()
    D_n_csr = D_n.tocsr()
    num_boundary = len(id_boundary)

    L_II = L_csr[id_interior, :][:, id_interior]
    L_IB = L_csr[id_interior, :][:, id_boundary]

    # BDF4: I - \gamma * \nu * dt * L
    I_I = sparse.eye(len(id_interior), format='csr')
    A_II = I_I - nu * dt * gamma * L_II
    A_IB = -nu * dt * gamma * L_IB

    B_BI = D_n_csr[id_boundary, :][:, id_interior] 
    I_B = sparse.eye(num_boundary, format='csr')
    B_BB = D_n_csr[id_boundary, :][:, id_boundary] + I_B

    B_BB_diag = B_BB.diagonal()
    B_BB_inv = sparse.diags(1.0 / B_BB_diag, format='csr')

    A_prime = A_II - A_IB @ B_BB_inv @ B_BI
    A_prime_csc = A_prime.tocsc()
    solve_LU = splinalg.factorized(A_prime_csc)
    
    return solve_LU, A_IB, B_BB_inv, B_BI


def heat_semi_sphere(N=6400, l=4, K=30, l_grad=4, K_grad=35, lap_opt='qp', dn_opt='qp', seed=None, max_retires_opt=20, vis=False):
    #-- PARAMETERS --#
    kappa = 3
    delta = 1e-5

    if seed is not None:
        np.random.seed(seed)

    #-- GEOMETRY --#
    manifold, id_interior, id_boundary, n_vecs = generate_semi_sphere(N)

    num_interior = len(id_interior)
    num_boundary = len(id_boundary)

    #-- MANUFACTURED SOLUTION --#
    u_vals, f_vals, u_lap_vals, u_grad_vals, g_vals = compute_mms_semi_sphere(manifold, n_vecs)

    #-- OPERATORS --#
    L = sparse.lil_matrix((N, N))

    for i_id in id_interior:
        fetcher_in = lambda k: manifold.get_in_stencil(i_id, k)
        
        lap_kwargs = {
            'tangent_basis': manifold.get_local_basis(manifold.params[i_id])[0],
            'operator': 'lap',
            'l': l,
        }

        weights_lap, stencil_ids = get_stable_weights(
            stencil_fetcher=fetcher_in,
            weight_kwargs=lap_kwargs,
            K_init=K,
            expected_sign=-1,
            opt=lap_opt,
            gamma=3.0,
            max_retries_opt=max_retires_opt
        )

        L[i_id, stencil_ids] = weights_lap

    D_n = sparse.lil_matrix((N, N))
    
    for b_id in id_boundary:
        n_vec = n_vecs[b_id]
        
        enhanced_tree = manifold.build_enhanced_tree(b_id, enhance_direction=-n_vec)
        
        fetcher_bd = lambda k: manifold.get_bd_stencil(b_id, k, method='restricted', enhanced_tree=enhanced_tree)

        grad_kwargs = {
            'tangent_basis': manifold.get_local_basis(manifold.params[b_id])[0],
            'operator': 'dn',
            'n_vec': n_vec,
            'l': l_grad,
        }

        weights_grad_n, stencil_ids = get_stable_weights(
            stencil_fetcher=fetcher_bd,
            weight_kwargs=grad_kwargs,
            K_init=K_grad,
            expected_sign=1,
            opt=dn_opt,
            gamma=3.0,
            max_retries_opt=max_retires_opt
        )
        
        D_n[b_id, stencil_ids] = weights_grad_n

    # BDF4
    nu = 0.1
    dt = 1e-3
    T_end = 0.05
    n_steps = int(T_end / dt)

    solve_LU, A_IB, B_BB_inv, B_BI = precompute_schur_solver_bdf4(L, D_n, id_interior, id_boundary, N, dt, nu)

    u_num = np.zeros(N) 

    u_n3 = np.exp(3 * dt) * u_vals  # t = -3dt
    u_n2 = np.exp(2 * dt) * u_vals  # t = -2dt
    u_n1 = np.exp(1 * dt) * u_vals  # t = -1dt
    u_n0 = np.exp(0 * dt) * u_vals  # t = 0

    for step in range(1, n_steps + 1):
        t_n1 = step * dt
        
        # boundary and source term at t_{n+1} 
        g_vals_n1 = np.exp(-t_n1) * g_vals
        f_vals_n1 = -np.exp(-t_n1) * u_vals - nu * np.exp(-t_n1) * u_lap_vals
        
        # history
        u_I_n0 = u_n0[id_interior]
        u_I_n1 = u_n1[id_interior]
        u_I_n2 = u_n2[id_interior]
        u_I_n3 = u_n3[id_interior]
        
        f_I_n1 = f_vals_n1[id_interior]
        g_B_n1 = g_vals_n1[id_boundary]
        
        # BDF4
        b_I = (48.0/25.0)*u_I_n0 - (36.0/25.0)*u_I_n1 + (16.0/25.0)*u_I_n2 - (3.0/25.0)*u_I_n3 + (12.0/25.0)*dt*f_I_n1
        
        b_prime = b_I - A_IB @ (B_BB_inv @ g_B_n1)
        
        u_num_interior = solve_LU(b_prime)
        u_num_boundary = B_BB_inv @ (g_B_n1 - B_BI @ u_num_interior)
        
        u_num[id_interior] = u_num_interior
        u_num[id_boundary] = u_num_boundary

        # shift right
        u_n3 = u_n2
        u_n2 = u_n1
        u_n1 = u_n0
        u_n0 = u_num.copy()

        # if step % 10 == 0:
        #     u_exact_current = np.exp(-t_n1) * u_vals
        #     current_ie_l2 = np.sqrt(np.sum(np.abs(u_num - u_exact_current) ** 2) / N)
        #     print(f"Step {step:4d} | t = {t_n1:.4f} | L2 Error: {current_ie_l2:.3e}")

    u_exact_T = np.exp(-T_end) * u_vals

    ie = np.abs(u_num - u_exact_T) # shape: (N,)
    ie_l2 = np.sqrt(np.sum(ie ** 2) / N)

    if vis:
        return manifold.points, u_num, ie
    return ie_l2

if __name__ == "__main__":
    ie_l2 = heat_semi_sphere(N=6400, l=4, K=25, l_grad=4, K_grad=30, seed=0, vis=False)
    print(f"L2 IE: {ie_l2:.3e}")