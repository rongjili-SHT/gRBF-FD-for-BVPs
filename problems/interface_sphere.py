from src import *
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

def generate_sphere_interface(N, R=0.5):
    theta, phi = sp.symbols('theta phi', real=True)

    x_sym = R * sp.sin(phi) * sp.cos(theta)
    y_sym = R * sp.sin(phi) * sp.sin(theta)
    z_sym = R * sp.cos(phi)

    manifold_p = Manifold([theta, phi], [x_sym, y_sym, z_sym])
    manifold_p.compute()
    
    manifold_m = Manifold([theta, phi], [x_sym, y_sym, z_sym])
    manifold_m.compute()

    A = 4 * np.pi * R**2 # area of the surface
    L = 2 * np.pi * R # length of interface

    num_interface = int(np.sqrt(L**2 * N / A))
    num_interior = (N - num_interface) // 2

    # sample interface
    theta_if = np.random.uniform(0, 2*np.pi, size=num_interface)
    phi_if = np.full(num_interface, np.pi/2)
    params_bd = np.column_stack([theta_if, phi_if])
    
    x_bd = R * np.sin(phi_if) * np.cos(theta_if)
    y_bd = R * np.sin(phi_if) * np.sin(theta_if)
    z_bd = R * np.cos(phi_if)
    points_bd = np.column_stack([x_bd, y_bd, z_bd])

    thres = 0.0
    # M+: phi \in (0, pi/2)
    manifold_p.sample([(0, 2*np.pi), (0 + thres, np.pi/2 - thres)], num_interior)
    
    # M-: phi \in (pi/2, pi)
    manifold_m.sample([(0, 2*np.pi), (np.pi/2 + thres, np.pi - thres)], num_interior)

    # share boundary points
    for mf, params_int, points_int in zip(
        [manifold_p, manifold_m], 
        [manifold_p.params, manifold_m.params], 
        [manifold_p.points, manifold_m.points]
    ):
        mf.params = np.vstack([params_int, params_bd])
        mf.points = np.vstack([points_int, points_bd])
        mf.id_interior = np.arange(num_interior)
        mf.id_boundary = np.arange(num_interior, num_interior + num_interface)
        mf.build_tree()

    n_vecs_if = np.zeros((num_interface, 3))
    n_vecs_if[:, 2] = 1.0 

    return manifold_p, manifold_m, n_vecs_if

def get_mms_data(manifold, u_sym, beta, kappa=1.0):
    u_lap_sym = manifold.get_laplacian(u_sym)
    u_grad_sym = manifold.get_gradient(u_sym)
    f_sym = -beta * u_lap_sym + kappa * u_sym

    theta, phi = manifold.xi
    u_func = sp.lambdify((theta, phi), u_sym, 'numpy')
    f_func = sp.lambdify((theta, phi), f_sym, 'numpy')
    u_grad_func = sp.lambdify((theta, phi), u_grad_sym, 'numpy')

    tt = manifold.params[:, 0]
    pp = manifold.params[:, 1]

    u_vals = u_func(tt, pp)
    f_vals = f_func(tt, pp)
    
    u_grad_vals = u_grad_func(tt, pp)
    if u_grad_vals.ndim == 3:
        u_grad_vals = u_grad_vals.squeeze()
    if u_grad_vals.shape[0] == 3:
        u_grad_vals = u_grad_vals.T

    return u_vals, f_vals, u_grad_vals

def compute_mms_interface(manifold_p, manifold_m, n_vecs_if, beta_p, beta_m, kappa=1.0):
    R_val = 0.5
    theta, phi = manifold_p.xi
    
    x_sym = R_val * sp.sin(phi) * sp.cos(theta)
    y_sym = R_val * sp.sin(phi) * sp.sin(theta)
    z_sym = R_val * sp.cos(phi)

    u_p_sym = x_sym * y_sym * sp.sin(4 * sp.pi * z_sym)
    u_m_sym = x_sym * y_sym * sp.cos(4 * sp.pi * z_sym)

    u_p, f_p, u_grad_p = get_mms_data(manifold_p, u_p_sym, beta_p, kappa)
    u_m, f_m, u_grad_m = get_mms_data(manifold_m, u_m_sym, beta_m, kappa)

    id_bd_p = manifold_p.id_boundary
    id_bd_m = manifold_m.id_boundary

    q_vals = u_p[id_bd_p] - u_m[id_bd_m]
    
    flux_p = np.sum(n_vecs_if * u_grad_p[id_bd_p], axis=1)
    flux_m = np.sum(n_vecs_if * u_grad_m[id_bd_m], axis=1)
    psi_vals = beta_p * flux_p - beta_m * flux_m

    return u_p, u_m, f_p, f_m, q_vals, psi_vals

def build_operators(manifold, n_vecs_if, K, l, K_grad, l_grad, is_plus=True):
    N = len(manifold.points)
    L = sparse.lil_matrix((N, N))
    D_n = sparse.lil_matrix((N, N))

    for i_id in manifold.id_interior:
        fetcher = lambda k: manifold.get_in_stencil(i_id, k)
        lap_kwargs = {
            'tangent_basis': manifold.get_local_basis(manifold.params[i_id])[0],
            'operator': 'lap', 'l': l
        }
        weights, stencil = get_stable_weights(fetcher, lap_kwargs, K, -1, 'qp', 3.0)
        L[i_id, stencil] = weights

    for i, b_id in enumerate(manifold.id_boundary):
        n_vec = n_vecs_if[i] # [0, 0, 1]
        
        enhance_dir = n_vec if is_plus else -n_vec
        enhanced_tree = manifold.build_enhanced_tree(b_id, enhance_direction=enhance_dir)
        
        fetcher = lambda k: manifold.get_bd_stencil(b_id, k, method='restricted', enhanced_tree=enhanced_tree)
        grad_kwargs = {
            'tangent_basis': manifold.get_local_basis(manifold.params[b_id])[0],
            'operator': 'dn', 'n_vec': n_vec, 'l': l_grad
        }
        weights, stencil = get_stable_weights(fetcher, grad_kwargs, K_grad, 1, 'qp', 3.0)
        D_n[b_id, stencil] = weights

    return L, D_n

def solve_interface_system(manifold_p, manifold_m, L_p, D_n_p, L_m, D_n_m, 
                           f_p, f_m, q_vals, psi_vals, beta_p, beta_m, kappa=1.0):
    
    Ni_m = len(manifold_m.id_interior)
    Nb = len(manifold_m.id_boundary)
    Ni_p = len(manifold_p.id_interior)

    N_m = Ni_m + Nb
    N_p = Ni_p + Nb
    N_tot = N_m + N_p

    A = sparse.lil_matrix((N_tot, N_tot))
    F = np.zeros(N_tot)

    id_int_m = manifold_m.id_interior
    id_bd_m = manifold_m.id_boundary
    id_int_p = manifold_p.id_interior
    id_bd_p = manifold_p.id_boundary

    # === Row Block 1: Internal M- Equation ===
    row_idx = id_int_m
    I_m = sparse.diags(np.full(N_m, kappa), format='lil')
    A[row_idx, 0:N_m] = (-beta_m * L_m + I_m)[id_int_m, :]
    F[row_idx] = f_m[id_int_m]

    # === Row Block 2: Dirichlet Jump [-u^- + u^+ = q] ===
    row_idx = Ni_m + np.arange(Nb)
    A[row_idx, id_bd_m] = -1.0
    A[row_idx, N_m + id_bd_p] = 1.0
    F[row_idx] = q_vals

    # === Row Block 3: Neumann Flux Jump [-beta^- D_n^- u^- + beta^+ D_n^+ u^+ = psi] ===
    row_idx = N_m + np.arange(Nb)
    A[row_idx, 0:N_m] = -beta_m * D_n_m[id_bd_m, :]
    A[row_idx, N_m:N_tot] = beta_p * D_n_p[id_bd_p, :]
    F[row_idx] = psi_vals

    # === Row Block 4: Internal M+ Equation ===
    row_idx = N_m + Nb + np.arange(Ni_p)
    I_p = sparse.diags(np.full(N_p, kappa), format='lil')
    A[row_idx, N_m:N_tot] = (-beta_p * L_p + I_p)[id_int_p, :]
    F[row_idx] = f_p[id_int_p]

    A_csr = A.tocsr()
    U_global = splinalg.spsolve(A_csr, F)

    U_m_num = U_global[0:N_m]
    U_p_num = U_global[N_m:N_tot]

    return U_p_num, U_m_num

def interface_sphere(N=6400, l=4, K=25, l_grad=4, K_grad=35, seed=None, vis=False):
    if seed is not None:
        np.random.seed(seed)

    beta_p = 1.0
    beta_m = 10.0
    kappa = 1.0

    manifold_p, manifold_m, n_vecs_if = generate_sphere_interface(N)

    u_p, u_m, f_p, f_m, q_vals, psi_vals = compute_mms_interface(
        manifold_p, manifold_m, n_vecs_if, beta_p, beta_m, kappa
    )

    L_p, D_n_p = build_operators(manifold_p, n_vecs_if, K, l, K_grad, l_grad, is_plus=True)
    L_m, D_n_m = build_operators(manifold_m, n_vecs_if, K, l, K_grad, l_grad, is_plus=False)

    U_p_num, U_m_num = solve_interface_system(
        manifold_p, manifold_m, L_p, D_n_p, L_m, D_n_m,
        f_p, f_m, q_vals, psi_vals, beta_p, beta_m, kappa
    )

    u_exact_global = np.concatenate([u_m, u_p])
    U_num_global = np.concatenate([U_m_num, U_p_num])
    
    N_T = len(u_exact_global)

    ie_global = np.abs(U_num_global - u_exact_global)

    ie_global_l2 = np.sqrt(np.sum(ie_global**2) / N_T)

    if vis:
        pts_global = np.vstack([manifold_m.points, manifold_p.points])
        return pts_global, U_num_global, ie_global,
    return ie_global_l2

if __name__ == "__main__":
    ie = interface_sphere(N=6400, seed=0)
    print(f"ie: {ie:.3e}")