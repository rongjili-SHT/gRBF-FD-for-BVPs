from src import *
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import copy

def generate_paraboloid_interface(N, r_a=0.7, eps=0.3, m_fold=5, alpha=11*np.pi/13):
    u, v = sp.symbols('u v', real=True)
    x_sym, y_sym = u, v
    z_sym = u**2 + v**2
    manifold = Manifold([u, v], [x_sym, y_sym, z_sym])
    manifold.compute()

    R_max = 1.4
    
    A = np.pi * R_max**2 * 1.5 
    L_if = 2 * np.pi * r_a        
    L_out = 8 * R_max 
    
    num_interface = int(np.sqrt(L_if**2 * N / A))
    num_outer = int(np.sqrt(L_out**2 * N / A))
    num_outer = num_outer - (num_outer % 4)
    num_surface = N - num_interface - num_outer

    manifold.sample([(-R_max, R_max), (-R_max, R_max)], num_points=num_surface) # manifold.params, manifold.points
    u_surf, v_surf, z_surf = manifold.points.T
    r_surf = np.sqrt(z_surf)
    phi_surf = np.arctan2(v_surf, u_surf)

    # sample interface
    t_if = np.random.uniform(0, 2*np.pi, size=num_interface)
    r_if = r_a + eps * np.cos(m_fold * (t_if - alpha))
    u_if = r_if * np.cos(t_if)
    v_if = r_if * np.sin(t_if)

    params_if = np.column_stack([u_if, v_if])
    points_if = np.column_stack([u_if, v_if, u_if**2 + v_if**2])

    # sample outer boundary
    pts_per_side = num_outer // 4
    s = np.random.uniform(-R_max, R_max, size=pts_per_side)
    
    u_out_list, v_out_list = [], []
    # Bottom (v = -R_max)
    u_out_list.append(s); v_out_list.append(np.full_like(s, -R_max))
    # Right (u = R_max)
    u_out_list.append(np.full_like(s, R_max)); v_out_list.append(s)
    # Top (v = R_max)
    u_out_list.append(-s); v_out_list.append(np.full_like(s, R_max))
    # Left (u = -R_max)
    u_out_list.append(np.full_like(s, -R_max)); v_out_list.append(-s)

    u_out = np.concatenate(u_out_list)
    v_out = np.concatenate(v_out_list)

    params_out = np.column_stack([u_out, v_out])
    points_out = np.column_stack([u_out, v_out, u_out**2 + v_out**2])

    # split surface points
    r_interface_at_phi = r_a + eps * np.cos(m_fold * (phi_surf - alpha))
    id_p = np.where(r_surf >= r_interface_at_phi)[0]
    id_m = np.where(r_surf < r_interface_at_phi)[0]

    params_p, params_m = manifold.params[id_p], manifold.params[id_m]
    points_p, points_m = manifold.points[id_p], manifold.points[id_m]

    manifold_p = manifold
    manifold_m = copy.deepcopy(manifold)

    # assemble
    manifold_m.params = np.vstack([params_m, params_if])
    manifold_m.points = np.vstack([points_m, points_if])
    manifold_m.id_interior = np.arange(len(params_m))
    manifold_m.id_boundary = np.arange(len(params_m), len(params_m) + num_interface)
    manifold_m.build_tree()

    manifold_p.params = np.vstack([params_p, params_if, params_out])
    manifold_p.points = np.vstack([points_p, points_if, points_out])
    manifold_p.id_interior = np.arange(len(params_p))
    manifold_p.id_boundary = np.arange(len(params_p), len(params_p) + num_interface)
    manifold_p.id_outer = np.arange(len(params_p) + num_interface, len(params_p) + num_interface + num_outer)
    manifold_p.build_tree()

    # compute n_vecs_if using conormal formula
    r_prime = -eps * m_fold * np.sin(m_fold * (t_if - alpha))
    u_prime = r_prime * np.cos(t_if) - r_if * np.sin(t_if)
    v_prime = r_prime * np.sin(t_if) + r_if * np.cos(t_if)
    
    # z = u^2 + v^2, 所以 z' = 2u*u' + 2v*v'
    c_prime = np.column_stack([u_prime, v_prime, 2 * u_if * u_prime + 2 * v_if * v_prime])
    
    # F(u,v,z) = u^2 + v^2 - z = 0 -> gradient: (2u, 2v, -1)
    N_surf = np.column_stack([-2 * u_if, -2 * v_if, np.ones_like(u_if)])
    N_surf /= np.linalg.norm(N_surf, axis=1)[:, None]
    
    conormal = np.cross(c_prime, N_surf)
    conormal /= np.linalg.norm(conormal, axis=1)[:, None]
    
    n_vecs_if = conormal

    return manifold_p, manifold_m, n_vecs_if


def get_mms_data(manifold, u_sym, beta, kappa=1.0):
    u_lap_sym = manifold.get_laplacian(u_sym)
    u_grad_sym = manifold.get_gradient(u_sym)
    f_sym = -beta * u_lap_sym + kappa * u_sym

    u, v = manifold.xi
    u_func = sp.lambdify((u, v), u_sym, 'numpy')
    f_func = sp.lambdify((u, v), f_sym, 'numpy')
    u_grad_func = sp.lambdify((u, v), u_grad_sym, 'numpy')

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


def compute_mms_interface(manifold_p, manifold_m, n_vecs_if, beta_p, beta_m, kappa_p, kappa_m):
    u, v = manifold_p.xi
    
    x_sym = u
    y_sym = v
    z_sym = u**2 + v**2

    u_p_sym = (x_sym**2 - 1) * (y_sym**2 - 1)
    u_m_sym = sp.cos(x_sym + y_sym) * sp.sin(z_sym)

    u_p, f_p, u_grad_p = get_mms_data(manifold_p, u_p_sym, beta_p, kappa_p)
    u_m, f_m, u_grad_m = get_mms_data(manifold_m, u_m_sym, beta_m, kappa_m)

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
        n_vec = n_vecs_if[i]
        
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
                           f_p, f_m, q_vals, psi_vals, beta_p, beta_m, kappa_p, kappa_m,
                           u_p_exact):
    
    N_m = len(manifold_m.points)
    N_p = len(manifold_p.points)
    N_tot = N_m + N_p

    A = sparse.lil_matrix((N_tot, N_tot))
    F = np.zeros(N_tot)

    id_int_m = manifold_m.id_interior
    id_bd_m = manifold_m.id_boundary
    
    id_int_p = manifold_p.id_interior
    id_bd_p = manifold_p.id_boundary
    id_out_p = manifold_p.id_outer

    I_m = sparse.diags(np.full(N_m, kappa_m), format='lil')
    A[id_int_m, 0:N_m] = (-beta_m * L_m + I_m)[id_int_m, :]
    F[id_int_m] = f_m[id_int_m]

    # dirichlet jump
    A[id_bd_m, id_bd_m] = -1.0
    A[id_bd_m, N_m + id_bd_p] = 1.0
    F[id_bd_m] = q_vals

    I_p = sparse.diags(np.full(N_p, kappa_p), format='lil')
    A[N_m + id_int_p, N_m:N_tot] = (-beta_p * L_p + I_p)[id_int_p, :]
    F[N_m + id_int_p] = f_p[id_int_p]

    # flux jump
    A[N_m + id_bd_p, 0:N_m] = -beta_m * D_n_m[id_bd_m, :]
    A[N_m + id_bd_p, N_m:N_tot] = beta_p * D_n_p[id_bd_p, :]
    F[N_m + id_bd_p] = psi_vals

    # dirichlet
    A[N_m + id_out_p, N_m + id_out_p] = 1.0
    F[N_m + id_out_p] = u_p_exact[id_out_p]

    A_csr = A.tocsr()
    U_global = splinalg.spsolve(A_csr, F)

    U_m_num = U_global[0:N_m]
    U_p_num = U_global[N_m:N_tot]

    return U_p_num, U_m_num


def interface_paraboloid(N=6400, l=4, K=25, l_grad=4, K_grad=35, seed=None, vis=False):
    if seed is not None:
        np.random.seed(seed)

    beta_m = 1.0
    kappa_m = 3.0
    
    beta_p = 2.0
    kappa_p = 1.0

    manifold_p, manifold_m, n_vecs_if = generate_paraboloid_interface(N)

    u_p, u_m, f_p, f_m, q_vals, psi_vals = compute_mms_interface(
        manifold_p, manifold_m, n_vecs_if, beta_p, beta_m, kappa_p, kappa_m
    )

    L_p, D_n_p = build_operators(manifold_p, n_vecs_if, K, l, K_grad, l_grad, is_plus=True)
    L_m, D_n_m = build_operators(manifold_m, n_vecs_if, K, l, K_grad, l_grad, is_plus=False)

    U_p_num, U_m_num = solve_interface_system(
        manifold_p, manifold_m, L_p, D_n_p, L_m, D_n_m,
        f_p, f_m, q_vals, psi_vals, beta_p, beta_m, kappa_p, kappa_m,
        u_p
    )

    u_exact_global = np.concatenate([u_m, u_p])
    U_num_global = np.concatenate([U_m_num, U_p_num])
    
    N_T = len(u_exact_global)

    ie_global = np.abs(U_num_global - u_exact_global)
    ie_global_l2 = np.sqrt(np.sum(ie_global**2) / N_T)

    if vis:
        pts_global = np.vstack([manifold_m.points, manifold_p.points])
        # return pts_global, U_num_global, ie_global,
        return pts_global, U_num_global, ie_global,
    return ie_global_l2

if __name__ == "__main__":
    ie = interface_paraboloid(N=6400, seed=0, K_grad=35)
    print(f"ie: {ie:.3e}")