from src import * 
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import numpy as np

jax.config.update("jax_enable_x64", True)

def generate_helical_pipe(N):
    a = 8.0
    b = 1.0
    denom = jnp.sqrt(a**2 + b**2)
    alpha = a / denom
    beta = b / denom

    num_turns = 1.5
    omega_max = num_turns * 2 * jnp.pi

    R_start_scale = 3.5
    R_end_scale = 3.5

    def x_map(param):
        theta, omega = param[0], param[1]
        scale = R_start_scale - (R_start_scale - R_end_scale) * (omega / omega_max)
        R_base = 0.6 + 0.075 * jnp.sin(5 * theta)
        R = R_base * scale

        x = a * jnp.cos(omega) + R * (beta * jnp.sin(theta) * jnp.sin(omega) - jnp.cos(theta) * jnp.cos(omega))
        y = a * jnp.sin(omega) - R * (beta * jnp.sin(theta) * jnp.cos(omega) + jnp.cos(theta) * jnp.sin(omega))
        z = b * omega + R * alpha * jnp.sin(theta)
        
        return jnp.array([x, y, z])

    manifold = ManifoldJax(d=2, n=3, x_map=x_map)

    num_boundary = int(np.sqrt(N) * 3)
    total_scale = R_start_scale + R_end_scale
    num_bdry1 = int(num_boundary * (R_start_scale / total_scale))
    num_bdry2 = num_boundary - num_bdry1
    num_interior = N - num_boundary

    # generate params
    theta_int = np.random.uniform(0, 2 * np.pi, num_interior)
    omega_int = np.random.uniform(0, omega_max, num_interior)
    params_int = np.column_stack([theta_int, omega_int])

    theta_bdry1 = np.random.uniform(0, 2 * np.pi, num_bdry1)
    params_bdry1 = np.column_stack([theta_bdry1, np.zeros(num_bdry1)])

    theta_bdry2 = np.random.uniform(0, 2 * np.pi, num_bdry2)
    params_bdry2 = np.column_stack([theta_bdry2, np.full(num_bdry2, omega_max)])

    # get points
    params = np.vstack([params_int, params_bdry1, params_bdry2])
    x_map_vmap = jax.jit(jax.vmap(x_map))
    points = np.array(x_map_vmap(params))

    manifold.params = params
    manifold.points = points

    id_interior = np.arange(num_interior)
    id_bdry1 = np.arange(num_interior, num_interior + num_bdry1)
    id_bdry2 = np.arange(num_interior + num_bdry1, N)
    id_boundary = np.concatenate([id_bdry1, id_bdry2])

    n_vecs = np.zeros((N, 3))
    
    d_omega_func = jax.jit(jax.vmap(lambda p: jax.jacfwd(x_map)(p)[:, 1]))

    n1 = -np.array(d_omega_func(params_bdry1))
    n1 /= np.linalg.norm(n1, axis=1, keepdims=True)
    n_vecs[id_bdry1] = n1

    n2 = np.array(d_omega_func(params_bdry2))
    n2 /= np.linalg.norm(n2, axis=1, keepdims=True)
    n_vecs[id_bdry2] = n2

    manifold.id_interior = id_interior
    manifold.id_boundary = id_boundary
    manifold.build_tree()

    return manifold, id_interior, id_boundary, n_vecs

def compute_mms_helical_pipe(manifold, n_vecs):
    # JAX
    def u_map(param):
        theta, omega = param[0], param[1]
        return jnp.sin(theta) * jnp.cos(omega)

    J_map = manifold.J_map
    grad_u_map = jax.grad(u_map)

    def G_map(param):
        J = J_map(param)
        return jnp.dot(J.T, J)

    def G_inv_map(param):
        return jnp.linalg.inv(G_map(param))

    def g_det_map(param):
        return jnp.linalg.det(G_map(param))

    def manifold_grad_map(param):
        return jnp.dot(J_map(param), jnp.dot(G_inv_map(param), grad_u_map(param)))

    def vector_field_map(param):
        return jnp.sqrt(g_det_map(param)) * jnp.dot(G_inv_map(param), grad_u_map(param))

    div_V_map = jax.jacfwd(vector_field_map)

    def laplacian_map(param):
        div_V = jnp.trace(div_V_map(param))
        return (1.0 / jnp.sqrt(g_det_map(param))) * div_V

    u_vmap = jax.jit(jax.vmap(u_map))
    lap_vmap = jax.jit(jax.vmap(laplacian_map))
    grad_vmap = jax.jit(jax.vmap(manifold_grad_map))

    params_jnp = jnp.array(manifold.params)

    u_vals = np.array(u_vmap(params_jnp))
    
    u_lap_vals = np.array(lap_vmap(params_jnp))
    f_vals = -u_lap_vals
    
    u_grad_vals = np.array(grad_vmap(params_jnp)) # Shape: (N, 3)

    g_vals = u_vals + np.sum(n_vecs * u_grad_vals, axis=1) # Robin condition

    return u_vals, f_vals, u_lap_vals, u_grad_vals, g_vals

def precompute_schur_solver_bdf4(L, D_n, id_interior, id_boundary, N, dt, nu):
    gamma = 12.0 / 25.0 

    L_csr = L.tocsr()
    D_n_csr = D_n.tocsr()
    num_boundary = len(id_boundary)

    L_II = L_csr[id_interior, :][:, id_interior]
    L_IB = L_csr[id_interior, :][:, id_boundary]

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

def heat_helical_pipe(N=6400, l=4, K=35, l_grad=4, K_grad=35, lap_opt='qp', dn_opt='qp', seed=None, vis=False):
    delta = 1e-5

    # print(f"Helical Pipe: N = {N} l_in = {l} l_bd = {l_grad}")

    if seed is not None:
        np.random.seed(seed)

    manifold, id_interior, id_boundary, n_vecs = generate_helical_pipe(N)

    u_vals, f_vals, u_lap_vals, u_grad_vals, g_vals = compute_mms_helical_pipe(manifold, n_vecs)

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
            gamma=3.0
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
            gamma=3.0
        )
        D_n[b_id, stencil_ids] = weights_grad_n

    nu = 0.1
    dt = 1e-3
    T_end = 0.05
    n_steps = int(T_end / dt)

    solve_LU, A_IB, B_BB_inv, B_BI = precompute_schur_solver_bdf4(L, D_n, id_interior, id_boundary, N, dt, nu)

    u_num = np.zeros(N) 

    u_n3 = np.exp(3 * dt) * u_vals
    u_n2 = np.exp(2 * dt) * u_vals
    u_n1 = np.exp(1 * dt) * u_vals
    u_n0 = np.exp(0 * dt) * u_vals

    for step in range(1, n_steps + 1):
        t_n1 = step * dt
        
        g_vals_n1 = np.exp(-t_n1) * g_vals
        f_vals_n1 = -np.exp(-t_n1) * u_vals - nu * np.exp(-t_n1) * u_lap_vals
        
        u_I_n0 = u_n0[id_interior]
        u_I_n1 = u_n1[id_interior]
        u_I_n2 = u_n2[id_interior]
        u_I_n3 = u_n3[id_interior]
        
        f_I_n1 = f_vals_n1[id_interior]
        g_B_n1 = g_vals_n1[id_boundary]
        
        b_I = (48.0/25.0)*u_I_n0 - (36.0/25.0)*u_I_n1 + (16.0/25.0)*u_I_n2 - (3.0/25.0)*u_I_n3 + (12.0/25.0)*dt*f_I_n1
        
        b_prime = b_I - A_IB @ (B_BB_inv @ g_B_n1)
        
        u_num_interior = solve_LU(b_prime)
        u_num_boundary = B_BB_inv @ (g_B_n1 - B_BI @ u_num_interior)
        
        u_num[id_interior] = u_num_interior
        u_num[id_boundary] = u_num_boundary

        u_n3 = u_n2
        u_n2 = u_n1
        u_n1 = u_n0
        u_n0 = u_num.copy()

        if step % 10 == 0:
            u_exact_current = np.exp(-t_n1) * u_vals
            current_ie_l2 = np.sqrt(np.sum(np.abs(u_num - u_exact_current) ** 2) / N)
            # print(f"Step {step:4d} | t = {t_n1:.4f} | L2 Error: {current_ie_l2:.3e}")

    u_exact_T = np.exp(-T_end) * u_vals

    ie = np.abs(u_num - u_exact_T)
    ie_l2 = np.sqrt(np.sum(ie ** 2) / N)
    ie_inf = np.max(ie)
    
    # print(f"Final Step | t = {T_end:.4f} | L2 Error: {ie_l2:.3e}")

    if vis:
        return manifold.points, u_num, ie
    else:
        return ie_l2

if __name__ == "__main__":
    ie_l2 = heat_helical_pipe(N=51200, l=4, K=25, l_grad=4, K_grad=30, seed=0)