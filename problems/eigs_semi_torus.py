from src import *
import scipy.sparse as sparse
from scipy.sparse.linalg import eigs

def generate_semi_torus(N, R=2.0, r=1.0):
    theta, phi = sp.symbols('theta phi', real=True)

    x_sym = (R + r * sp.cos(theta)) * sp.cos(phi)
    y_sym = (R + r * sp.cos(theta)) * sp.sin(phi)
    z_sym = r * sp.sin(theta)

    manifold = Manifold([theta, phi], [x_sym, y_sym, z_sym])
    manifold.compute()

    theta_range = (0, 2*np.pi)
    phi_range = (0, np.pi)

    phi_min = phi_range[0]
    phi_max = phi_range[1]

    num_boundary = 2 * int(np.round(np.sqrt(2*r/R)*np.sqrt(N)))
    num_interior = N - num_boundary

    manifold.sample([theta_range, phi_range], num_interior)

    # sample boundary

    x_sym_left = (R + r * sp.cos(theta)) * sp.cos(phi_min)
    y_sym_left = (R + r * sp.cos(theta)) * sp.sin(phi_min)

    boundary_left = Manifold([theta], [x_sym_left, y_sym_left, z_sym])
    boundary_left.sample([theta_range], num_boundary // 2)

    x_sym_right = (R + r * sp.cos(theta)) * sp.cos(phi_max)
    y_sym_right = (R + r * sp.cos(theta)) * sp.sin(phi_max)

    boundary_right = Manifold([theta], [x_sym_right, y_sym_right, z_sym])
    boundary_right.sample([theta_range], num_boundary // 2)

    manifold.params = np.vstack([
        manifold.params,
        np.insert(boundary_left.params, 1, values=phi_min, axis=1),
        np.insert(boundary_right.params, 1, values=phi_max, axis=1)
    ])

    manifold.points = np.vstack([
        manifold.points,
        boundary_left.points,
        boundary_right.points
    ])

    id_interior = np.arange(num_interior)
    id_boundary = np.arange(num_interior, N)

    # outward normal at each boundary point
    n_vecs = np.zeros((N, manifold.n)) 
    n_vecs[id_boundary] = [0.0, -1.0, 0.0]

    manifold.id_interior = id_interior
    manifold.id_boundary = id_boundary

    manifold.build_tree()

    return manifold, id_interior, id_boundary, n_vecs

def get_eigs(num_eigs, L, D_n, id_interior, id_boundary):
    """
    system:
    [ A_II   A_IB ] [ u_I ] = [ f_I ]
    [ B_BI   B_BB ] [ u_B ]   [ g_B ]
    """
    L_csr = L.tocsr()
    D_n_csr = D_n.tocsr()

    num_boundary = len(id_boundary)

    A = -L_csr
    A_II = A[id_interior, :][:, id_interior]
    A_IB = A[id_interior, :][:, id_boundary]

    B_BI = D_n_csr[id_boundary, :][:, id_interior] 
    
    I_B = sparse.eye(num_boundary, format='csr')
    B_BB = D_n_csr[id_boundary, :][:, id_boundary] + I_B

    B_BB_diag = B_BB.diagonal()
    B_BB_inv = sparse.diags(1.0 / B_BB_diag, format='csr')

    A_prime = A_II - A_IB @ B_BB_inv @ B_BI

    evals, evecs = eigs(A_prime.tocsc(), k=num_eigs, sigma=0, which='LM')

    idx = np.argsort(np.abs(evals))
    evals = np.real(evals[idx])
    evecs = evecs[:, idx]

    return evals, evecs

def eigs_semi_torus(N=6400, l=4, K=25, l_grad=4, K_grad=30, num_eigs=20, lap_opt='qp', dn_opt='qp', seed=None, require_manifold=False):
    #-- PARAMETERS --#
    kappa = 3
    delta = 1e-5

    if seed is not None:
        np.random.seed(seed)

    #-- GEOMETRY --#

    manifold, id_interior, id_boundary, n_vecs = generate_semi_torus(N)

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
    
    evals, evecs = get_eigs(num_eigs, L, D_n, id_interior, id_boundary)

    if require_manifold:
        return evals, evecs, manifold
    return evals, evecs

if __name__ == "__main__":
    evals, evecs = eigs_semi_torus(num_eigs=20)
    print(evals)