from itertools import combinations_with_replacement
import numpy as np
from cvxopt import matrix, solvers
from scipy.optimize import linprog

solvers.options['show_progress'] = False

#-- Helper Functions -- #

def get_polynomial_basis(degree, dim):
    """
    Generates multi-indices for a complete polynomial basis in 'dim' dimensions.
    Example: 
        - degree=2, dim=2 -> [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]
        - degree=2, dim=3 -> [[0,0,0], [1,0,0], ..., [0,1,1], ..., [0,0,2]]
    """
    indices = []
    for d in range(degree + 1):
        for p in combinations_with_replacement(range(dim), d):
            index = np.zeros(dim, dtype=int)
            for i in p:
                index[i] += 1
            indices.append(index)
    return indices

def get_operator_weights(
        stencil, tangent_basis, l,
        operator='lap',
        opt=None, w1_min=None, n_vec=None,
    ):
    # hyperparameters
    kappa = 3
    delta = 1e-5

    K, n = stencil.shape
    d = tangent_basis.shape[0]

    W = np.diag([1.0] + [(1.0 / K)] * (K - 1)) # 1/K weight matrix

    stencil_center = stencil[0]
    local_coords = (stencil - stencil_center) @ tangent_basis.T # shape: (K, d)

    diameter = np.sqrt(np.max(np.sum((local_coords[:, None, :] - local_coords[None, :, :]) ** 2, axis=-1)))
    norm_coords = local_coords / diameter

    poly_basis = get_polynomial_basis(l, d)
    m = len(poly_basis)

    P = np.zeros((K, m))
    for j, alpha in enumerate(poly_basis):
        P[:, j] = np.prod(norm_coords**alpha, axis=1)

    dist = np.sqrt(np.sum((norm_coords[:, None, :] - norm_coords[None, :, :]) ** 2, axis=-1))
    Phi = dist ** (2 * kappa + 1)

    if operator == 'lap':
        scale_factor = diameter ** 2

        dP = np.zeros((1, m))
        for j, alpha in enumerate(poly_basis):
            if sum(alpha) == 2 and max(alpha) == 2:
                dP[0, j] = 2.0
        
        b_eq = dP[0, :]

    elif operator == 'dn':
        if n_vec is None:
            raise ValueError('n_vec required for operator dn')

        scale_factor = diameter

        dP = np.zeros((d, m))
        for j, alpha in enumerate(poly_basis):
            if sum(alpha) == 1:
                dim_idx = np.where(np.array(alpha) == 1)[0][0]
                dP[dim_idx, j] = 1.0
                
        dP = (n_vec @ tangent_basis.T @ dP).reshape(1, -1) # shape: (1, m)
        b_eq = dP[0, :]

    if opt == 'qp':
        H_diag = np.ones(K + 1)
        H_diag[0] = 1.0 / (K**2)
        H_diag[-1] = K**2 

        H = np.diag(H_diag)
        f = np.zeros(K + 1)

        # equality
        A_eq = np.zeros((m, K + 1))
        A_eq[:, :K] = P.T
        
        # # inequaliy 
        if operator == 'lap':
            num_ineq = K + 2
            A = np.zeros((num_ineq, K + 1))
            b = np.zeros(num_ineq)

            # 1. w1 + C <= 0 
            A[0, 0] = 1.0
            A[0, -1] = 1.0

            # 2. -w_2:K - C <= 0 
            idx = np.arange(1, K)

            A[idx, idx] = -1.0
            A[idx, -1] = -1.0

            # 3. -C <= 0  AND  C <= 1e5
            A[K, -1] = -1.0

            A[K + 1, -1] = 1.0
            b[K + 1] = 1e5
        elif operator == 'dn':
            num_ineq = 2 * K + 1
            A = np.zeros((num_ineq, K + 1))
            b = np.zeros(num_ineq)

            # 1.-w1 <= 0 (dn)
            A[0, 0] = -1.0

            # 2. -w_2:K - C <= 0 AND  w_2:K - C <= 0
            idx = np.arange(1, K)

            A[idx, idx] = -1.0
            A[idx, -1] = -1.0

            A[idx + K - 1, idx] = 1.0
            A[idx + K - 1, -1] = -1.0

            # 3. -C <= 0  AND  C <= 1e5
            A[2 * K - 1, -1] = -1.0

            A[2 * K, -1] = 1.0
            b[2 * K] = 1e5

        P_t_P = P.T @ P 
        if np.linalg.cond(P_t_P) > 1e14:
            U, S, Vh = np.linalg.svd(A_eq, full_matrices=False)
            
            tol = 1e-12 * S[0] 
            rank = np.sum(S > tol)
            
            A_eq = U.T[:rank, :] @ A_eq
            b_eq = U.T[:rank, :] @ b_eq
                
        sol = solvers.qp(matrix(H), matrix(f), matrix(A), matrix(b), matrix(A_eq), matrix(b_eq))
        if sol['status'] != 'optimal':
            return None

        weights = np.array(sol['x'])[:K, :].T
        return weights.flatten()  / scale_factor

    r = np.linalg.norm(norm_coords, axis=1) # shape: (K,)

    if operator == 'lap':
        dPhi = (4 * kappa**2 + 2 * d * kappa + d - 1) * r**(2 * kappa - 1)
        dPhi = dPhi.reshape(1, -1) 
    elif operator == 'dn':
        coeffs = -(2 * kappa + 1) * r**(2 * kappa - 1)
        dPhi = n_vec @ tangent_basis.T @ (coeffs.reshape(-1, 1) * norm_coords).T # shape: (1, K)

    reg_term = (delta**2) * np.eye(K)
    Phi_inv = np.linalg.solve(Phi.T @ W @ Phi + reg_term, Phi.T @ W)

    P_t_W_P = P.T @ W @ P
    if np.linalg.cond(P_t_W_P) > 1e14:
        P_t_W_P += 1e-12 * np.eye(m)
    
    P_inv_block = np.linalg.solve(P_t_W_P, P.T @ W)

    w_poly = dP @ P_inv_block
    projector = np.eye(K) - P @ P_inv_block
    w_rbf = dPhi @ Phi_inv @ projector
    
    weights = (w_poly + w_rbf) / scale_factor
    return weights.flatten()

def get_stable_weights(stencil_fetcher, weight_kwargs, K_init, expected_sign, opt, gamma=3.0, max_retries_opt=20):
    K_current = K_init
    max_retries_auto_K = 15

    # best of history
    best_ratio = -1.0
    best_weights = None
    best_stencil_ids = None

    # auto K

    for _ in range(max_retries_auto_K + 1):
        stencil_points, stencil_ids = stencil_fetcher(K_current)
        weights = get_operator_weights(stencil=stencil_points, **weight_kwargs)
        
        if K_current == K_init:
            initial_weights = weights
            initial_stencil_ids = stencil_ids

        w_center, w_neighbors = weights[0], weights[1:]
        
        is_correct_sign = (w_center * expected_sign) > 0.0
        ratio = np.abs(w_center) / np.max(np.abs(w_neighbors))
        is_stable = ratio >= gamma

        if is_correct_sign:
            if is_stable:
                return weights, stencil_ids
            elif ratio > best_ratio:
                best_ratio = ratio
                best_weights = weights
                best_stencil_ids = stencil_ids

        K_current += 2

    if opt is None:
        if best_weights is not None:
            return best_weights, best_stencil_ids
        else:
            return initial_weights, initial_stencil_ids 
    else:
        K_current = K_init
        opt_kwargs = weight_kwargs.copy()
        opt_kwargs['opt'] = opt

        # best of history
        best_ratio = -1.0
        best_weights = None
        best_stencil_ids = None
        
        for _ in range(max_retries_opt + 1):
            stencil_points, stencil_ids = stencil_fetcher(K_current)
            weights = get_operator_weights(stencil=stencil_points, **opt_kwargs)

            if weights is not None:
                w_center, w_neighbors = weights[0], weights[1:]
                ratio = np.abs(w_center) / np.max(np.abs(w_neighbors))
                
                is_correct_sign = (w_center * expected_sign) > 0.0
                is_stable = ratio >= gamma

                if is_correct_sign:
                    if is_stable:
                        return weights, stencil_ids
                    elif ratio > best_ratio:
                        best_ratio = ratio
                        best_weights = weights
                        best_stencil_ids = stencil_ids
            
            K_current += 2

        if best_weights is not None:
            return best_weights, best_stencil_ids
        else:
            raise RuntimeError(f"Operator {weight_kwargs.get('operator')} not optimal.")