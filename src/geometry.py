import sympy as sp
import numpy as np
from scipy.spatial import cKDTree
import jax
import jax.numpy as jnp

class Manifold:
    def __init__(self, xi_syms, x_syms):
        """
        Args:
            xi (list): list of parameters
            x (list)
        """
        self.d = len(xi_syms) # manifold dimension
        self.n = len(x_syms) # ambient dimension

        self.xi = sp.Matrix(xi_syms) # shape: (d,)
        self.x = sp.Matrix(x_syms) # shape: (n, )

        self.id_interior = None
        self.id_boundary = None

    def compute(self):
        self.J = self.x.jacobian(self.xi)
        self.G = sp.simplify(self.J.T @ self.J)

        self.G_inv = sp.simplify(self.G.inv())
        self.g = sp.simplify(self.G.det())

    def get_local_basis(self, xi_val):
        if not hasattr(self, 'J_func'):
            self.J_func = sp.lambdify(self.xi, self.J, 'numpy')

        J_val = self.J_func(*xi_val) # shape: (n, d)

        Q, _ = np.linalg.qr(J_val, mode='complete')

        tangent_basis = Q[:, :self.d].T # Shape (d, n)
        normal_basis = Q[:, self.d:].T  # Shape (n-d, n)

        return tangent_basis, normal_basis

    def get_gradient(self, u_sym):
        du_dxi = sp.Matrix([sp.diff(u_sym, xii) for xii in self.xi])
        return sp.simplify(self.J @ self.G_inv @ du_dxi)

    def get_laplacian(self, u_sym):
        du_dxi = sp.Matrix([sp.diff(u_sym, xii) for xii in self.xi])
        V_contra = self.G_inv @ du_dxi 

        scaled_V = sp.sqrt(self.g) * V_contra
        
        divergence_sum = 0
        for i in range(self.d):
            divergence_sum += sp.diff(scaled_V[i], self.xi[i])
            
        laplacian = (1 / sp.sqrt(self.g)) * divergence_sum
        
        return sp.simplify(laplacian)

    def sample(self, xi_ranges, num_points):
        x_syms = [self.x[i] for i in range(self.n)]
        self.x_func = sp.lambdify(self.xi, x_syms, 'numpy')

        params = np.zeros((num_points, self.d))

        for i, (t_min, t_max) in enumerate(xi_ranges):
            params[:, i] = np.random.uniform(t_min, t_max, size=num_points)

        args = [params[:, i] for i in range(self.d)]
        raw_coords = self.x_func(*args)
        coords_broadcasted = [np.broadcast_to(c, (num_points,)) for c in raw_coords]
        
        self.params = params # shape: (num_points, d)
        self.points = np.column_stack(coords_broadcasted) # shape: (num_points, n)

    def build_tree(self):
        self.tree = cKDTree(self.points)

    def build_enhanced_tree(self, center_id, enhance_direction=None, s=3.0):
        interior_points = self.points[self.id_interior]
        e_vec = enhance_direction

        diff = interior_points - self.points[center_id]
        # s: enhance strength
        proj_lengths = diff @ e_vec  
        enhanced_diff = diff - (1.0 - 1/s) * np.outer(proj_lengths, e_vec)

        return cKDTree(enhanced_diff)
    
    def get_in_stencil(self, center_id, K):
        _, stencil_ids = self.tree.query(self.points[center_id], K)
        return self.points[stencil_ids], stencil_ids

    def get_bd_stencil(self, center_id, K, method='restricted', enhanced_tree=None):
        if method == 'direct':
            _, stencil_ids = self.tree.query(self.points[center_id], K)
            return self.points[stencil_ids], stencil_ids
        elif method == 'restricted':
            _, interior_idx = enhanced_tree.query(np.zeros(3), k=K-1)

            stencil_ids = np.zeros(K, dtype=int)
            stencil_ids[0] = center_id
            stencil_ids[1:] = self.id_interior[interior_idx]

            return self.points[stencil_ids], stencil_ids

class ManifoldJax:
    def __init__(self, d, n, x_map):
        self.d = d
        self.n = n
        self.x_map = x_map
        
        self.J_map = jax.jit(jax.jacfwd(x_map))

        self.id_interior = None
        self.id_boundary = None
        self.params = None
        self.points = None

    def get_local_basis(self, xi_val):
        J_val = np.array(self.J_map(jnp.array(xi_val)))
        
        Q, _ = np.linalg.qr(J_val, mode='complete')

        tangent_basis = Q[:, :self.d].T # Shape (d, n)
        normal_basis = Q[:, self.d:].T  # Shape (n-d, n)

        return tangent_basis, normal_basis

    def build_tree(self):
        self.tree = cKDTree(self.points)

    def build_enhanced_tree(self, center_id, enhance_direction=None, s=3.0):
        interior_points = self.points[self.id_interior]
        e_vec = enhance_direction

        diff = interior_points - self.points[center_id]
        # s: enhance strength
        proj_lengths = diff @ e_vec  
        enhanced_diff = diff - (1.0 - 1/s) * np.outer(proj_lengths, e_vec)

        return cKDTree(enhanced_diff)
    
    def get_in_stencil(self, center_id, K):
        _, stencil_ids = self.tree.query(self.points[center_id], K)
        return self.points[stencil_ids], stencil_ids

    def get_bd_stencil(self, center_id, K, method='restricted', enhanced_tree=None):
        if method == 'direct':
            _, stencil_ids = self.tree.query(self.points[center_id], K)
            return self.points[stencil_ids], stencil_ids
        elif method == 'restricted':
            _, interior_idx = enhanced_tree.query(np.zeros(3), k=K-1)

            stencil_ids = np.zeros(K, dtype=int)
            stencil_ids[0] = center_id
            stencil_ids[1:] = self.id_interior[interior_idx]

            return self.points[stencil_ids], stencil_ids