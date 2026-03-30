from src import *
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg
import sympy as sp
import numpy as np
import os
import dill

def generate_upper_rbc(N):
    """
    生成上半部分红细胞 (Upper RBC) 流形，并在赤道 theta=0 处截断生成边界。
    """
    # RBC 几何参数
    r_val = 3.91 / 3.39
    c0 = 0.81 / 3.39
    c2 = 7.83 / 3.39
    c4 = -4.39 / 3.39

    theta, phi = sp.symbols('theta phi', real=True)

    # 参数化方程
    x_sym = r_val * sp.cos(theta) * sp.cos(phi)
    y_sym = r_val * sp.cos(theta) * sp.sin(phi)
    z_sym = 0.5 * sp.sin(theta) * (c0 + c2 * sp.cos(theta)**2 + c4 * sp.cos(theta)**4)

    manifold = Manifold([theta, phi], [x_sym, y_sym, z_sym])
    manifold.compute()

    # 参数域：上半部分
    theta_range = (0, np.pi / 2)
    phi_range = (-np.pi, np.pi)

    # 估算边界点数量 (依据周长与面积比例经验设置)
    num_boundary = int(np.sqrt(N) * 3)
    num_interior = N - num_boundary

    # 采样内部点
    manifold.sample([theta_range, phi_range], num_interior)

    # 采样边界点 (theta = 0)
    x_sym_bdry = x_sym.subs(theta, 0)
    y_sym_bdry = y_sym.subs(theta, 0)
    z_sym_bdry = z_sym.subs(theta, 0)
    
    boundary = Manifold([phi], [x_sym_bdry, y_sym_bdry, z_sym_bdry])
    boundary.sample([phi_range], num_boundary)

    # 将 theta=0 插入边界的参数矩阵第一列
    bdry_params = np.insert(boundary.params, 0, values=0.0, axis=1)

    # 合并内部点与边界点
    manifold.params = np.vstack([manifold.params, bdry_params])
    manifold.points = np.vstack([manifold.points, boundary.points])

    id_interior = np.arange(num_interior)
    id_boundary = np.arange(num_interior, N)

    # 计算边界处的余法向向量 (Conormal vector)
    # 在赤道处 (theta=0)，流形向外延伸的余法向正指向 z 轴负方向
    n_vecs = np.zeros((N, manifold.n))
    n_vecs[id_boundary] = [0.0, 0.0, -1.0]

    manifold.id_interior = id_interior
    manifold.id_boundary = id_boundary
    manifold.build_tree()

    return manifold, id_interior, id_boundary, n_vecs


def compute_mms_rbc(manifold, n_vecs, cache_file='./data/rbc_mms_funcs.dill'):
    """
    计算红细胞流形上的 Manufactured Solution
    """
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cached_funcs = dill.load(f)
        u_func, f_func, u_lap_func, u_grad_func = cached_funcs
    else:
        theta, phi = manifold.xi
        x_sym, y_sym, z_sym = manifold.x
        # 给定的精确解
        u_sym = sp.sin(x_sym) * sp.cos(y_sym)

        # Sympy 符号计算：Laplacian 和 Gradient
        u_lap_sym = manifold.get_laplacian(u_sym)
        u_grad_sym = manifold.get_gradient(u_sym)
        f_sym = -u_lap_sym

        # Lambdify：将 Sympy 表达式转换为 Numpy 快速执行函数
        u_func = sp.lambdify((theta, phi), u_sym, 'numpy')
        f_func = sp.lambdify((theta, phi), f_sym, 'numpy')
        u_lap_func = sp.lambdify((theta, phi), u_lap_sym, 'numpy')
        u_grad_func = sp.lambdify((theta, phi), u_grad_sym, 'numpy')

        funcs_to_cache = (u_func, f_func, u_lap_func, u_grad_func)
        with open(cache_file, 'wb') as f:
            dill.dump(funcs_to_cache, f)

    # 提取评估点
    tt = manifold.params[:, 0]
    pp = manifold.params[:, 1]

    u_vals = u_func(tt, pp)
    f_vals = f_func(tt, pp)
    u_lap_vals = u_lap_func(tt, pp)

    # 计算法向导数并构建 Robin 边界值 g = u + du/dn
    u_grad_vals = u_grad_func(tt, pp).squeeze().T # shape: (N, n)
    g_vals = u_vals + np.sum(n_vecs * u_grad_vals, axis=1) # shape: (N,)

    return u_vals, f_vals, u_lap_vals, u_grad_vals, g_vals


def solve_poisson_robin_schur(L, D_n, f, g, id_interior, id_boundary, N, require_st=False):
    """
    求解 Robin 边界问题的 Schur 补系统：
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

    f_I = f[id_interior]
    g_B = g[id_boundary]

    # 构建 Schur 补系统
    A_prime = A_II - A_IB @ B_BB_inv @ B_BI
    b_prime = f_I - A_IB @ (B_BB_inv @ g_B)

    # 求解内部和边界解
    u_num_interior = splinalg.spsolve(A_prime, b_prime)
    u_num_boundary = B_BB_inv @ (g_B - B_BI @ u_num_interior)

    u_num = np.zeros(N)
    u_num[id_interior] = u_num_interior
    u_num[id_boundary] = u_num_boundary

    if require_st:
        return u_num, np.linalg.norm(np.linalg.inv(A_prime.toarray()), ord=np.inf)
    return u_num


def robin_semi_rbc(N=6400, l=4, K=25, l_grad=4, K_grad=35, lap_opt='qp', dn_opt='qp', seed=None):
    """
    主流程：生成 RBC 流形 -> 准备 MMS 解 -> 组装 Laplacian 和 Neumann 离散算子 -> 求解 -> 计算误差验证
    """
    #-- PARAMETERS --#
    kappa = 3
    delta = 1e-5

    if seed is not None:
        np.random.seed(seed)

    #-- GEOMETRY --#
    manifold, id_interior, id_boundary, n_vecs = generate_upper_rbc(N)

    num_interior = len(id_interior)
    num_boundary = len(id_boundary)

    #-- MANUFACTURED SOLUTION --#
    u_vals, f_vals, u_lap_vals, u_grad_vals, g_vals = compute_mms_rbc(manifold, n_vecs)

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
    
    # 求解
    u_num = solve_poisson_robin_schur(L, D_n, f_vals, g_vals, id_interior, id_boundary, N)

    #-- VALIDATION --#
    fe_interior = np.abs(L[id_interior, :].dot(u_vals) - u_lap_vals[id_interior])
    fe_interior_l2 = np.sqrt(np.sum(fe_interior ** 2) / num_interior)

    fe_boundary = np.abs(D_n[id_boundary, :].dot(u_vals) - np.sum(n_vecs[id_boundary] * u_grad_vals[id_boundary], axis=1))
    fe_boundary_l2 = np.sqrt(np.sum(fe_boundary ** 2) / num_boundary)

    ie = np.abs(u_num - u_vals) # shape: (N,)
    ie_l2 = np.sqrt(np.sum(ie ** 2) / N)

    return fe_interior_l2, fe_boundary_l2, ie_l2

if __name__ == "__main__":
    fe_interior_l2, fe_boundary_l2, ie_l2 = robin_semi_rbc(N=25600)
    print(f"FE_IN: {fe_interior_l2:.3e} FE_BD: {fe_boundary_l2:.3e} IE: {ie_l2:.3e}")