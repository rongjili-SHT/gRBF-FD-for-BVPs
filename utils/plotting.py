import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.colors import LogNorm

def plot_convergence(N_vals, err_stat_list, log_normal=False, title=None, ref_list=None, filename=None):
    N_vals = np.array(N_vals)

    fig, ax = plt.subplots(figsize=(9, 9))

    for stat in err_stat_list:
        if log_normal:
            log_mean = stat['mean']
            log_std = stat['std']

            line, = plt.loglog(N_vals, 10 ** log_mean, marker='o', **stat['plot_kwargs'])
            
            color = line.get_color()
            
            plt.fill_between(
                N_vals, 
                10 ** (log_mean - log_std), 
                10 ** (log_mean + log_std), 
                color=color, 
                alpha=0.2,
                edgecolor=None
            )

        else:
            mean = stat['mean']
            std = stat['std']

            line, = plt.loglog(N_vals, mean, marker='o', **stat['plot_kwargs'])
            
            color = line.get_color()
            
            plt.fill_between(
                N_vals, 
                mean - std, 
                mean + std, 
                color=color, 
                alpha=0.2,
                edgecolor=None
            )

    ax.set_xscale('log')
    ax.set_yscale('log')

    def plot_ref_line(anchor, slope, label):
        N0, E0 = anchor
        C = E0 / (N0 ** slope)
        ref_line = C * (N_vals ** slope)

        ax.plot(N_vals, ref_line, 'k--', linewidth=0.5,)

        ax.annotate(
            label, 
            xy=anchor,
            xytext=(50, -5),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=16,
        )

    if ref_list is not None:
        for ref in ref_list:
            plot_ref_line(*ref)

    ax.set_xticks(N_vals)
    ax.set_xticklabels(N_vals)
    ax.minorticks_off()

    ax.legend()
    ax.grid(True)

    if title is not None:
        plt.title(title)

    if filename is not None:
        fig.savefig(f'{filename}.pdf', dpi=300, bbox_inches='tight')

    plt.show()

def plot_point_cloud(points, id_list, filename=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for ids in id_list:
        ax.scatter(points[ids, 0], points[ids, 1], points[ids, 2], s=10)
    
    ax.set_aspect('equal')
    
    if filename is not None:
            fig.savefig(f'./static/{filename}.pdf', dpi=300, bbox_inches='tight')

    plt.show()

def plot_manifold(points, color_vals, filename=None, figsize=(9, 9), shrink=0.8, look_at=(30, -60), color_log=False, add=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color_vals, s=10, cmap='coolwarm')
    cbar = plt.colorbar(sc, ax=ax, shrink=shrink, pad=0.1)

    if add == 'pipe':
        a = 8.0
        b = 1.0
        denom = np.sqrt(a**2 + b**2)
        alpha = a / denom
        beta = b / denom
        num_turns = 1.5
        omega_max = num_turns * 2 * np.pi
        R_start_scale = 3.0
        R_end_scale = 3.0

        theta_grid = np.linspace(0, 2 * np.pi, 60)
        omega_grid = np.linspace(0, omega_max, 200)
        Theta, Omega = np.meshgrid(theta_grid, omega_grid)

        scale = R_start_scale - (R_start_scale - R_end_scale) * (Omega / omega_max)
        R_base = 0.6 + 0.075 * np.sin(5 * Theta)
        R = R_base * scale

        X = a * np.cos(Omega) + R * (beta * np.sin(Theta) * np.sin(Omega) - np.cos(Theta) * np.cos(Omega))
        Y = a * np.sin(Omega) - R * (beta * np.sin(Theta) * np.cos(Omega) + np.cos(Theta) * np.sin(Omega))
        Z = b * Omega + R * alpha * np.sin(Theta)

        ax.plot_surface(X, Y, Z, color='white', alpha=0.3, edgecolor='none', rstride=2, cstride=2)
    elif add == 'rbc':
        r_val = 3.91 / 3.39
        c0 = 0.81 / 3.39
        c2 = 7.83 / 3.39
        c4 = -4.39 / 3.39

        # 3. 创建参数网格
        # theta: 0 到 pi/2 (只取上半部分)
        theta_grid = np.linspace(0, np.pi / 2, 60)
        phi_grid = np.linspace(-np.pi, np.pi, 100)
        Theta, Phi = np.meshgrid(theta_grid, phi_grid)

        # 4. 计算三维坐标 (X, Y, Z)
        X = r_val * np.cos(Theta) * np.cos(Phi)
        Y = r_val * np.cos(Theta) * np.sin(Phi)
        Z = 0.5 * np.sin(Theta) * (c0 + c2 * np.cos(Theta)**2 + c4 * np.cos(Theta)**4)

        # 5. 绘制曲面
        ax.plot_surface(X, Y, Z, color='white', alpha=0.3, edgecolor='none', rstride=2, cstride=2)

    if color_log:
        formatter = ticker.ScalarFormatter(useMathText=True) # 使用数学字体 (10^{-5} 而不是 1e-5)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0)) # 强制对所有数值使用科学计数法指数
        
        cbar.ax.yaxis.set_major_formatter(formatter)

    ax.set_xlabel(r'$x$', labelpad=20)
    ax.set_ylabel(r'$y$', labelpad=20)
    # ax.set_zlabel(r'$z$')

    ax.set_aspect('equal')

    elev, azim = look_at
    ax.view_init(elev=elev, azim=azim)

    if filename is not None:
            fig.savefig(f'./static/{filename}.pdf', dpi=300, bbox_inches='tight')

    plt.show()

def plot_weights(weights, filename=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ids = np.arange(len(weights))
    bars = ax.bar(ids, weights)
    bars[0].set_color('red')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    
    ax.set_xlabel(r'$k$ nearest neighbor')
    ax.set_ylabel(r'coefficient $w_k$')
    plt.tight_layout()

    if filename is not None:
        fig.savefig(f'../static/{filename}.pdf', dpi=300, bbox_inches='tight')

    plt.show()

def plot_error_distribution(params, error_vals, filename=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sc = ax.scatter(
        params[:, 0], params[:, 1], c=error_vals,
        s=100,
        cmap='coolwarm', 
        norm=LogNorm(),
        marker='o',
        edgecolors='none',
        alpha=0.8
    )

    cbar = plt.colorbar(sc, ax=ax)
    # cbar.set_label('Forward Error', rotation=270, labelpad=20)

    ax.set_xlim(0.0, 2*np.pi)
    ax.set_ylim(0.0, np.pi)

    ax.set_yticks([0, np.pi/2, np.pi])
    ax.set_yticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])
    ax.set_xticks([0, np.pi/2, np.pi, 3 * np.pi/2, 2*np.pi])
    ax.set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    
    ax.set_xlabel(r'$\psi^1$')
    ax.set_ylabel(r'$\psi^2$', rotation=0)
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    # ax.set_aspect('equal')

    ax.minorticks_off()

    if filename is not None:
        fig.savefig(f'../static/{filename}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def plot_bad_points(params, pos_idx, ill_idx, filename=None):
    fig, ax = plt.subplots(figsize=(8, 6))

    pos_points = params[pos_idx]
    ill_points = params[ill_idx]
    
    ax.scatter(
        params[:, 0], params[:, 1],
        s=20,
        c='green',
        marker='o',
        edgecolors='none',
        alpha=0.1,
        label=r'$w_1 < 0 $ and $\gamma \geq 1$'
    )

    ax.scatter(pos_points[:, 0], pos_points[:, 1], label=r'$w_1 \geq 0$',
               c='red', marker='x', s=120, alpha=1.0, zorder=5)
    

    ax.scatter(ill_points[:, 0], ill_points[:, 1], label=r'$\gamma < 1$',
               edgecolors='blue', marker='o', s=60, alpha=0.9, facecolors='none',
              zorder=5)
    
    ax.set_xlim(0.0, 2*np.pi)
    ax.set_ylim(0.0, np.pi)

    ax.set_xlabel(r'$\psi^1$')
    ax.set_ylabel(r'$\psi^2$', rotation=0)

    ax.set_yticks([0, np.pi/2, np.pi])
    ax.set_yticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])
    ax.set_xticks([0, np.pi/2, np.pi, 3 * np.pi/2, 2*np.pi])
    ax.set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])

    leg = ax.legend()
    for lh in leg.legend_handles: 
        lh.set_alpha(1.0)

    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    plt.tight_layout()
    if filename is not None:
        fig.savefig(f'../static/{filename}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def plot_k_vals(k_vals):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(k_vals, marker='o', ms=3, linestyle='--', label='Trend', alpha=0.5)
    plt.show()