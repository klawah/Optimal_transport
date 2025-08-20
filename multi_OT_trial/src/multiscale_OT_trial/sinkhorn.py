import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))  # adds 'src' parent
import src.utils.signal_simulation as signal
import src.utils.optimal_transport_computation as opt_t
import src.utils.OT_interpolation as itp

def main():

    N = 100
    D = 2
    N_theta = 21
    theta_seq = np.linspace(0,1,N_theta)

    if D == 2:
        mu_parameters = [
            {'amplitude': 1, 'fc': [0.3,0.3], 'damping': 0}, 
            {'amplitude': 1, 'fc': [-0.2,-0.1], 'damping': 0},
            # {'amplitude': 1.1, 'fc': [-0.2,0.3], 'damping': 0}
        ]
        nu_parameters = [
            {'amplitude': 1, 'fc': [0.1,0.3], 'damping': 0}, 
            {'amplitude': 5, 'fc': [-0.2,-0.4], 'damping': 0}
        ]
    if D == 3:
        mu_parameters = [
        {'amplitude': 1, 'fc': [-0.3,-0.5,-0.1], 'damping': 0},
        #{'amplitude': 1, 'fc': [0.3,0.25,0.45], 'damping': 0}
        ]

        nu_parameters = [
        {'amplitude': 1, 'fc': [0.35, 0.1,0.35], 'damping': 0},
        #{'amplitude': 1, 'fc': [-0.3,-0.20,-0.1], 'damping': 0} 
        ]

    mu = signal.generate_multi_dimensional_sinusoid(D, N, mu_parameters)
    nu = signal.generate_multi_dimensional_sinusoid(D, N, nu_parameters)
    space_shape = np.shape(mu)

    print(mu[0,0])

    # plot_imshow(mu)
    # plot_imshow(nu)
    
    P, mu_coords, nu_coords, log = sinkhorn_log(mu,nu,return_log=True)

    # plot_transport_distributions_sparse(P,mu_coords,nu_coords,space_shape)

    P_coord,P_val = nonzero_coords_and_values(P)
    
    i = np.argwhere(P_coord[:,0]==1)
    print(i)
    sum = np.sum(P_val[i])
    print(sum)
    
    print(log)
    print(P_coord,P_val, np.sum(P_val))
    print(mu_coords,nu_coords)

    itp_seq = itp.interpolate_sparse_OT(P,mu_coords,nu_coords,theta_seq,space_shape)

    
    # plot_sparse_interpolation_freqs(itp_seq,space_shape)
    # plot_sparse_interpolation_overlay(itp_seq,space_shape)

def sinkhorn(mu, nu, epsilon = 1e-2, max_iter = 5000, tol = 1e-6, return_log = False, threshold = 1e-6):
    """
    Entropy reguralised optimal transport using the sinkhorn-knopp algorithm for 
    sparse sourse and target distributions with M_mu and M_nu support points.

    Parameters
    ----------
    mu: Array of shape ((N,)*D)
        Sparse source distribution 
    nu: Array of shape ((N,)*D)
        Sparse target distribution
    epsilon: float
        Entropic regulariiser
    max_iter: int
        Maximum number of iterations
    tol: float
        Relative change stopping criterion
    return_log: boolean
        If True, logs will be returned

    Returns
    -------
    P: Array of shape (M_mu,N_nu)
        Optimal transport plan between support points. P[i,j] corresponds to the transport between 
        source support point i and target support point j
    mu_coords: Array of shape (M_mu, D)
        Source support points in the original space
    mu_coords: Array of shape (M_nu, D)
        Target support points in the original space
    log: dict, optional
        Returns the total transport cost for P and the number of iterations perfomed

    """

    mu_coords, mu_values = nonzero_coords_and_values(mu,threshold)
    nu_coords, nu_values = nonzero_coords_and_values(nu,threshold)
    C = sqeuclidean_dist(mu_coords,nu_coords)
    C_normal = C / np.max(C)

    print(mu_coords,mu_values)
    print(nu_coords,nu_values)

    u = np.ones_like(mu_values)
    v = np.ones_like(nu_values)

    K = np.exp(-C_normal / epsilon)


    for i in range(max_iter):
        u_prev = u.copy()
        v_prev = v.copy()

        u = mu_values / (K @ v)
        v = nu_values / (K.T @ u)

        delta_u = np.max(np.abs(u-u_prev)/np.maximum(np.abs(u_prev), 1e-16))
        delta_v = np.max(np.abs(v-v_prev)/np.maximum(np.abs(v_prev), 1e-16))
        if max(delta_u, delta_v) < tol:
            break
    
    P = u[:, None] * K * v[None, :]  
    cost = np.sum(P * C)

    log = {'cost': cost, 'num_iter': i+1}

    if return_log:
        return P, mu_coords, nu_coords, log
    return P, mu_coords, nu_coords

def sinkhorn_log(mu, nu, epsilon = 1, max_iter = 5000, tol = 1e-6, return_log = False, threshold = 1e-6):
    """
    Entropy reguralised optimal transport using the logarithmic sinkhorn algorithm for 
    sparse sourse and target distributions with M_mu and M_nu support points.

    Parameters
    ----------
    mu: Array of shape ((N,)*D)
        Sparse source distribution 
    nu: Array of shape ((N,)*D)
        Sparse target distribution
    epsilon: float
        Entropic regulariiser
    max_iter: int
        Maximum number of iterations
    tol: float
        Relative change stopping criterion
    return_log: boolean
        If True, logs will be returned

    Returns
    -------
    P: Array of shape (M_mu,N_nu)
        Optimal transport plan between support points. P[i,j] corresponds to the transport between 
        source support point i and target support point j
    mu_coords: Array of shape (M_mu, D)
        Source support points in the original space
    mu_coords: Array of shape (M_nu, D)
        Target support points in the original space
    log: dict, optional
        Returns the total transport cost for P and the number of iterations perfomed

    """

    mu_coords, mu_values = nonzero_coords_and_values(mu,threshold)
    nu_coords, nu_values = nonzero_coords_and_values(nu,threshold)
    C = sqeuclidean_dist(mu_coords,nu_coords)
    C_normal = C / np.max(C)

    u_log = np.zeros_like(mu_values)
    v_log = np.zeros_like(nu_values)

    num_iter = 0
    for i in range(max_iter):
        u_log_prev = u_log.copy()
        v_log_prev = v_log.copy()

        u_log = epsilon * (np.log(mu_values) - logsumexp((-C + v_log[None, :]) / epsilon, axis=1))
        v_log = epsilon * (np.log(nu_values) - logsumexp((-C + u_log[:, None]) / epsilon, axis=0))

        delta_u = np.max(np.abs(u_log - u_log_prev) / np.maximum(np.abs(u_log_prev), 1e-16))
        delta_v = np.max(np.abs(v_log - v_log_prev) / np.maximum(np.abs(v_log_prev), 1e-16))
        if max(delta_u, delta_v) < tol:
                break

    logP = (u_log[:, None] + v_log[None, :] - C) / epsilon
    m = np.max(logP)
    P = np.exp(logP - m)
    P *= np.exp(m)  
    cost = np.sum(P * C)

    log = {'cost': cost, 'num_iter': i+1}

    if return_log:
        return P, mu_coords, nu_coords, log
    return P, mu_coords, nu_coords

def sinkhorn_sparse(mu, nu, epsilon=1e-2, max_iter=5000, tol=1e-6, return_log=False, threshold = 1e-6):
    """
    Entropy reguralised optimal transport using the sinkhorn-knopp algorithm for 
    sparse sourse and target distributions with M_mu and M_nu support points. Differs 
    from standard sinkhorn in that it does not construct the full K

    Parameters
    ----------
    mu: Array of shape ((N,)*D)
        Sparse source distribution 
    nu: Array of shape ((N,)*D)
        Sparse target distribution
    epsilon: float
        Entropic regulariiser
    max_iter: int
        Maximum number of iterations
    tol: float
        Relative change stopping criterion
    return_log: boolean
        If True, logs will be returned

    Returns
    -------
    P: Array of shape (M_mu,N_nu)
        Optimal transport plan between support points. P[i,j] corresponds to the transport between 
        source support point i and target support point j
    mu_coords: Array of shape (M_mu, D)
        Source support points in the original space
    mu_coords: Array of shape (M_nu, D)
        Target support points in the original space
    log: dict, optional
        Returns the total transport cost for P and the number of iterations perfomed

    """
    
    # Get nonzero entries
    mu_coords, mu_values = nonzero_coords_and_values(mu,threshold)
    nu_coords, nu_values = nonzero_coords_and_values(nu,threshold)
    
    # Squared Euclidean distances between sparse points
    C = sqeuclidean_dist(mu_coords, nu_coords)
    C_normal = C / np.max(C)
    
    # Initialize scaling factors
    u = np.ones_like(mu_values)
    v = np.ones_like(nu_values)
    
    for i in range(max_iter):
        u_prev = u.copy()
        v_prev = v.copy()
        
        # Update u and v without forming full K
        # For each row i: u[i] = mu[i] / sum_j exp(-C[i,j]/epsilon) * v[j]
        u = mu_values / np.sum(np.exp(-C_normal / epsilon) * v[None, :], axis=1)
        v = nu_values / np.sum(np.exp(-C_normal / epsilon) * u[:, None], axis=0)
        
        # Relative change
        delta_u = np.max(np.abs(u-u_prev)/np.maximum(np.abs(u_prev), 1e-16))
        delta_v = np.max(np.abs(v-v_prev)/np.maximum(np.abs(v_prev), 1e-16))
        if max(delta_u, delta_v) < tol:
            break
    
    # Construct sparse transport plan
    P = u[:, None] * np.exp(-C_normal / epsilon) * v[None, :]
    cost = np.sum(P * C)
    
    log = {'cost': cost, 'num_iter': i+1}
    
    if return_log:
        return P, mu_coords, nu_coords, log
    return P, mu_coords, nu_coords

def nonzero_coords_and_values(A, threshold = 1e-6):
    """
    
    Parameters
    ----------
    A: Sparse matrix in dimension D with equal axis lengths. Shape ((N,)*D)
    threshold: All values <= threshold are considered to be noise and treated as zero-entries

    Returns
    -------
    values: An array of all on-zero values of A. Shape (M,)
    coords: An array of all non-zero coordinates of A (M,D)
    """

    coords = np.argwhere(A > threshold)
    values = A[tuple(coords.T)]

    return coords, values

def sqeuclidean_dist(A,B):
    """
    
    Parameters
    ----------
    A: First array of coordinate points. Shape (N,D)
    B: Second array of coordinate points. Shape (M,D)

    Returns
    -------
    sq_dist: Array of pointwise distances between A and B. Shape (N,M)
    """

    if np.shape(A)[1] != np.shape(B)[1]:
        raise ValueError('A and B musta have same dimension.')
    
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]  
    sq_dist = np.sum(diff**2, axis=-1)             

    return sq_dist

def plot_sparse_interpolation(sparse_sequence, space_shape):
    D = len(space_shape)
    N_theta = len(sparse_sequence)
    
    colors = plt.cm.viridis(np.linspace(0, 1, N_theta))  # one color per step

    if D == 2:
        plt.figure(figsize=(6,6))
        ax = plt.subplot(projection='polar')
        
        for t, (coords, values) in enumerate(sparse_sequence):
            theta = 2 * np.pi * coords[:, 0] / space_shape[0]
            r = values / values.max()
            ax.scatter(theta, r, color=colors[t], label=f"t={t}", alpha=0.7)
        
        ax.set_title("Sparse OT Interpolation (D=2)")
        ax.legend()
        plt.show()
    
    elif D == 3:
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot unit sphere for reference
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
        xs = np.sin(v)*np.cos(u)
        ys = np.sin(v)*np.sin(u)
        zs = np.cos(v)
        ax.plot_wireframe(xs, ys, zs, color='gray', alpha=0.2)
        
        for t, (coords, values) in enumerate(sparse_sequence):
            phi = 2 * np.pi * coords[:, 0] / space_shape[0]     # azimuth
            theta = np.pi * coords[:, 1] / space_shape[1]       # polar
            r = values / values.max()
            
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            
            ax.scatter(x, y, z, color=colors[t], label=f"t={t}", s=50, alpha=0.7)
        
        ax.set_title("Sparse OT Interpolation (D=3)")
        ax.legend()
        plt.show()


def plot_imshow(psd, name = 'spectrum'):

    plt.figure(figsize=(6, 5))
    plt.imshow(psd, extent=[-0.5, 0.5, 0.5, -0.5], origin='upper', aspect='auto', cmap='viridis')
    plt.colorbar(label='Normalized Power')
    plt.title(f'Power Spectral Density ({name})')
    plt.xlabel('Frequency axis 1')
    plt.ylabel('Frequency axis 2')
    plt.tight_layout()
    plt.show()

def plot_transport_distributions_sparse(P, mu_coords, nu_coords, grid_shape):
    """
    Visualize where mass starts and ends based on the transport plan P
    with consistent color scaling for both plots.
    
    Parameters
    ----------
    P: array, shape (M_mu, M_nu)
        Optimal transport plan between sparse source and target
    mu_coords: array, shape (M_mu, D)
        Source support coordinates
    nu_coords: array, shape (M_nu, D)
        Target support coordinates
    grid_shape: tuple of length D
        Shape of the full grid (e.g., (N,N) or (N,N,N))
    """

    D = len(grid_shape)

    # Compute marginal masses
    source_mass = np.sum(P, axis=1)  # mass leaving each source point
    target_mass = np.sum(P, axis=0)  # mass arriving at each target point

    # Embed into full grid
    source_mass_grid = np.zeros(grid_shape)
    target_mass_grid = np.zeros(grid_shape)

    for coord, mass in zip(mu_coords, source_mass):
        source_mass_grid[tuple(coord)] = mass

    for coord, mass in zip(nu_coords, target_mass):
        target_mass_grid[tuple(coord)] = mass

    # Visualization
    vmax = max(source_mass_grid.max(), target_mass_grid.max())
    vmin = 0

    if D == 2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        im0 = axes[0].imshow(source_mass_grid, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title('Source Mass Distribution')
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(target_mass_grid, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title('Target Mass Distribution')
        plt.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        plt.show()

    elif D == 3:
        # show max projections for quick visualization
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        for i, grid in enumerate([source_mass_grid, target_mass_grid]):
            axes[0, i].imshow(grid.max(axis=0), cmap="viridis", vmin=vmin, vmax=vmax)
            axes[0, i].set_title(["Source", "Target"][i] + " (proj x)")
            
            axes[1, i].imshow(grid.max(axis=1), cmap="viridis", vmin=vmin, vmax=vmax)
            axes[1, i].set_title(["Source", "Target"][i] + " (proj y)")

        plt.tight_layout()
        plt.show()
    else:
        raise NotImplementedError("Visualization only implemented for 2D or 3D grids")

    return source_mass_grid, target_mass_grid

def plot_sparse_interpolation_overlay(sparse_sequence, space_shape):
    D = len(space_shape)
    N_theta = len(sparse_sequence)
    
    colors = plt.cm.viridis(np.linspace(0, 1, N_theta))  # one color per step

    if D == 2:
        plt.figure(figsize=(6,6))
        ax = plt.subplot(projection='polar')
        
        for t, (coords, values) in enumerate(sparse_sequence):
            freqs = (coords - np.array(space_shape)//2) / np.array(space_shape)  
            angles = np.arctan2(freqs[:, 1], freqs[:, 0]) 
            r_plot = values          
            ax.scatter(angles, r_plot, color=colors[t], label=f"t={t}", alpha=0.7)
        
        ax.set_title("Sparse OT Interpolation (D=2)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
    
    elif D == 3:
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot unit sphere for reference
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
        xs = np.sin(v)*np.cos(u)
        ys = np.sin(v)*np.sin(u)
        zs = np.cos(v)
        ax.plot_wireframe(xs, ys, zs, color='gray', alpha=0.2)
        
        for t, (coords, values) in enumerate(sparse_sequence):
            freqs = (coords - np.array(space_shape)//2) / np.array(space_shape)  
            dirs = freqs / np.linalg.norm(freqs, axis=1, keepdims=True)               

            r_plot = values                               
            # Cartesian coordinates
            x = dirs[:,0] * r_plot
            y = dirs[:,1] * r_plot
            z = dirs[:,2] * r_plot

            ax.scatter(x, y, z, color=colors[t], alpha=0.7)
        
        ax.set_title("Sparse OT Interpolation (D=3)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()


def plot_sparse_interpolation_freqs(sparse_sequence, space_shape, scale_mass=100):
    """
    Plot sparse OT interpolation steps in centered frequency space with
    reference unit circle/sphere and radial lines.

    Parameters
    ----------
    sparse_sequence : list of (coords_sparse, values_sparse)
        Each element corresponds to one interpolation step:
        coords_sparse: (M, D) array of integer coordinates
        values_sparse: (M,) array of masses for those coordinates
    space_shape : tuple
        Shape of the original D-dimensional grid.
    scale_mass : float
        Scaling factor for scatter point sizes.
    """
    D = sparse_sequence[0][0].shape[1]
    N_theta = len(sparse_sequence)
    
    space_shape = np.array(space_shape)
    freq_center = space_shape // 2
    colors = plt.cm.viridis(np.linspace(0, 1, N_theta))

    if D == 2:
        fig, ax = plt.subplots(figsize=(6, 6))
        
        
        
        for t, (coords_sparse, values_sparse) in enumerate(sparse_sequence):
            freqs = (coords_sparse - freq_center) / space_shape  # center at zero freq
            radii = np.linalg.norm(freqs, axis=1)
            
            # Scatter points
            ax.scatter(freqs[:, 0].T, freqs[:, 1],
                       s=scale_mass * (values_sparse / np.max(values_sparse)),
                       color=colors[t], alpha=0.6, label=f"step {t}")
            

        ax.set_xlabel("f1 (cycles/unit)")
        ax.set_ylabel("f2 (cycles/unit)")
        ax.set_aspect('equal')
        ax.set_xlim(-0.55, 0.55)
        ax.set_ylim(0.55, -0.55)
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

    elif D == 3:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Unit sphere
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
        xs = np.cos(u) * np.sin(v)
        ys = np.sin(u) * np.sin(v)
        zs = np.cos(v)
        ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.3)
        
        for t, (coords_sparse, values_sparse) in enumerate(sparse_sequence):
            freqs = (coords_sparse - freq_center) / space_shape
            norms = np.linalg.norm(freqs, axis=1)
            
            # Scatter points
            ax.scatter(freqs[:, 0], freqs[:, 1], freqs[:, 2],
                       s=scale_mass * (values_sparse / np.max(values_sparse)),
                       color=colors[t], alpha=0.6, label=f"step {t}")
            
            # Radial lines from point to sphere
            for (fx, fy, fz) in freqs:
                if fx == 0 and fy == 0 and fz == 0:
                    continue
                norm = np.sqrt(fx**2 + fy**2 + fz**2)
                unit_x, unit_y, unit_z = fx / norm, fy / norm, fz / norm
                ax.plot([fx, unit_x], [fy, unit_y], [fz, unit_z],
                        color=colors[t], alpha=0.3)

        ax.set_xlabel("f1 (cycles/unit)")
        ax.set_ylabel("f2 (cycles/unit)")
        ax.set_zlabel("f3 (cycles/unit)")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_zlim(-1.05, 1.05)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

    else:
        raise ValueError("D must be 2 or 3.")

if __name__ == "__main__":
    main()