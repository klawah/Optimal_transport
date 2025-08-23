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
            {'amplitude': 2, 'fc': [-0.2,-0.4], 'damping': 0}
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

    # plot_imshow(mu)
    # plot_imshow(nu)
    
    threshold = 1e-6
    mu_coords, mu_values = nonzero_indices_and_values(mu,threshold)
    nu_coords, nu_values = nonzero_indices_and_values(nu,threshold)

    P, mu_coords, nu_coords, log = sinkhorn(mu_coords, mu_values,nu_coords, nu_values,return_log=True)

    
    plot_sparse_transport_marginals(P,mu_coords,nu_coords,space_shape)

    P_coord,P_val = nonzero_indices_and_values(P)

    itp_seq = itp.interpolate_sparse_OT(P,mu_coords,nu_coords,theta_seq,space_shape)

    # plot_sparse_interpolation_freqs(itp_seq,space_shape)
    # plot_sparse_interpolation_polar(itp_seq,space_shape)

########## Sinkhorn functions ##########

def sinkhorn(mu_coords,mu_values, nu_coords,nu_values, epsilon = 1e-1, max_iter = 5000, tol = 1e-3, return_log = False):
    """
    Entropy reguralised optimal transport using the sinkhorn-knopp algorithm for 
    sparse sourse and target distributions with M_mu and M_nu support points.

    Parameters
    ----------
    mu_coords, mu_values: Arrays of shape ((M_mu,)*D)
        Coordinates and respective densities of the source support points 
    nu_coords, nu_values: Arrays of shape ((M_nu,)*D)
        Coordinates and respective densities of the target support points 
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
        Optimal transport plan between source and target support points. For P[i,j], 
        the indices i, j coorespond to support point indices in mu_coords and nu_coords
    mu_coords: Array of shape (M_mu, D)
        Source support point coordinates in the original space
    nu_coords: Array of shape (M_nu, D)
        Target support point coordinates in the original space
    log: dict, optional
        Returns specifications for the optimisation

    """

    # Cost matrix for the support points
    C = sqeuclidean_dist(mu_coords,nu_coords) 
    C_normal = C / np.max(C) # Normalize the cost to make the exponential stable

    # Initialize scaling factors vith ones
    u = np.ones_like(mu_values) 
    v = np.ones_like(nu_values) 

    K = np.exp(-C_normal / epsilon) # Kernel matrix for the support points

    # Sinkhorn-Knopp iterations
    for i in range(max_iter):
        u_prev = u.copy()
        v_prev = v.copy()

        # Update u and v
        u = mu_values / (K @ v) 
        v = nu_values / (K.T @ u) 

        # Check relate change stopping criterion
        delta_u = np.max(np.abs(u-u_prev)/np.maximum(np.abs(u_prev), 1e-16))
        delta_v = np.max(np.abs(v-v_prev)/np.maximum(np.abs(v_prev), 1e-16))
        if max(delta_u, delta_v) < tol:
            break
    
    # Construct the sparse transport plan
    P = u[:, None] * K * v[None, :]  # P[i,j] = u[i] * K[i,j] * v[j]
    cost = np.sum(P * C) # Total transport cost
    
    if return_log:
        log = {'cost': cost, 'num_iter': i+1, 'epsilon': epsilon, 'tol': tol, 'mu_shape': mu_coords.shape, 'nu_shape': nu_coords.shape}
        return P, mu_coords, nu_coords, log
    return P, mu_coords, nu_coords

def sinkhorn_log(mu_coords, mu_values, nu_coords, nu_values, epsilon = 1e-1, max_iter = 5000, tol = 1e-3, return_log = False):
    """
    Entropy reguralised optimal transport using the logarithmic sinkhorn algorithm for 
    sparse sourse and target distributions with M_mu and M_nu support points.

    Parameters
    ----------
    mu_coords, mu_values: Arrays of shape ((M_mu,)*D)
        Coordinates and respective densities of the source support points 
    nu_coords, nu_values: Arrays of shape ((M_nu,)*D)
        Coordinates and respective densities of the target support points 
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
        Optimal transport plan between source and target support points. For P[i,j], 
        the indices i, j coorespond to support point indices in mu_coords and nu_coords
    mu_coords: Array of shape (M_mu, D)
        Source support point coordinates in the original space
    nu_coords: Array of shape (M_nu, D)
        Target support point coordinates in the original space
    log: dict, optional
        Returns specifications for the optimisation

    """

    # Cost matrix for for the support points
    C = sqeuclidean_dist(mu_coords,nu_coords) 

    # Initialize scaling factors in log space
    u_log = np.zeros_like(mu_values)
    v_log = np.zeros_like(nu_values)

    # Sinkhorn-Knopp iterations in log space
    for i in range(max_iter):
        u_log_prev = u_log.copy()
        v_log_prev = v_log.copy()

        # Update u and v in log space
        u_log = epsilon * (np.log(mu_values) - logsumexp((-C + v_log[None, :]) / epsilon, axis=1))
        v_log = epsilon * (np.log(nu_values) - logsumexp((-C + u_log[:, None]) / epsilon, axis=0))

        # Check relative change stopping criterion
        delta_u = np.max(np.abs(u_log - u_log_prev) / np.maximum(np.abs(u_log_prev), 1e-16))
        delta_v = np.max(np.abs(v_log - v_log_prev) / np.maximum(np.abs(v_log_prev), 1e-16))
        if max(delta_u, delta_v) < tol:
            break

    # Construct the sparse transport plan in log space
    logP = (u_log[:, None] + v_log[None, :] - C) / epsilon

    # Convert logP to P (safe exponentiation)
    m = np.max(logP)
    P = np.exp(logP - m)
    P *= np.exp(m)  
    cost = np.sum(P * C) # Total transoport cost

    if return_log:
        log = {'cost': cost, 'num_iter': i+1, 'epsilon': epsilon, 'tol': tol, 'mu_shape': mu_coords.shape, 'nu_shape': nu_coords.shape}
        return P, mu_coords, nu_coords, log
    return P, mu_coords, nu_coords

def sinkhorn_sparse(mu_coords, mu_values, nu_coords, nu_values, epsilon=1e-1, max_iter=5000, tol = 1e-3, return_log=False):
    """
    Entropy reguralised optimal transport using the sinkhorn-knopp algorithm for 
    sparse sourse and target distributions with M_mu and M_nu support points. Differs 
    from standard sinkhorn in that it does not construct the full K

    Parameters
    ----------
    mu_coords, mu_values: Arrays of shape ((M_mu,)*D)
        Coordinates and respective densities of the source support points 
    nu_coords, nu_values: Arrays of shape ((M_nu,)*D)
        Coordinates and respective densities of the target support points 
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
        Optimal transport plan between source and target support points. For P[i,j], 
        the indices i, j coorespond to support point indices in mu_coords and nu_coords
    mu_coords: Array of shape (M_mu, D)
        Source support point coordinates in the original space
    nu_coords: Array of shape (M_nu, D)
        Target support point coordinates in the original space
    log: dict, optional
        Returns specifications for the optimisation

    """
    
    # Cost matrix for the support points
    C = sqeuclidean_dist(mu_coords, nu_coords) 
    C_normal = C / np.max(C) # Normalize the cost to make the exponential stable
    
    # Initialize scaling factors
    u = np.ones_like(mu_values)
    v = np.ones_like(nu_values)
    
    # Sinkhorn-Knopp iterations
    for i in range(max_iter):
        u_prev = u.copy()
        v_prev = v.copy()
        
        # Update u and v without forming full K
        # For each row i: u[i] = mu[i] / sum_j exp(-C[i,j]/epsilon) * v[j]
        u = mu_values / np.sum(np.exp(-C_normal / epsilon) * v[None, :], axis=1)
        v = nu_values / np.sum(np.exp(-C_normal / epsilon) * u[:, None], axis=0)
        
        # Check relative change stopping criterion
        delta_u = np.max(np.abs(u-u_prev)/np.maximum(np.abs(u_prev), 1e-16))
        delta_v = np.max(np.abs(v-v_prev)/np.maximum(np.abs(v_prev), 1e-16))
        if max(delta_u, delta_v) < tol:
            break
    
    # Construct the sparse transport plan
    P = u[:, None] * np.exp(-C_normal / epsilon) * v[None, :]
    cost = np.sum(P * C) # Total transport cost
    
    if return_log:
        log = {'cost': cost, 'num_iter': i+1, 'epsilon': epsilon, 'tol': tol, 'mu_shape': mu_coords.shape, 'nu_shape': nu_coords.shape}
        return P, mu_coords, nu_coords, log
    return P, mu_coords, nu_coords

########## Plot functions ##########

def plot_imshow(psd, name = 'spectrum'):
    """
    Visualise the power spectral density of a signal in 2D using imshow.

    Parameters
    ----------
    psd: array, shape (N, M)
        Power spectral density represented as a 2D array.
    name: str, optional
        Title of the plot. Default is 'spectrum'.

    Returns
    -------
    None
        Displays the plot.

    """

    if np.ndim(psd) != 2:
        raise NotImplementedError("Visualization only implemented for 2D grids")
    else:
        plt.figure(figsize=(6, 5))
        plt.imshow(psd, extent=[-0.5, 0.5, -0.5, 0.5], origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar(label='Normalized Power')
        plt.title(f'Power Spectral Density ({name})')
        plt.xlabel('Frequency axis 1')
        plt.ylabel('Frequency axis 2')
        plt.tight_layout()
        plt.show()

def plot_sparse_transport_marginals(P, mu_coords, nu_coords, grid_shape):
    """
    Visualise the marginal distributions in full space of a sparce OT plan. The source marginal has M_mu source support points and the trget marginal hhas M_nu support points.
    Only transports in 2D are supported. For both marginals the same colour scale is used.

    Parameters
    ----------
    P: array, shape (M_mu, M_nu)
        Optimal transport plan between sparse source and target distributions. 
        P[i,j] represents the transported mass from source support point i to target support point j.
    mu_coords: array, shape (M_mu, D)
        Source support point coordinates in the original space.
    nu_coords: array, shape (M_nu, D)
        Target support point coordinates in the original space.
    grid_shape: tuple of int, length D
        Shape of the full grid in which marginals are embedded (e.g., (N, N) or (N, N, N)).

    Returns
    -------
    None
        Displays the plot.
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

        im0 = axes[0].imshow(source_mass_grid, cmap='viridis', extent=[-0.5, 0.5, -0.5, 0.5], origin='lower', vmin=vmin, vmax=vmax)
        axes[0].set_title('Source Mass Distribution')
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(target_mass_grid, cmap='viridis',extent=[-0.5, 0.5, -0.5, 0.5], origin='lower', vmin=vmin, vmax=vmax)
        axes[1].set_title('Target Mass Distribution')
        plt.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        plt.show()

    else:
        raise NotImplementedError("Visualization only implemented for 2D grids")



def plot_sparse_interpolation_polar(sparse_sequence, space_shape, scale_mass=100):
    """
    Visualise the interpolation sequence of a sparse OT interpolation in polar coordinates. The angular coordinate 
    corresponds to the frequency of each support point (computed from centered coordinates in [-1/2, 1/2]). The radius corresponds 
    to the transported mass at that frequency location.

    Parameters
    ----------
    sparse_sequence: list of tuples
        Interpolation sequence [(coords, values), ...]. Each tuple contains the support point 
        coordinates and their corresponding mass values at each interpolation step.
    space_shape: tuple of int, length D
        Shape of the transport space. Shape (N, N) for 2D or (N, N, N) for 3D.
    scale_mass: int, optional
        Scaling factor for the scatter point sizes. Default is 100.

    Returns
    -------
    None
        Displays the plot.
    """

    D = len(space_shape) # Determine the space dimension
    N_theta = len(sparse_sequence) # Determine the number of interpolation steps
    space_shape = np.array(space_shape) # Convert to numpy array 
    freq_center = space_shape // 2 # Define the center of the frequency space
    colors = plt.cm.viridis(np.linspace(0, 1, N_theta)) # Generate a colourmap for the steps

    fig = plt.figure(figsize=(16, 7)) # Set figure size
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1])  # Set subplots (1 row, 2 columns) with specified ratio of 4:1
    ax_legend = fig.add_subplot(gs[0, 1]) # Let the legend occupy right subplot
    ax_legend.axis('off')  # Blank subplot for legend

    # Initialise the plot and let it occupy the left subplot. Projection according to dimension D
    if D == 2:
        ax_plot = fig.add_subplot(gs[0, 0], projection='polar') # Let the plot occupy left subplot
    elif D == 3:
        ax_plot = fig.add_subplot(gs[0, 0], projection='3d') 
    else:
        raise ValueError("D must be 2 or 3.")

    # Initialise lists for legend handles and labels
    handles = []
    labels = []


    # Loop through each interpolation step
    for t, (coords, values) in enumerate(sparse_sequence):
        freqs = (coords - freq_center) / space_shape # Convert coordinates to centered frequencies in [-1/2, 1/2]
        
        if D == 2:
            angles = np.arctan2(freqs[:, 1], freqs[:, 0]) # Calculate angles in polar coordinates
            r_plot = values # Let radial coodinate be the mass at that frequency location
            sc = ax_plot.scatter(angles, r_plot, color = colors[t], alpha = 0.9, s = scale_mass) # Plot the points for the current interpolation step
        else:  # D==3
            norms = np.linalg.norm(freqs, axis=1, keepdims=True) # Calcuulate norms of the frequency vectors
            dirs = freqs / norms # Normalise frequnecy vectors to unit length
            r_plot = values # Let radial coodinate be the mass at that frequency location
            x = dirs[:,0] * r_plot # Point x-coordinate 
            y = dirs[:,1] * r_plot # Point y-coordinate
            z = dirs[:,2] * r_plot # Point z-coordinate
            sc = ax_plot.scatter(x, y, z, color=colors[t], alpha=0.9, s = scale_mass) # Plot the points for the current interpolation step

        handles.append(sc) 
        labels.append(f"t={t}")

    # Legend on the right
    ax_legend.legend(handles, labels, loc='center left', fontsize=8) 

    # Axis settings
    if D == 2:
        ax_plot.set_title("Sparse OT Interpolation (D=2)")
    else: # D==3
        ax_plot.set_title("Sparse OT Interpolation (D=3)")

        # Unit sphere reference
        u, v = np.mgrid[0:2*np.pi:25j, 0:np.pi:15j]
        xs = np.sin(v)*np.cos(u)
        ys = np.sin(v)*np.sin(u)
        zs = np.cos(v)
        ax_plot.plot_wireframe(xs, ys, zs, color='gray', alpha = 0.2, linewidth = 0.7) 

  
    plt.tight_layout()
    plt.show()


def plot_sparse_interpolation_freqs(sparse_sequence, space_shape, scale_mass=100):
    """
    Visualise the interpolation sequence of a sparse OT interpolation in the cartesian frequency space. 
    The size of the scatter point corresonds to the transported mass at that frequency location.

    Parameters
    ----------
    sparse_sequence: list of tuples
        Interpolation sequence [(coords, values), ...]. Each tuple contains the support point 
        coordinates and their corresponding mass values at each interpolation step.
    space_shape: tuple of int, length D
        Shape of the transport space. Shape (N, N) for 2D or (N, N, N) for 3D.
    scale_mass: int, optional
        Scaling factor for the scatter point sizes. Default is 100.

    Returns
    -------
    None
        Displays the plot.
    """

    D = len(space_shape) # Determine the space dimension
    N_theta = len(sparse_sequence) # Determine the number of interpolation steps
    space_shape = np.array(space_shape) # Convert to numpy array 
    freq_center = space_shape // 2 # Define the center of the frequency space
    colors = plt.cm.viridis(np.linspace(0, 1, N_theta)) # Generate a colourmap for the steps
    
    fig = plt.figure(figsize=(16, 7)) # Set figure size
    gs = fig.add_gridspec(1, 2, width_ratios=[4, 1])  # Set subplots (1 row, 2 columns) with specified ratio of 4:1
    ax_legend = fig.add_subplot(gs[0, 1]) # Let the legend occupy right subplot
    ax_legend.axis('off')  # Blank subplot for legend

    # Initialise the plot and let it occupy the left subplot. Projection according to dimension D
    if D == 2:
        ax_plot = fig.add_subplot(gs[0, 0])
    elif D == 3:
        ax_plot = fig.add_subplot(gs[0, 0], projection='3d')
    else:
        raise ValueError("D must be 2 or 3.")

    # Initialise lists for legend handles and labels
    handles = []
    labels = []

    # Loop through each interpolation step
    for t, (coords_sparse, values_sparse) in enumerate(sparse_sequence):
        freqs = (coords_sparse - freq_center) / space_shape # Convert coordinates to centered frequencies in [-1/2, 1/2]
        sizes = scale_mass * (values_sparse / np.max(values_sparse)) # Scale the sizes of the scatter points based on the normailsed mass values
        
        if D == 2:
            sc = ax_plot.scatter(freqs[:, 0], freqs[:, 1],
                                 s=sizes, color=colors[t], alpha=0.8) # Plot the points for the current interpolation step
            coords_str = ", ".join([f"({fx:.2f},{fy:.2f})" for fx, fy in freqs]) # Add coordinates to the legend string
        else:  # D==3
            sc = ax_plot.scatter(freqs[:, 0], freqs[:, 1], freqs[:, 2],
                                 s=sizes, color=colors[t], alpha=0.6) # Plot the points for the current interpolation step
            coords_str = ", ".join([f"({fx:.2f},{fy:.2f},{fz:.2f})" for fx, fy, fz in freqs]) # Add coordinates to the legend string

        handles.append(sc)
        labels.append(f"step {t}: {coords_str}")

    # Legend on the right
    ax_legend.legend(handles, labels, loc='center left', fontsize=8)

    # Axis settings
    if D == 2:
        ax_plot.set_xlabel("f1 (cycles/unit)")
        ax_plot.set_ylabel("f2 (cycles/unit)")
        ax_plot.set_aspect('equal')
        ax_plot.set_xlim(-0.55, 0.55)
        ax_plot.set_ylim(-0.55, 0.55)
        ax_plot.grid(True)
    else:  # D==3
        ax_plot.set_xlabel("f1 (cycles/unit)")
        ax_plot.set_ylabel("f2 (cycles/unit)")
        ax_plot.set_zlabel("f3 (cycles/unit)")
        ax_plot.set_xlim(-0.55, 0.55)
        ax_plot.set_ylim(-0.55, 0.55)
        ax_plot.set_zlim(-0.55, 0.55)

    plt.tight_layout()
    plt.show()

########## Helper functions ##########

def nonzero_indices_and_values(A, threshold = 1e-6):
    """
    Retrieve the indices and values of all non-zero entries of a sparse array A. 
    The values are normalized to sum to 1.

    Parameters
    ----------
    A: array, shape (N1, ..., ND)
        Sparse numpy array in D dimensions. The number of non-zero entries is M.
    threshold: float, optional
        All values <= threshold are treated as zero. Default is 1e-6.

    Returns
    -------
    values: array, shape (M,)
        Values of all non-zero entries of A.
    indices: array, shape (M, D)
        Indices of all non-zero entries of A.
    """

    indices = np.argwhere(A > threshold) # Get indices of non-zero entries

    # If no non-zero entries, return empty arrays
    if indices.size == 0:
        return np.empty((0, A.ndim), dtype=int), np.array([], dtype=float)
    
    values = A[tuple(indices.T)] # get associated values of non-zero entries
    values /= np.sum(values)  # Normalize values to sum to 1

    return indices, values

def sqeuclidean_dist(A,B):
    """
    Calculates the squared euclidean distance bewteen two sets of points A and B. 

    Parameters
    ----------
    A: array, shape (N, D)
        First set of coordinate points.
    B: array, shape (M, D)
        Second set of coordinate points.

    Returns
    -------
    sq_dist: array, shape (N, M)
        Squared Euclidean distances between each point in A and each point in B.
    """

    # Assert that A and B have the same dimensions
    if np.shape(A)[1] != np.shape(B)[1]:
        raise ValueError('A and B musta have same dimension.')
    
    # Calculate squared Euclidean distances with vector operations (bmo broadcasting)
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]  
    sq_dist = np.sum(diff**2, axis=-1)             

    return sq_dist



if __name__ == "__main__":
    main()