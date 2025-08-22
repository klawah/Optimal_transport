import numpy as np
from scipy.ndimage import gaussian_filter



def interpolate_multi_dimensional_OT(P, space_dim, theta_seq, smoothing_sigma=10,max_sigma_fraction=0.05):
    
    N_flat = P.shape[0]                                                         # Total number of points in the space: N_flattened = N^D
    N = round(P.shape[0] ** (1 / space_dim))                                      # Space axis dimension (square space)
    N_theta = len(theta_seq)
  
    axis = [np.arange(N) for _ in range(space_dim)]
    grid = np.meshgrid(*axis, indexing = 'ij')
    coords = np.stack([g.ravel() for g in grid], axis=-1)                       # Shape (N^D,N). Each row represents one grid point's coordinates in {R}^D [x1,x2,...xD]

    theta_seq = np.asarray(theta_seq).flatten()
    N_theta = len(theta_seq)

    T_flat = P.flatten()                                             # Shape (N^D)^2. T_flat[i * N^D + j] holds the amount of mass transported from coords[i] to coords[j]

    psd_seq = np.zeros((N_flat, N_theta))

    for i, theta in enumerate(theta_seq):

        # Interpolated positions
        source = coords[:, None, :]                                             # shape: (N^D, 1, D)
        target = coords[None, :, :]                                             # shape: (1, N^D, D)
        point_cont = (1 - theta) * source + theta * target                      # shape: (N^D, N^D, D). point_cont[i,j] = linear interpoaltion between i,j

        # Round to nearest grid point (continous --> discrete)
        point_disc = np.clip(np.round(point_cont).astype(int), 0, N-1)          # Rounds to nearest integer. Is bounded by 0 below and N-1 above

        # Coordinates --> flat indexes
        flat_idx = np.ravel_multi_index(point_disc.reshape(-1, space_dim).T, dims = (N,) * space_dim)

        psd = np.bincount(flat_idx, weights=   T_flat, minlength=N_flat)        # Shape(N^D,) of the inetrpolated distribution at theta
        psd_seq[:, i] = psd

    psd_seq_smooth = None
    if N > 10 and smoothing_sigma > 0:
        sigma_used = min(smoothing_sigma, max_sigma_fraction * N)
        sigma_tuple = (0,) * (space_dim - 1) +(sigma_used,) 
        psd_seq_smooth = np.zeros_like(psd_seq)
        for i in range(N_theta):
            psd_seq_smooth[:, i] = gaussian_filter(psd_seq[:, i].reshape((N,) * space_dim), sigma=sigma_tuple, mode='constant').ravel()

        return psd_seq_smooth, sigma_used
    else:
        sigma = 0

    return psd_seq, sigma


def interpolate_sparse_OT(P,mu_coords,nu_coords,theta_seq,space_shape):

    D = mu_coords.shape[1]
    shape = np.array(space_shape)
    N_flat = np.prod(space_shape)
    N_theta = len(theta_seq)
    
    itp_seq = np.zeros((N_flat,N_theta))

    src_idx, tgt_idx = np.nonzero(P)
    mass = P[src_idx, tgt_idx]

    sparse_sequence = []

    for theta in theta_seq:
        
        itp_coords = (1-theta) * mu_coords[src_idx] + theta * nu_coords[tgt_idx]
        itp_coords = np.clip(np.round(itp_coords).astype(int), 0, np.array(space_shape)-1)

        flat_idx, inv = np.unique(np.ravel_multi_index(itp_coords.T, dims=space_shape), return_inverse=True)
        values_sparse = np.bincount(inv, weights=mass)

        coords_sparse = np.array(np.unravel_index(flat_idx, shape=space_shape)).T

        sparse_sequence.append([coords_sparse, values_sparse])

    return sparse_sequence

def interpolate_sparse_OT_from_sparse(P_sparse, mu_coords, nu_coords, theta_seq, space_shape):
    """
    Interpolate along sparse OT transport plan without densifying P.

    Parameters
    ----------
    P_sparse : [indices, values]
        - indices: (M, 2) array of [i, j] pairs for nonzero transport
        - values : (M,) array of transported masses
    mu_coords : (M_mu, D) array
        Source support coordinates
    nu_coords : (M_nu, D) array
        Target support coordinates
    theta_seq : list or array
        Sequence of interpolation parameters in [0, 1]
    space_shape : tuple
        Shape of the grid in each dimension (e.g. (N, N) or (N, N, N))

    Returns
    -------
    sparse_sequence : list
        Each element is [coords_sparse, values_sparse], where
        - coords_sparse: (K, D) array of support points at that interpolation step
        - values_sparse: (K,) array of masses at those points
    """

    indices, values = P_sparse
    src_idx, tgt_idx = indices[:,0], indices[:,1]

    D = mu_coords.shape[1]
    shape = np.array(space_shape)
    N_theta = len(theta_seq)

    sparse_sequence = []

    for theta in theta_seq:
        # Barycentric interpolation of coordinates
        itp_coords = (1 - theta) * mu_coords[src_idx] + theta * nu_coords[tgt_idx]
        itp_coords = np.clip(np.round(itp_coords).astype(int), 0, shape - 1)

        # Collapse duplicate coordinates
        flat_idx, inv = np.unique(
            np.ravel_multi_index(itp_coords.T, dims=space_shape),
            return_inverse=True
        )
        values_sparse = np.bincount(inv, weights=values)

        coords_sparse = np.array(np.unravel_index(flat_idx, shape=space_shape)).T

        sparse_sequence.append([coords_sparse, values_sparse])

    return sparse_sequence