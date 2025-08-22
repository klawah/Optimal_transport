
import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))  # adds 'src' parent
import src.utils.signal_simulation as signal
import sinkhorn as ot
import src.utils.OT_interpolation as itp
import time

def main():
    sizes = [10,50,100]
    N = sizes[-1]
    D = 3 # Spacial dimension
    N_theta = 21
    theta_seq = np.linspace(0,1,N_theta)
    
    # Signal parameters 
    if D == 2:
        mu_parameters = [
            {'amplitude': 0.7, 'fc': [0.3,-0.3], 'damping': 0}, 
            {'amplitude': 0.5, 'fc': [0.2,-0.1], 'damping': 0.0},
            {'amplitude': 1.1, 'fc': [-0.2,0.3], 'damping': 0}
        ]
        nu_parameters = [
            {'amplitude': 0.7, 'fc': [0.1,-0.3], 'damping': 0}, 
            {'amplitude': 1, 'fc': [-0.2,0.05], 'damping': 0.0}
        ]
    if D == 3:
        mu_parameters = [
        {'amplitude': 1, 'fc': [-0.3,0.05,-0.1], 'damping': 0},
        {'amplitude': 1, 'fc': [0.3,0.25,0.45], 'damping': 0}
        ]

        nu_parameters = [
        {'amplitude': 1, 'fc': [0.05, 0.1,0.05], 'damping': 0},
        {'amplitude': 1, 'fc': [-0.3,-0.20,-0.1], 'damping': 0} 
        ]

    mu_original = signal.generate_multi_dimensional_sinusoid(D, N, mu_parameters) # shape((N,)*D)
    nu_original = signal.generate_multi_dimensional_sinusoid(D, N, nu_parameters) # shape((N,)*D)
    space_shape = np.shape(mu_original)
    print(f"Shape of the original signal: {space_shape}")


    # ot.plot_imshow(mu_original)

    multiscale_ot = multiscale_sinkhorn(D,mu_original,nu_original,sizes)

    ot_original = multiscale_ot[-1]
    
    print(f"Original OT cost: {total_cost(ot_original[0], ot_original[1], ot_original[2])}")
    
    itp_seq = itp.interpolate_sparse_OT_from_sparse(ot_original[0], ot_original[1], ot_original[2], theta_seq, space_shape)
    
    ot.plot_sparse_interpolation_polar(itp_seq,space_shape)
    ot.plot_sparse_interpolation_freqs(itp_seq,space_shape)
    if D == 2:
        for i, P in enumerate(multiscale_ot):
            N = sizes[i]
            P_fine_global, mu_coords_fine_global, nu_coords_fine_global = P
            plot_transport_marginals(P_fine_global,mu_coords_fine_global,nu_coords_fine_global,N)

def multiscale_sinkhorn(D,mu_original,nu_original,sizes):
    """
    Perform multiscale otimal transport on the given source and target signals specified in D dimesnions. The source and target 
    signals are downsampled to the given sizes, and the optimal transport is computed on each level. The results are returned 
    as a list of transport plans for each level.

    Parameters
    ----------
    D: int
        Space dimension for the signals.
    mu_original: array, shape ((N,)*D)
        Original source distribution as a D-dimensional spectral density.
    nu_original : array, shape ((N,)*D)
        Original target distribution as a D-dimensional spectral density.
    sizes: list of int
        List of sizes for each level of the multiscale OT. The first element is the size of the core level, and the 
        last element is the size of the original axes. 

    Returns
    -------
    P_elevels: list [P, source_coords, target_coords]
        List of transport plans for each level, where P = [indices, values] is the transport plan. 
        The indices are associated with the coordinates of the source and target distributions, and values are the transported masses.
    """

    start = time.time() # start time for entire optimisation
    print(f"Starting multiscale OT with sizes: {sizes} and D: {D}")

    # Check if the scaling factors are valid and return the corresponding scaling factors and cumulative scalings for each step
    is_valid, scaling_factors, cumulative_scalings = is_valid_scaling(sizes) 
    if not is_valid:
         raise ValueError("Invalid input. All sequential scaling divisions must be exact divisions.")
    
    num_levels=len(scaling_factors) # Number of scaling jumps in the multiscale OT
    N_core= sizes[0] # Size of the axes in the core level optimisation (most coarse)

    # Downsmple the original signals to the core level. 
    mu_core = downsample_signal(mu_original, N_core, cumulative_scalings[0], D)
    nu_core = downsample_signal(nu_original, N_core, cumulative_scalings[0], D)

    #ot.plot_imshow(mu_core)

    # Perform the core level OT
    P_indices_core, P_values_core, source_mass_coords, target_mass_coords = core_level_OT(mu_core,nu_core,N_core,D)
    

    # plot_transport_marginals([P_coords_core,P_values_core],source_mass_coords,target_mass_coords,N_core)

    P_levels = [[[P_indices_core, P_values_core],source_mass_coords,target_mass_coords]]
    for i in range(num_levels):
        
        scaling_factor = scaling_factors[i]
        cumulative_scaling = cumulative_scalings[i+1]
        P_coarse = P_levels[i][0]
        source_coords_coarse = P_levels[i][1]
        target_coords_coarse = P_levels[i][2]
        N_fine = sizes[i+1]

        threshold_fine = 1e-3 / (N_fine**D)

        P_fine_global,mu_coords_fine_global,nu_coords_fine_global = inter_level_OT(P_coarse,mu_original,nu_original,source_coords_coarse,target_coords_coarse,scaling_factor,cumulative_scaling,threshold_fine)
        P_levels.append([P_fine_global,mu_coords_fine_global,nu_coords_fine_global])
        print(time.time() - start)
    
    end = time.time() # end time for entire optimisation
    elapsed = end - start # total time for optimisation
    print(f"Total time for multiscale OT: {elapsed:.2f} seconds")
        
    return P_levels

def core_level_OT(mu_core,nu_core,N_core,D):
    """
    Perform the core level optimal transport between the downsampled source and target distributions.

    Parameters
    ----------
    mu_core: array, shape ((N_core,)*D)
        Downsampled source distribution in D dimensions.
    nu_core: array, shape ((N_core,)*D)
        Downsampled target distribution in D dimensions.
    N_core: int
        Size of the axes in the core level optimisation (most coarse).
    D: int
        Space dimension for the signals.

    Returns
    -------
    P_indices_core: array, shape (M, 2)
        List of source, target index pairs of the optimal transport plan in the core level, 
        where M is the number of non-zero entries in the transport plan. The indexes map to source 
        and target coordinates through source_mass_coords and target_mass_coords.
    P_values_core: array, shape (M,)
        Values of the optimal transport plan in the core level, corresponding to the transported mass beween each source and target index pair.
    source_mass_coords: array, shape (M_mu, D)
        Coordinates of the source mass points in the core level, where M_mu is the number of support points.
    target_mass_coords: array, shape (M_nu, D)
        Coordinates of the target mass points in the core level, where M_nu is the number of support points.
    """

    # Only keep coordinates with above zero-mass (i.e. all coordinates with values below a certain threshold)
    threshold_core = 1e-3 / (N_core**D)  
    mu_core_coords,mu_core_values = nonzero_indices_and_values(mu_core,threshold_core)
    nu_core_coords,nu_core_values = nonzero_indices_and_values(nu_core,threshold_core)

    # Perform the core level OT
    P_core, source_mass_coords, target_mass_coords = ot.sinkhorn_sparse(mu_core_coords,mu_core_values,nu_core_coords,nu_core_values) 

    # Reformat the transport plan P_core to get the indices and values
    P_indices_core, P_values_core = nonzero_indices_and_values(np.array(P_core), threshold=0)
    
    return P_indices_core, P_values_core, source_mass_coords, target_mass_coords

def inter_level_OT(P_coarse,mu_original,nu_original,mu_coords_coarse,nu_coords_coarse,scaling_factor,cumulative_scaling,threshold_fine):
    P_coords_coarse = P_coarse[0]
    P_values_coarse = P_coarse[1]

    P_fine = []
    source_mass_coords_fine = []
    target_mass_coords_fine = []
    
    for i, source_coord_coarse in enumerate(mu_coords_coarse):
        source_indices = np.argwhere(P_coords_coarse[:,0] == i) # sparce OT indices of all transports from source indices
        source_mass = np.sum(P_values_coarse[source_indices])
        mu_coords_fine, mu_values_fine = nonzero_fine_coords_and_values(source_coord_coarse,scaling_factor,cumulative_scaling,mu_original,threshold_fine)

        for j, target_coord_coarse in enumerate(nu_coords_coarse):
            target_indices = np.argwhere(P_coords_coarse[:,1] == j)
            target_mass = np.sum(P_values_coarse[target_indices])

            source_target_index = np.argwhere((P_coords_coarse[:,0] == i) & (P_coords_coarse[:,1] == j))
            source_target_mass = np.sum(P_values_coarse[source_target_index])

            source_mass_ratio = source_target_mass / source_mass
            target_mass_ratio = source_target_mass / target_mass

            nu_coords_fine, nu_values_fine = nonzero_fine_coords_and_values(target_coord_coarse,scaling_factor,cumulative_scaling,nu_original,threshold_fine)

            
            if mu_coords_fine.size == 0 or nu_coords_fine.size == 0:
                continue
            if source_mass == 0 or target_mass == 0 or source_target_mass == 0:
                continue

            P, mu_coords, nu_coords,log = ot.sinkhorn_sparse(mu_coords_fine,mu_values_fine*source_mass_ratio,nu_coords_fine,nu_values_fine*target_mass_ratio,return_log=True)
            
            P_fine.append(P)
            source_mass_coords_fine.append(mu_coords)
            target_mass_coords_fine.append(nu_coords)

    P_fine_global, mu_coords_fine_global, nu_coords_fine_global = accumulate_fine_transport(P_fine, source_mass_coords_fine, target_mass_coords_fine)
    
    return P_fine_global,mu_coords_fine_global,nu_coords_fine_global

def plot_transport_marginals(P, mu_coords, nu_coords, N):
    """
    Plot source and target marginals from fine-scale transport plan P.

    Parameters
    ----------
    P : list [indices, values]
        Transport plan with indices (i,j) into mu_coords/nu_coords and values = transported mass.
    mu_coords : ndarray, shape (M, d)
        Fine-level source coordinates.
    nu_coords : ndarray, shape (N, d)
        Fine-level target coordinates.
    N : int or tuple
        Size of the fine grid, e.g. 10 or (10,10) for 2D.
    """
    indices, values = P

    # Ensure N is a tuple (for d-dimensional grid)
    if np.isscalar(N):
        d = mu_coords.shape[1]
        N = (N,) * d

    # Compute marginals
    mu_marginal = np.zeros(len(mu_coords))
    nu_marginal = np.zeros(len(nu_coords))
    for (i, j), v in zip(indices, values):
        mu_marginal[i] += v
        nu_marginal[j] += v

    # Put marginals into images
    mu_image = np.zeros(N)
    nu_image = np.zeros(N)
    for coord, val in zip(mu_coords, mu_marginal):
        mu_image[tuple(coord)] = val
    for coord, val in zip(nu_coords, nu_marginal):
        nu_image[tuple(coord)] = val

    # Plot only if 2D
    if len(N) == 2:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(mu_image, origin="lower", cmap="viridis")
        ax[0].set_title("Source marginal")
        ax[1].imshow(nu_image, origin="lower", cmap="viridis")
        ax[1].set_title("Target marginal")
        plt.show()
    else:
        print(f"Grid dimension {len(N)} not supported for imshow.")
        return mu_image, nu_image



def total_cost(P, mu_coords, nu_coords, freq_cost = False):
    """
    Compute total transport cost in frequency space (centered at zero, scaled -1/2 to 1/2).

    Parameters
    ----------
    P_fine : [indices, values]
        - indices: (M,2) array of source/target indices
        - values: (M,) array of transported masses
    mu_coords_fine : (M_mu, D) array
        Source coordinates
    nu_coords_fine : (M_nu, D) array
        Target coordinates

    Returns
    -------
    total_cost : float
        Weighted squared distance in frequency space
    """
    if freq_cost:
        P_indices, P_values = P

        # Determine the grid shape for normalization
        all_coords = np.vstack([mu_coords, nu_coords])
        space_shape = np.max(all_coords, axis=0) + 1
        freq_center = space_shape / 2

        # Convert coordinates to centered frequency space
        mu_freqs = (mu_coords - freq_center) / space_shape
        nu_freqs = (nu_coords - freq_center) / space_shape

        # Extract transported points
        mu_t = mu_freqs[P_indices[:, 0]]
        nu_t = nu_freqs[P_indices[:, 1]]

        # Compute weighted squared distances
        sq_dists = np.sum((mu_t - nu_t)**2, axis=1)
        total_cost = np.sum(P_values * sq_dists)
        
        return total_cost
    else:
        src_coords, tgt_coords, values = reformat_OT_plan(P, mu_coords, nu_coords)
        sq_dists = np.sum((src_coords - tgt_coords)**2, axis=1)
        
        return np.sum(values * sq_dists)    


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


def nonzero_fine_coords_and_values(coarse_coord, scaling_factor, cumulative_scaling, psd_original, threshold):
    
    coarse_coord = np.array(coarse_coord)
    if np.isscalar(scaling_factor):
        scaling_factor = np.full_like(coarse_coord, scaling_factor)
    else:
        scaling_factor = np.array(scaling_factor)
    
    ranges = [np.arange(c*sf, c*sf + sf) for c, sf in zip(coarse_coord, scaling_factor)]
    
    grids = np.meshgrid(*ranges, indexing='ij')
    coords = np.stack([g.ravel() for g in grids], axis=-1)
    
    coords_list = []
    values_list = []
    for coord in coords:
        val = coordinate_mass(psd_original, coord, cumulative_scaling)
        if val > threshold*scaling_factor[0]:
            coords_list.append(coord)
            values_list.append(val)
    
    if coords_list:
        return np.array(coords_list), np.array(values_list)
    else:
        return np.empty((0, len(coarse_coord))), np.empty((0,))

def coordinate_mass(psd_original, coarse_coord, scaling_factor):
    """
    Sum up fine-grid values corresponding to a given coarse-grid coordinate.
    
    Parameters
    ----------
    arr : ndarray of shape (N1,...,Nk)
        Fine-grid array
    coarse_coord : iterable of length k
        Coarse-grid coordinate
    scaling_factor : int or iterable of length k
        Relation between coarse and fine grids
    
    Returns
    -------
    mass : scalar
        Sum of fine-grid values in the corresponding block
    """
    coarse_coord = np.array(coarse_coord)
    if np.isscalar(scaling_factor):
        scaling_factor = np.full_like(coarse_coord, scaling_factor)
    else:
        scaling_factor = np.array(scaling_factor)
    
    start = coarse_coord * scaling_factor
    slices = tuple(slice(s, s+f) for s, f in zip(start, scaling_factor))
    
    return psd_original[slices].sum()
            

def accumulate_fine_transport(P_blocks, mu_coords_blocks, nu_coords_blocks):
    """
    Merge fine-level transports from multiple blocks.
    
    Parameters
    ----------
    P_blocks : list of P from each block, where P = [indices, values]
    mu_coords_blocks : list of mu_coords for each block
    nu_coords_blocks : list of nu_coords for each block
    
    Returns
    -------
    P_fine : [indices, values] for global fine-level transport
    mu_coords_fine : array of unique fine source coordinates
    nu_coords_fine : array of unique fine target coordinates
    """
    
    # Flatten all coordinates
    all_mu_coords = np.vstack(mu_coords_blocks)
    all_nu_coords = np.vstack(nu_coords_blocks)

    # Unique coordinates
    mu_unique, mu_idx_map = np.unique(all_mu_coords, axis=0, return_inverse=True)
    nu_unique, nu_idx_map = np.unique(all_nu_coords, axis=0, return_inverse=True)

    # Build global indices for P_blocks
    P_indices = []
    P_values = []

    mu_offset = 0
    nu_offset = 0
    for block_idx, P in enumerate(P_blocks):
        indices, values = nonzero_indices_and_values(P)
        for (i_local, j_local), val in zip(indices, values):
            global_i = mu_idx_map[mu_offset + i_local]
            global_j = nu_idx_map[nu_offset + j_local]
            P_indices.append([global_i, global_j])
            P_values.append(val)
        mu_offset += len(mu_coords_blocks[block_idx])
        nu_offset += len(nu_coords_blocks[block_idx])

    return [np.array(P_indices), np.array(P_values)], mu_unique, nu_unique

def reformat_OT_plan(P, mu_coord_index_ref, nu_coord_index_ref): 
    """ 
    Make the connection between OT plan P[i,j] and the coordinates at indices i, j
    of the source and target support points. Transportation of mass P_values[n] happens 
    from P_source_coords[n] to P_target_coords[n]. 
    
    Parameters
    ---------- 
    P: array, shape (M_mu, M_nu) 
        Optimal transport plan between sparse source and target distributions. 
        P[i,j] represents the transported mass from source support point i to target support point j. 
    mu_coords: array, shape (M_mu, D)
        Source support point coordinates in the original space. 
    nu_coords: array, shape (M_nu, D) 
        Target support point coordinates in the original space. 
        
    Returns 
    ------- 
    P_source_coords : array, shape (M, D) 
        Coordinates corresponding to source points for mass transportation 
    P_target_coords : array, shape (M, D) 
        Coordinates corresponding to target points for mass transportation 
    values : array, shape (M,) 
        Transported masses aligned with coords. M is the number of different transport paths in P. 
    """ 
    
    P_indices, P_values = P 
    
    P_source_coords = mu_coord_index_ref[P_indices[:, 0]] 
    P_target_coords = nu_coord_index_ref[P_indices[:, 1]] 
    
    return P_source_coords, P_target_coords, P_values


def coord_list(N,D):
    axes = [np.linspace(0,N-1,N) for _ in range(D)] # Define each axis of the grid
    grid = np.meshgrid(*axes, indexing = 'ij') # Create a meshgrid object with the given axis length
    coords = np.stack([g.ravel() for g in grid], axis=-1) # List of coordinates in the grid (row major order)
    return coords.astype(int)
    
def coordinate_mapping(coords_coarse, coords_fine, scaling_factor):
    index_map = [] # Initialise a map of indexes that will correspond to the block of coordinate points in the fine grid associated with a coordinate point in the coarse grid

    for i, coarse_coord in enumerate(coords_coarse): # iterate through the coordinate points (index i and coordinate, coarse_coord) in the coarse grid
        start = coarse_coord * scaling_factor # Starting coordinate point for that coarse point in the fine grid (upper left corner)
        end = start + scaling_factor # End coordinate point for the block (lower right corner)

        block = np.all((coords_fine >= start) & (coords_fine < end), axis=1) # Find all fine grid point coordinates within the current block (iterate all elements along axis 1 (indices) and compare the elements with start/end). Returns a boolean array of shape (N**D,)
        block_indices = np.where(block)[0]  # Returns an array containing the indices where block is True

        index_map.append([i, block_indices.tolist()]) # Map where i is the index in coarse_grid and block_indices are the associated indices in fine_grid

    return index_map

def coarse_to_fine(coarse_coords,scaling_factor):
    fine_coords = []
    for coarse_coord in coarse_coords:
        for i in coarse_coord:
            fine_coord = []
            for j in scaling_factor:
                fine_coord.append(i**scaling_factor+j) 
            fine_coords.append(fine_coord)

    return fine_coords
     

def is_valid_scaling(sizes, coarse_to_fine=True):

    if coarse_to_fine:
        sizes = sizes[::-1]  # flip so we can reuse same logic

    scaling_factors = []
    cumulative_scaling = [1]

    current_size = sizes[0]
    for size in sizes[1:]:
        if current_size % size != 0:
            return False, [], []
        scaling_factor = current_size // size
        scaling_factors.append(scaling_factor)
        cumulative_scaling.append(scaling_factor * cumulative_scaling[-1])
        current_size = size


    if coarse_to_fine:
        # Flip back the results so they align with the original input order
        scaling_factors = scaling_factors[::-1]
        cumulative_scaling = cumulative_scaling[::-1]

    return True, scaling_factors, cumulative_scaling

def generate_signal_levels(mu_original,nu_original,scaling_factors,sizes,D):
    mu_signal_levels = [mu_original]
    nu_signal_levels = [nu_original]

    inverse_sizes = sizes[::-1]
    inverse_scaling_factors = scaling_factors[::-1]
    
    for scaling_level in range(len(inverse_scaling_factors)):
        scaling_factor = inverse_scaling_factors[scaling_level]
        N_coarse = inverse_sizes[scaling_level+1]

        mu_fine = mu_signal_levels[scaling_level]
        nu_fine = nu_signal_levels[scaling_level]

        mu_coarse = downsample_signal(mu_fine, N_coarse, scaling_factor, D)
        nu_coarse = downsample_signal(nu_fine, N_coarse, scaling_factor, D)

        mu_signal_levels.append(mu_coarse)
        nu_signal_levels.append(nu_coarse)

    return mu_signal_levels[::-1], nu_signal_levels[::-1]

def downsample_signal(psd, N_coarse, scaling_factor,D):
    psd_coarse = np.zeros((N_coarse,) * D) # Initialise the coarse distribution grid
    for index in np.ndindex((N_coarse,) * D): # Iterate through all coordinates in the D-dimensional grid
        block = tuple(
            slice(i * scaling_factor, (i + 1) * scaling_factor) # Get a D-dimensional slice ("coarse block") of size scaling_factor in each dimension, with its upper left "corner" at the starting location of that coarse grid. Contains relevant fine grid indices (row major)
            for i in index
        )
        psd_coarse[index] = np.sum(psd[block]) # Sum up all the mass in the relevant block as the mass fo that coarse distribution grid point
    return psd_coarse

def compute_adaptive_threshold(P, N, D, top_percentile=80, min_ratio=0.05):
    
    min_threshold = 1e-3 / (N**D)     

    # Flatten and filter non-zero entries
    P_flat = P.flatten()
    P_nonzero = P_flat[P_flat > 0]

    if len(P_nonzero) == 0:
        return 0.0  # Nothing to threshold

    # Sort and extract top X% values
    sorted_P = np.sort(P_nonzero)[::-1]  # Descending
    cutoff_index = int(len(sorted_P) * (top_percentile / 100))
    top_values = sorted_P[:cutoff_index]

    # Compute mean of top values
    mean_top = np.mean(top_values)

    # Set threshold as a ratio of that mean
    threshold = min_ratio * mean_top


    return max(min_threshold,threshold)

if __name__ == "__main__":
    main()