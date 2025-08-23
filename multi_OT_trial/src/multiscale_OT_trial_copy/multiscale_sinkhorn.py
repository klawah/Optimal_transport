import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))  

import numpy as np
import matplotlib.pyplot as plt
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
        {'amplitude': 0.7, 'fc': [-0.3,0.05,-0.1], 'damping': 0},
        {'amplitude': 0.5, 'fc': [0.3,0.25,0.45], 'damping': 0}
        ]

        nu_parameters = [
        {'amplitude': 0.5, 'fc': [0.05, 0.1,0.05], 'damping': 0},
        {'amplitude': 0.5, 'fc': [-0.3,-0.20,-0.1], 'damping': 0} 
        ]

    generation_start = time.time()
    print('Generating signals.')
    mu_original = signal.generate_multi_dimensional_sinusoid(D, N, mu_parameters) # shape((N,)*D)
    nu_original = signal.generate_multi_dimensional_sinusoid(D, N, nu_parameters) # shape((N,)*D)
    generation_stop = time.time()
    print('Generation took: ', generation_stop-generation_start , ' seconds.')
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


########## Main functions ###########
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
    
    # plot_transport_marginals([P_indices_core,P_values_core],source_mass_coords,target_mass_coords,N_core)

    # Initiate a list of transport plans for each level with the core level transport plan as thhe first element
    P_levels = [[[P_indices_core, P_values_core],source_mass_coords,target_mass_coords]]

    # Iterate through the levels of the multiscale OT
    for i in range(num_levels):
        # Set the current level scaling factors 
        scaling_factor = scaling_factors[i] # scaling between current coarse and fine levels
        cumulative_scaling = cumulative_scalings[i+1] # Cululatives scaling from original size to current fine level

        # Current coarse level transport plan
        P_coarse = P_levels[i][0]
        mu_coords_coarse = P_levels[i][1]
        nu_coords_coarse = P_levels[i][2]

        # Fine level size
        N_fine = sizes[i+1]

        # Generate fine level threshold for non-zero entries in the transport plan
        threshold_fine = 1e-3 / (N_fine**D)

        # Perfrom the optimal transport between levels
        P_fine,mu_coords_fine,nu_coords_fine = inter_level_OT(P_coarse,mu_coords_coarse,nu_coords_coarse,mu_original,nu_original,scaling_factor,cumulative_scaling,threshold_fine)
        
        # Add the fine level transport plan to the list of transport plans
        P_levels.append([P_fine,mu_coords_fine,nu_coords_fine])
    
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
    P_core, source_mass_coords, target_mass_coords = ot.sinkhorn_log(mu_core_coords,mu_core_values,nu_core_coords,nu_core_values) 

    # Reformat the transport plan P_core to get the indices and values
    P_indices_core, P_values_core = nonzero_indices_and_values(np.array(P_core), threshold_core)
    
    return P_indices_core, P_values_core, source_mass_coords, target_mass_coords

def inter_level_OT(P_coarse,mu_coords_coarse,nu_coords_coarse,mu_original,nu_original,scaling_factor,cumulative_scaling,threshold_fine):
    """
    Perform inter-level optimal transport between the coarse and fine levels. Fo each coarse transport path, 
    the corresponding fine grid-blocks--as sub-blocks of the original source and target distributions--are constructed 
    and optimal transport between those blocks is computed. The fine-level transport plans are accumulated and returned as a global fine-level transport plan.

    Parameters
    ----------
    P_coarse: [indices, values]
        Coarse-level transport plan. Each entry of indices is a pair of indices (i,j) which are mapped to full coarse
        space coordinates through mu_coords_coarse and nu_coords_coarse. Each entry of values are the transported masses.
    mu_coords_coarse: array, shape (M_mu_coarse, D)
        Coarse-level source coordinates, where M_mu is the number of support points.
    nu_coords_coarse: array, shape (M_nu_coarse, D)
        Coarse-level target coordinates, where M_nu is the number of support points.
    mu_original: array, shape ((N,)*D)
        Original source distribution in D dimensions.
    nu_original: array, shape ((N,)*D)
        Original target distribution in D dimensions.
    scaling_factor: int
        Scaling factor between the coarse and fine levels. It defines the size of the fine grid block corresponding to each coarse grid point.
    cumulative_scaling: int
        Cumulative scaling factor from the original size to the current fine level. It defines the size of the fine grid block corresponding 
        to each coarse grid point in the original distribution.
    threshold_fine: float
        Threshold for non-zero entries in the fine-level transport plan. 

    Returns
    -------
    P_fine_global: [indices, values]
        Global fine-level transport plan. Each entry of indices is a pair of indices (i,j) which are mapped to full fine
        space coordinates through mu_coords_coarse and nu_coords_coarse. Each entry of values are the transported masses.
    mu_coords_fine_global: array, shape (M_mu_fine, D)
        Fine-level source coordinates, where M_mu_fine is the number of support points in the fine level.
    nu_coords_fine_global: array, shape (M_nu_fine, D)
        Fine-level target coordinates, where M_nu_fine is the number of support points in the fine level.
    """
    # Extract the indices and values from the coarse transport plan
    P_indices_coarse = P_coarse[0] 
    P_values_coarse = P_coarse[1]

    # Initiate lists to store fine level block transport plans for each coarse path 
    P_fine = []
    source_mass_coords_fine = []
    target_mass_coords_fine = []
    
    # Iterate through all coarse source support points
    for i, source_coord_coarse in enumerate(mu_coords_coarse):
        source_indices = np.argwhere(P_indices_coarse[:,0] == i) # All transport paths from the current source point
        source_mass = np.sum(P_values_coarse[source_indices]) # Total mass transported from the current source point

        # Get the block of non-zero fine-level coordinates corresponding to the current coarse source point 
        mu_coords_fine, mu_values_fine = nonzero_fine_block(source_coord_coarse,scaling_factor,cumulative_scaling,mu_original,threshold_fine)

        # Iterate through all coarse target support points
        for j, target_coord_coarse in enumerate(nu_coords_coarse):
            target_indices = np.argwhere(P_indices_coarse[:,1] == j) # All transport paths from the current source point
            target_mass = np.sum(P_values_coarse[target_indices]) # Total mass transported to the current target point

            # Get the block of non-zero fine-level coordinates corresponding to the current coarse target point
            nu_coords_fine, nu_values_fine = nonzero_fine_block(target_coord_coarse,scaling_factor,cumulative_scaling,nu_original,threshold_fine)

            source_target_index = np.argwhere((P_indices_coarse[:,0] == i) & (P_indices_coarse[:,1] == j)) # Index of the transport path between the current source and target points
            source_target_mass = np.sum(P_values_coarse[source_target_index]) # Total mass transported over the current path

            # Make sure there is mass to transport and that the fine blocks are not empty
            if mu_coords_fine.size == 0 or nu_coords_fine.size == 0:
                continue
            if source_mass == 0 or target_mass == 0 or source_target_mass == 0:
                continue

            # Relative transport masses for the current path (mu_values_fine = nu_values_fine = source_target_mass)
            mu_values_fine = mu_values_fine / mu_values_fine.sum() * source_target_mass
            nu_values_fine = nu_values_fine / nu_values_fine.sum() * source_target_mass

            # Perform the fine-level OT between the fine source and target blocks
            P, mu_coords, nu_coords,log = ot.sinkhorn_log(mu_coords_fine,mu_values_fine,nu_coords_fine,nu_values_fine,return_log=True)
            
            # Add the fine-level block transport plan to the lists 
            P_fine.append(P)
            source_mass_coords_fine.append(mu_coords)
            target_mass_coords_fine.append(nu_coords)

    # Accumulate the fine-level block transport plans into a full fine-level transport plan
    P_fine_full, mu_coords_fine_full, nu_coords_fine_full = accumulate_fine_transport(P_fine, source_mass_coords_fine, target_mass_coords_fine)
    
    return P_fine_full, mu_coords_fine_full, nu_coords_fine_full

def accumulate_fine_transport(P_blocks, mu_coords_blocks, nu_coords_blocks):
    """
    Merge fine-level transports from multiple blocks into a full fine-level transport plan.
    
    Parameters
    ----------
    P_blocks : list of P from each block, where P = [indices, values]. Indices map to full coordinates through mu_coords_blocks and nu_coords_blocks
    mu_coords_blocks : list of mu_coords for each block
    nu_coords_blocks : list of nu_coords for each block
    
    Returns
    -------
    P_fine : [indices, values] for full fine-level transport. Indices map to full coordinates through mu_coords_fine and nu_coords_fine
    mu_coords_fine : array of unique fine source coordinates
    nu_coords_fine : array of unique fine target coordinates
    """
    
    # Flatten all coordinates from all blocks
    all_mu_coords = np.vstack(mu_coords_blocks)
    all_nu_coords = np.vstack(nu_coords_blocks)

    # Unique coordinates (remove duplicates from overlapping blocks) and get reverse mapping to global indices
    mu_unique, mu_idx_map = np.unique(all_mu_coords, axis=0, return_inverse=True)
    nu_unique, nu_idx_map = np.unique(all_nu_coords, axis=0, return_inverse=True)

    # Build global indices for P_blocks
    P_indices = []
    P_values = []

    mu_offset = 0
    nu_offset = 0
    # Iterate through each block transport plan and map local indices to global indices
    for block_idx, P in enumerate(P_blocks):
        indices, values = nonzero_indices_and_values(P,normalise=False) # Get non-zero entries of the block transport plan
        for (i_local, j_local), val in zip(indices, values): # Local indices within the block
            global_i = mu_idx_map[mu_offset + i_local] # Map to global index using the reverse mapping
            global_j = nu_idx_map[nu_offset + j_local] # Map to global index using the reverse mapping
            P_indices.append([global_i, global_j]) # Append the global index pair to collective list
            P_values.append(val) # Append the corresponding value to collective list
        mu_offset += len(mu_coords_blocks[block_idx]) # Update offsets for next block
        nu_offset += len(nu_coords_blocks[block_idx]) # Update offsets for next block
    
    P_indices = np.array(P_indices, dtype=int)
    P_values = np.array(P_values, dtype=float)
    P_values /= np.sum(P_values)

    if P_indices.size == 0:
        # Nothing transported at this level
        return [np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=float)], mu_unique, nu_unique

    # return the full fine-level transport plan and unique coordinates where all transports over the same coordinates are summed up
    return [np.array(P_indices), np.array(P_values)], mu_unique, nu_unique

########## Helper functions ###########

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

def nonzero_indices_and_values(A, threshold = 1e-6, normalise = True):
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

    if normalise:
        values /= np.sum(values)  # Normalize values to sum to 1
        return indices, values
    
    return indices, values

def nonzero_fine_block(coarse_coord, scaling_factor, cumulative_scaling, psd_original, threshold):
    """
    Retrieve the fine-grid support points from the block of fine coordinates corresponding to a given coarse support coordinate.

    Parameters
    ----------
    coarse_coord : array, shape (D,)
        Coordinate of the coarse support point in D dimensions.
    scaling_factor : int or array_like, shape (D,)
        Upsampling factor from coarse to fine grid. If scalar, applied equally across dimensions.
    cumulative_scaling : int or array_like, shape (D,)
        Cumulative scaling factor to map fine-grid coordinates to the original full-resolution space.
    psd_original : ndarray, shape ((N,)*D)
        Original high-resolution distribution (e.g., PSD) from which fine-grid masses are drawn.
    threshold : float
        Minimum relative threshold for retaining mass. Values below `threshold * scaling_factor[0]` 
        are treated as noise and discarded.

    Returns
    -------
    coords : ndarray, shape (M, D)
        Coordinates of fine-grid support points corresponding to the coarse point.
    values : ndarray, shape (M,)
        Mass values at the corresponding coordinates, aligned with `coords`.
        If no values exceed the threshold, both outputs are empty arrays.
    """
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
    
    if len(coords_list) > 0:
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

########## Plotting functions ##########
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




if __name__ == "__main__":
    main()