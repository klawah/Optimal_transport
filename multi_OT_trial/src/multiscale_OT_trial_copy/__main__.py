import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))  # adds 'src' parent
import src.utils.signal_simulation as signal
import src.utils.optimal_transport_computation as opt_t
import src.utils.OT_interpolation as itp
import time
import numpy as np
import matplotlib.pyplot as plt

def main():

    # Define the fine and coarse spaces
    D = 2 # Spacial dimension
    N = 200 # Original (fine) grid axis dimension
    scaling_factor = 10 # Decides how many fine points are included in each of the coarse grid points (per axis). Small scaling factor --> smaller fine-grid blocks --> generally faster optimisation
    N_coarse = int(N/scaling_factor) # Coarse grid axis simension

    reg = 3e-1 # Regularisation paramater for the sinkhorn algorithm
    N_theta = 21 # Number of interpolation steps between soarce and target
    theta_seq = np.linspace(0,1,N_theta) # The values for the interpolation given the number of interpolation steps 
    sigma = 5 # Interpolation smoothing parameter
 
    # Create a mapping between a coarse grid point and its associated block of fine grid points (idex mapping)
    coords_fine = coord_list(N,D) # A list of coordinates (row major order) in the fine grid
    coords_coarse = coord_list(N_coarse,D) # A list of coordinates (row major order) in the coarse grid
    coord_index_map = coordinate_mapping(coords_coarse,coords_fine,scaling_factor) # Maps the indices of the coarse coordinate list to a list of associated coordinates in a corresponding list of fine coordinates
    
    # Signal parameters 
    if D == 2:
        mu_parameters = [
            {'amplitude': 0.7, 'fc': [0.35,-0.3], 'damping': 0.1}, 
            {'amplitude': 0.5, 'fc': [0.27,-0.1], 'damping': 0.09},
            {'amplitude': 1.1, 'fc': [-0.2,0.3], 'damping': 0.2}
        ]

        nu_parameters = [
            {'amplitude': 0.7, 'fc': [0.1,-0.3], 'damping': 0.1}, 
            {'amplitude': 1, 'fc': [-0.2,0.05], 'damping': 0.05}
        ]
  
    if D ==3:
        mu_parameters = [
            {'amplitude': 2, 'fc': [0.45,-0.3,0.14], 'damping': 0.2}, 
            {'amplitude': 0.5, 'fc': [0.23,-0.06,0.1], 'damping': 0.08},
            {'amplitude': 1.1, 'fc': [-0.2,0,0], 'damping': 0.15}]

        nu_parameters = [
            {'amplitude': 1.8, 'fc': [0.4, -0.25, 0.12], 'damping': 0.5},
            {'amplitude': 0.7, 'fc': [-0.15, 0.05, 0.08], 'damping': 0.2}
        ]

    # Generated signals
    mu = signal.generate_multi_dimensional_sinusoid(D, N, mu_parameters)
    nu = signal.generate_multi_dimensional_sinusoid(D, N, nu_parameters)

    # Signals in the coarse grid (sum of mass in each block)
    mu_coarse = downsample_signal(mu,N_coarse,scaling_factor,D)
    nu_coarse = downsample_signal(nu,N_coarse,scaling_factor,D)

    min_threshold_fine = 1e-3 / (N**D)  
    adaptive_threshold_fine = compute_adaptive_threshold(mu)
    threshold_fine = max(min_threshold_fine,adaptive_threshold_fine)
    print(threshold_fine)

    min_threshold_coarse = 1e-3 / (N_coarse**D)  
    adaptive_threshold_coarse = compute_adaptive_threshold(mu_coarse)
    threshold_coarse = max(min_threshold_coarse,adaptive_threshold_coarse)
    print(threshold_coarse)
    
    # plot_imshow(nu_coarse)
    
    # Compute the OT over the coarse grid
    OT = opt_t.OptimalTransport(mu_coarse,nu_coarse,reg,D) # OT object
    start = time.time() # start time for entire optimisation
    P,log = OT.compute_thresholded_OT(threshold_coarse) # Fisrt row of P is a list of transported mass from grid point zero in sourdce grid to each grid point in target grid (flattened)
    
    # # Make a more sparse copy of P by removing small transports (view as noise) and re-normalise
    # P_sparse = P.copy()
    # P_sparse[P_sparse < 1e-10] = 0.0
    # if np.sum(P_sparse) > 0:
    #     P_sparse /= np.sum(P_sparse)

    
    P_fine = np.zeros((N**D, N**D)) # Initialise a transport plan for the fine grid
    num_transport_paths = 0 # Count for how many different transports are made (how much is the energy split)

    for coarse_source_index in range(N_coarse**D): # soarse index in the coarse grid
        coarse_target_vector = P[coarse_source_index] # vector of transports from that soarse index 

        for coarse_target_index in range(N_coarse**D): # target index in the coarse grid
            coarse_index_target_mass = coarse_target_vector[coarse_target_index] # amount of mass transported from current soarse index to current target index (coarse)
            if coarse_index_target_mass > threshold_coarse: # Only consider mass moved > 1e-10. All else considered noise
                num_transport_paths += 1 # Mass is moved --> add to count

                source_ratio = coarse_index_target_mass / np.sum(coarse_target_vector) # ratio of the source mass going to specific target index
                target_ratio = coarse_index_target_mass / np.sum(P[:,coarse_target_index]) # ratio of target mass coming from specific source index

                source_fine_indices = coord_index_map[coarse_source_index][1] # retrieves the list of fine indices associated with the specific coarse index
                target_fine_indices = coord_index_map[coarse_target_index][1] # retrieves the list of fine indices associated with the specific coarse index

                mu_fine_block = mu.flatten()[source_fine_indices].reshape((scaling_factor,) * D) # Retrieves the part of mu that corresponds to the coarse grid point (mass values) (then reshaped to a square grid)
                nu_fine_block = nu.flatten()[target_fine_indices].reshape((scaling_factor,) * D) # Retrieves the part of nu that corresponds to the coarse grid point (mass values) (then reshaped to a square grid)

                fine_scale_ot_block = fine_scale_ot(mu_fine_block,nu_fine_block,source_ratio,target_ratio,reg,D,threshold_fine) # Compute the fine scale optimal transport for the current block
                P_fine[np.ix_(source_fine_indices, target_fine_indices)] += fine_scale_ot_block * coarse_index_target_mass # Insert this plan into the full fine scale optimal transport plan (scaled according to ratio of total mass = coarse_index_target_mass/1 (as sum(P) is always 1))
                print(coarse_source_index, ', ', coarse_target_index)
           
                
    end = time.time() # end time for entire optimisation
    elapsed = end - start # total time for optimisation
    print('Coarse transport paths: ', num_transport_paths,'. Time elapsed: ', elapsed, '. Total cost: ', )
    print(format(1e-10, '.10f'), ' ', format(1e-5, '.10f'), ' ', 1/N**D)

    # Perform the interpolation and plot
    # itp_seq, sigma_used = itp.interpolate_multi_dimensional_OT(P_fine, D, theta_seq, sigma)
    # if D < 3:
    #     figure1 = plot_interpolations_2d(itp_seq,theta_seq,N)

    if D < 3:
        plot_transport_distributions(P,(N_coarse,N_coarse))
        plot_transport_distributions(P_fine,(N,N))


def plot_transport_distributions(P, grid_shape):
    """
    Visualize where mass starts and ends based on the transport plan P.
    
    Parameters:
    - P: (N^2, N^2) transport plan matrix
    - grid_shape: (N, N) shape of the spatial grid
    """

    # Compute marginal masses
    source_mass = np.sum(P, axis=1)  # sum over target → mass leaving each source point
    target_mass = np.sum(P, axis=0)  # sum over source → mass arriving at each target point

    # Reshape to grid
    source_mass_grid = source_mass.reshape(grid_shape)
    target_mass_grid = target_mass.reshape(grid_shape)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(source_mass_grid, cmap='viridis')
    axes[0].set_title('Source Mass Distribution')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(target_mass_grid, cmap='viridis')
    axes[1].set_title('Target Mass Distribution')
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()

    
def fine_scale_ot(mu_fine_block,nu_fine_block,source_ratio,target_ratio,reg,D,threshold):
    mu_fine_scaled = mu_fine_block.copy()*source_ratio # Scale to mass transport ratio from specific source to specific target
    nu_fine_scaled = nu_fine_block.copy()*target_ratio # Scale to mass transport ratio from specific source to specific target
    mu_fine_scaled /= np.sum(mu_fine_scaled) # Renormalise
    nu_fine_scaled /= np.sum(nu_fine_scaled) # Renormalise
    OT = opt_t.OptimalTransport(mu_fine_scaled,nu_fine_scaled,reg,D) # bOT object for fine grid
    P,log = OT.compute_thresholded_OT(threshold) # Compute the OT (dense function removes close-to-zero mass distributions in mu and nu)

    return P


def coordinate_mapping(coords_coarse, coords_fine, scaling_factor):
    index_map = [] # Initialise a map of indexes that will correspond to the block of coordinate points in the fine grid associated with a coordinate point in the coarse grid
    
    for i, coarse_coord in enumerate(coords_coarse): # iterate through the coordinate points (index i and coordinate, coarse_coord) in the coarse grid
        start = coarse_coord * scaling_factor # Starting coordinate point for that coarse point in the fine grid (upper left corner)
        end = start + scaling_factor # End coordinate point for the block (lower right corner)

        block = np.all((coords_fine >= start) & (coords_fine < end), axis=1) # Find all fine grid point coordinates within the current block (iterate all elements along axis 1 (indices) and compare the elements with start/end). Returns a boolean array of shape (N**D,)
        block_indices = np.where(block)[0]  # Returns an array containing the indices where block is True

        index_map.append([i, block_indices.tolist()]) # Map where i is the index in coarse_grid and block_indices are the associated indices in fine_grid

    return index_map
    
def coord_list(N,D):
    axes = [np.linspace(0,N-1,N) for _ in range(D)] # Define each axis of the grid
    grid = np.meshgrid(*axes, indexing = 'ij') # Create a meshgrid object with the given axis length
    coords = np.stack([g.ravel() for g in grid], axis=-1) # List of coordinates in the grid (row major order)
    return coords.astype(int)

def downsample_signal(psd, N_coarse, scaling_factor,D):
    psd_coarse = np.zeros((N_coarse,) * D) # Initialise the coarse distribution grid
    for index in np.ndindex((N_coarse,) * D): # Iterate through all coordinates in the D-dimensional grid
        block = tuple(
            slice(i * scaling_factor, (i + 1) * scaling_factor) # Get a D-dimensional slice ("coarse block") of size scaling_factor in each dimension, with its upper left "corner" at the starting location of that coarse grid. Contains relevant fine grid indices (row major)
            for i in index
        )
        psd_coarse[index] = np.sum(psd[block]) # Sum up all the mass in the relevant block as the mass fo that coarse distribution grid point
    return psd_coarse


def plot_imshow(psd, name = 'spectrum'):

    plt.figure(figsize=(6, 5))
    plt.imshow(psd, extent=[-0.5, 0.5, 0.5, -0.5], origin='upper', aspect='auto', cmap='viridis')
    plt.colorbar(label='Normalized Power')
    plt.title(f'Power Spectral Density ({name})')
    plt.xlabel('Frequency axis 1')
    plt.ylabel('Frequency axis 2')
    plt.tight_layout()
    plt.show()

def plot_block_grid_indices(block, source_fine_indices, N):
    coords = np.array(np.unravel_index(source_fine_indices, (N, N))).T
    i_min, j_min = coords.min(axis=0)
    i_max, j_max = coords.max(axis=0) + 1

    extent = [j_min, j_max, i_max, i_min]  # flip vertical order

    plt.imshow(block, extent=extent, origin='upper', aspect='auto', cmap='viridis')
    plt.title(f'Block in fine grid indices ({i_min}:{i_max}, {j_min}:{j_max}) (origin upper)')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    plt.colorbar()
    plt.show()

def plot_interpolations_2d(psd_seq, theta_seq, N):
    N_theta = psd_seq.shape[1]
    nrows = 3
    ncols = int(np.ceil(N_theta / nrows))

    figure = plt.figure(figsize=(4 * ncols, 4 * nrows))
    
    for i in range(N_theta):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(psd_seq[:, i].reshape(N, N), origin='upper', cmap='viridis')
        plt.title(f'theta = {theta_seq[i]:.2f}')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')

    plt.suptitle("OT Interpolation Distributions on 2D Grid")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return figure

def compute_adaptive_threshold(P, top_percentile=80, min_ratio=0.05):
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
    return threshold



if __name__ == "__main__":
    main()

