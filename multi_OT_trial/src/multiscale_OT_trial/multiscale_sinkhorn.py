
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

def main():
    N = 100
    D = 2 # Spacial dimension
    
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
            {'amplitude': 0.6, 'fc': [0.05, 0.1,0.05], 'damping': 0},
            {'amplitude': 1, 'fc': [-0.3,-0.20,-0.1], 'damping': 0} 
        ]
    mu_original = signal.generate_multi_dimensional_sinusoid(D, N, mu_parameters)
    nu_original = signal.generate_multi_dimensional_sinusoid(D, N, nu_parameters)

    multiscale_ot = multiscale_sinkhorn(D,mu_original,nu_original,[5,20,100])

def multiscale_sinkhorn(D,mu_original,nu_original,sizes):
    is_valid, scaling_factors = is_valid_scaling(sizes) 
    if not is_valid:
         raise ValueError("Invalid input. All sequential scaling divisions must be exact divisions.")
    
    N_original = sizes[-1]
    space_shape = np.shape(mu_original)
    num_levels=len(scaling_factors)

    mu_levels, nu_levels = generate_signal_levels(mu_original,nu_original,scaling_factors,sizes,D)

    ot.plot_imshow(mu_levels[0])

    P_core, mu_coords_core, nu_coords_core, log = core_level_OT(mu_levels[0],nu_levels[0])

    P_levels = [[P_core,mu_coords_core,nu_coords_core]]

    for i in range(num_levels):
        # N_fine = sizes[i+1]
        # N_coarse = sizes[i]
        # scaling_factor = scaling_factors[i]
        # mu_fine = mu_levels[i+1]
        # mu_coarse = mu_levels[i]
        # nu_fine = nu_levels[i+1]
        
        # P_coarse = P_levels[i]
       
        # threshold_fine = compute_adaptive_threshold(mu_fine, N_fine, D)      
        # threshold_coarse = compute_adaptive_threshold(mu_coarse, N_coarse, D)
        
        # P_fine = inter_level_OT(N_fine,N_coarse,P_coarse,mu_fine,nu_fine,threshold_coarse,threshold_fine,scaling_factor,D)

        N_fine = sizes[i+1]
        N_coarse = sizes[i]
        scaling_factor = scaling_factors[i]
        P_coarse = P_levels[i,0]
        source_coords_coarse = P_levels[i,1]
        target_coords_coarse = P_levels[i,2]

        mu_fine = mu_levels[i+1]
        nu_fine = nu_levels[i+1]

        threshold_fine = compute_adaptive_threshold(mu_fine, N_fine, D)      
        threshold_coarse = compute_adaptive_threshold(source_coords_coarse, N_coarse, D)



def inter_level_OT(P_coarse,mu_coarse,nu_coarse,mu_fine,nu_fine,threshold_fine,scaling_factor):
    P_coords_coarse = P_coarse[0]
    P_values_coarse = P_coarse[1]

    P_fine = []
    mu_coords_fine = []
    nu_coords_fine = []
    
    for i, souarce_coord_coarse in enumerate(mu_coarse):
        source_indices = np.argwhere(P_coords_coarse[:,0] == i ) # sparce OT indices of all transports from source indices
        source_mass = np.sum(P_values_coarse[source_indices])

        source_coords_fine = coarse_to_fine([souarce_coord_coarse],scaling_factor)

        mu_fine = mu_fine[source_coords_fine]

        for j, target_coord_coarse in enumerate(nu_coarse):
            target_indices = np.argwhere(P_coords_coarse[:,1] == j)
            target_mass = np.sum(P_values_coarse[target_indices])

            source_target_index = np.argwhere(P_coords_coarse[:,0] == i and P_coords_coarse[:,1] == j)
            source_target_mass = P_values_coarse[source_target_index]

            source_mass_ratio = source_target_mass / source_mass
            target_mass_ratio = source_target_mass / target_mass

            target_coords_fine = coarse_to_fine([target_coord_coarse],scaling_factor)

            P, mu_coords, nu_coords = ot.sinkhorn_log(source_coords_fine,target_coords_fine,threshold = threshold_fine)
            P_fine.append(P)
            mu_coords_fine.append[mu_coords]
            nu_coords_fine.append[nu_coords]
    
    return [P_fine,mu_coords_fine,nu_coords_fine]
            


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

def core_level_OT(mu_core,nu_core,N_core,D):
    min_threshold = 1e-3 / (N_core**D)  
    adaptive_threshold = compute_adaptive_threshold(mu_core)
    threshold = max(min_threshold,adaptive_threshold)    

    return ot.sinkhorn_log(mu_core,nu_core,threshold=threshold) 
     

def is_valid_scaling(sizes, coarse_to_fine=True):

    if coarse_to_fine:
        sizes = sizes[::-1]  # flip so we can reuse same logic

    scaling_factors = []

    current_size = sizes[0]
    for size in sizes[1:]:
        if current_size % size != 0:
            return False, []
        scaling_factor = current_size // size
        scaling_factors.append(scaling_factor)
        current_size = size

    if coarse_to_fine:
        # Flip back the results so they align with the original input order
        scaling_factors = scaling_factors[::-1]

    return True, scaling_factors

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