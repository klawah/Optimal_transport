import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import ot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))  # adds 'src' parent
import src.utils.signal_simulation as signal
import src.utils.optimal_transport_computation as opt_t
import src.utils.OT_interpolation as itp
import time

def main():

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

    multiscale_ot = OT_scaler(D,[100,20,5])
    multiscale_ot.thresholded_multiscale_OT(mu_parameters,nu_parameters)

def check_single_point_mass(P,mu= None,nu= None):
    source_mass = P.sum(axis=1)   # mass at each source grid point
    target_mass = P.sum(axis=0)   # mass at each target grid point

    src_nonzero = np.flatnonzero(source_mass)
    tgt_nonzero = np.flatnonzero(target_mass)

    print(f"Source nonzero indices: {src_nonzero}")
    print(f"Target nonzero indices: {tgt_nonzero}")

    if len(src_nonzero) == 1 and len(tgt_nonzero) == 1:
        print("✅ One point mass in source and target")
        return True
    else:
        print("❌ Not a single point mass in source/target")
        return False


class OT_scaler:
    def __init__(self,D,N_sizes):
        
        is_valid, scaling_factors, cumulative_scaling = self._is_valid_scaling(N_sizes)

        if is_valid:
            self.N_original = N_sizes[0]
            self.D = D
            self.scaling_factors = scaling_factors
            self.N_sizes = N_sizes
            self.cumulative_scaling = cumulative_scaling
        else:
            raise ValueError("Invalid input. All sequential scaling divisions must be exact divisions.")
        
    def thresholded_multiscale_OT(self,mu_parameters,nu_parameters,reg = 3e-1, threshold_fine = None, threshold_coarse = None):
        start = time.time() # start time for entire optimisation

        # Generated signals
        mu_original = signal.generate_multi_dimensional_sinusoid(self.D, self.N_original, mu_parameters)
        nu_original = signal.generate_multi_dimensional_sinusoid(self.D, self.N_original, nu_parameters)


        N_levels = self.N_sizes[::-1] # Size of N for al levels (coarse-to-fine)
        scaling_factors = self.scaling_factors[::-1] # Scaling factors between all levels (coarse-to-fine)
        num_levels=len(scaling_factors)

        mu_levels, nu_levels = self._generate_signal_levels(mu_original,nu_original) # Source and target distributions for all levels (coarse-to-fine)

        P_core = self._core_level_OT(mu_levels[0],nu_levels[0],N_levels[0],reg)

        # for mu in mu_levels:
        #     plot_imshow(mu)

        P_levels = [P_core]
        for i in range(num_levels):
            N_fine = N_levels[i+1]
            N_coarse = N_levels[i]
            mu_fine = mu_levels[i+1]
            mu_coarse = mu_levels[i]
            nu_fine = nu_levels[i+1]
            scaling_factor = scaling_factors[i]
            P_coarse = P_levels[i]

            if threshold_fine is None:
                min_threshold_fine = 1e-3 / (N_fine**self.D)  
                adaptive_threshold_fine = compute_adaptive_threshold(mu_fine)
                threshold_fine = max(min_threshold_fine,adaptive_threshold_fine)    
            if threshold_coarse is None:
                min_threshold_coarse = 1e-3 / (N_coarse**self.D)  
                adaptive_threshold_coarse = compute_adaptive_threshold(mu_coarse)
                threshold_coarse = max(min_threshold_coarse,adaptive_threshold_coarse)    

            P_fine = self._inter_level_OT(N_fine,N_coarse,P_coarse,mu_fine,nu_fine,threshold_coarse,threshold_fine,scaling_factor,reg)
            P_fine /= np.sum(P_fine)
            P_levels.append(P_fine)
        
        end = time.time() # end time for entire optimisation
        elapsed = end - start # total time for optimisation

        N = self.N_original
        D = self.D
        N_theta = 21
        theta_seq = np.linspace(0,1,N_theta)
        # check_single_point_mass(P_fine,mu_original,nu_original)
        itp_seq, sigma_used = itp.interpolate_multi_dimensional_OT(P_fine, D, theta_seq,smoothing_sigma = 0)
        print('done')

        print_nonzero_indices(itp_seq)
        
        if D == 2:
            for P_fine in P_levels:
                N = int(np.sqrt(len(P_fine[0])))
                plot_transport_distributions(P_fine,(N,N))
            
            plot_interpolations_2d(itp_seq,theta_seq,N)
            plot_psd_seq_unit_circle(itp_seq*0.8,N,theta_seq)

        if D == 3:
            plot_psd_seq_unit_sphere(itp_seq,N,theta_seq)


        coords_fine = coord_list(self.N_original,self.D)
        cost = self._get_cost(P_fine,coords_fine)
        print('Time elapsed: ', elapsed, '. Total cost: ',cost)

    

            
    def _core_level_OT(self,mu,nu,N_core,reg):
        min_threshold = 1e-3 / (N_core**self.D)  
        adaptive_threshold = compute_adaptive_threshold(mu)
        threshold = max(min_threshold,adaptive_threshold)    

        OT = opt_t.OptimalTransport(mu,nu,reg,self.D) # OT object
        P,log = OT.compute_thresholded_OT(threshold) # Fisrt row of P is a list of transported mass from grid point zero in sourdce grid to each  
        
        return P   

    def _inter_level_OT(self,N_fine,N_coarse,P_coarse,mu_fine,nu_fine,threshold_fine,threshold_coarse,scaling_factor,reg):
        # Create a mapping between a coarse grid point and its associated block of fine grid points (idex mapping)
        coords_fine = coord_list(N_fine,self.D) # A list of coordinates (row major order) in the fine grid
        coords_coarse = coord_list(N_coarse,self.D) # A list of coordinates (row major order) in the coarse grid
        coord_index_map = coordinate_mapping(coords_coarse,coords_fine,scaling_factor) # Maps the indices of the coarse coordinate list to a list of associated coordinates in a corresponding list of fine coordinates
    
        
        P_fine = np.zeros((N_fine**self.D, N_fine**self.D)) # Initialise a transport plan for the fine grid

        for coarse_source_index in range(N_coarse**self.D): # source index in the coarse grid
            coarse_target_vector = P_coarse[coarse_source_index] # vector of transports from that soarse index 

            for coarse_target_index in range(N_coarse**self.D): # target index in the coarse grid
                coarse_index_target_mass = coarse_target_vector[coarse_target_index] # amount of mass transported from current soarse index to current target index (coarse)
                if coarse_index_target_mass > threshold_coarse: # Only consider mass moved > 1e-10. All else considered noise

                    source_ratio = coarse_index_target_mass / np.sum(coarse_target_vector) # ratio of the source mass going to specific target index
                    target_ratio = coarse_index_target_mass / np.sum(P_coarse[:,coarse_target_index]) # ratio of target mass coming from specific source index

                    source_fine_indices = coord_index_map[coarse_source_index][1] # retrieves the list of fine indices associated with the specific coarse index
                    target_fine_indices = coord_index_map[coarse_target_index][1] # retrieves the list of fine indices associated with the specific coarse index

                    mu_fine_block = mu_fine.flatten()[source_fine_indices].reshape((scaling_factor,) * self.D) # Retrieves the part of mu that corresponds to the coarse grid point (mass values) (then reshaped to a square grid)
                    nu_fine_block = nu_fine.flatten()[target_fine_indices].reshape((scaling_factor,) * self.D) # Retrieves the part of nu that corresponds to the coarse grid point (mass values) (then reshaped to a square grid)

                    fine_scale_ot_block = self._fine_scale_ot(mu_fine_block,nu_fine_block,source_ratio,target_ratio,reg,threshold_fine) # Compute the fine scale optimal transport for the current block
                    P_fine[np.ix_(source_fine_indices, target_fine_indices)] += fine_scale_ot_block * coarse_index_target_mass # Insert this plan into the full fine scale optimal transport plan (scaled according to ratio of total mass = coarse_index_target_mass/1 (as sum(P) is always 1))
            #print(coarse_source_index)
        return P_fine

    def _generate_signal_levels(self,mu_original,nu_original):
        mu_signal_levels = [mu_original]
        nu_signal_levels = [nu_original]
        
        for scaling_level in range(len(self.scaling_factors)):
            scaling_factor = self.scaling_factors[scaling_level]
            N_coarse = self.N_sizes[scaling_level+1]

            mu_fine = mu_signal_levels[scaling_level]
            nu_fine = nu_signal_levels[scaling_level]

            mu_coarse = _downsample_signal(mu_fine, N_coarse, scaling_factor, self.D)
            nu_coarse = _downsample_signal(nu_fine, N_coarse, scaling_factor, self.D)

            mu_signal_levels.append(mu_coarse)
            nu_signal_levels.append(nu_coarse)


        return mu_signal_levels[::-1], nu_signal_levels[::-1]


    def _is_valid_scaling(self,sizes):
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

        return True, scaling_factors, cumulative_scaling

    def _get_cost_matrix(self, source_coords, target_coords = None, cost_metric='sqeuclidean'):
        if target_coords is None:
            return ot.dist(source_coords,source_coords,cost_metric)
        
        else:
            return ot.dist(source_coords,target_coords,cost_metric)
        

    def _get_cost(self, P, source_coords, target_coords = None, cost_metric='sqeuclidean'):
        if target_coords is None:
            cost_matrix =  ot.dist(source_coords,source_coords,cost_metric)
        
        else:
            cost_matrix = ot.dist(source_coords,target_coords,cost_metric)

        return np.sum(P * cost_matrix)
    
    def _fine_scale_ot(self,mu_fine_block,nu_fine_block,source_ratio,target_ratio,reg,threshold):
        mu_fine_scaled = mu_fine_block.copy()*source_ratio # Scale to mass transport ratio from specific source to specific target
        nu_fine_scaled = nu_fine_block.copy()*target_ratio # Scale to mass transport ratio from specific source to specific target
        mu_fine_scaled /= np.sum(mu_fine_scaled) # Renormalise
        nu_fine_scaled /= np.sum(nu_fine_scaled) # Renormalise
        OT = opt_t.OptimalTransport(mu_fine_scaled,nu_fine_scaled,reg,self.D) # bOT object for fine grid
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

def _downsample_signal(psd, N_coarse, scaling_factor,D):
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
    
    vmax = np.percentile(psd_seq, 100)
    vmin = 0

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()  # flatten in case of 2D array

    for i in range(N_theta):
        ax = axes[i]
        im = ax.imshow(
            psd_seq[:, i].reshape(N, N),
            origin='upper',
            cmap='viridis',
            aspect='equal',
            vmin = vmin,
            vmax = vmax
        )
        ax.set_title(f'theta = {theta_seq[i]:.2f}')
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)  # colorbar per subplot with fixed scale

    # Hide unused subplots
    for j in range(N_theta, len(axes)):
        axes[j].axis('off')

    fig.suptitle("OT Interpolation Distributions on 2D Grid", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    return fig

def plot_transport_distributions(P, grid_shape):
    """
    Visualize where mass starts and ends based on the transport plan P
    with consistent color scaling for both plots.
    
    Parameters:
    - P: (N^2, N^2) transport plan matrix
    - grid_shape: (N, N) shape of the spatial grid
    """

    # Compute marginal masses
    source_mass = np.sum(P, axis=1)  # sum over target → mass leaving each source point
    target_mass = np.sum(P, axis=0)  # sum over source → mass arriving at each target point

    # print(np.sum(source_mass), ' ', np.sum(target_mass))

    # Reshape to grid
    source_mass_grid = source_mass.reshape(grid_shape)
    target_mass_grid = target_mass.reshape(grid_shape)

    # Compute global color scale limits
    vmax = max(source_mass_grid.max(), target_mass_grid.max())
    vmin = 0

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(source_mass_grid, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title('Source Mass Distribution')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(target_mass_grid, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title('Target Mass Distribution')
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.show()


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


from mpl_toolkits.mplot3d import Axes3D

def plot_psd_seq_unit_circle(psd_seq, N, theta_seq=None):
    """
    Plots all interpolation steps from a 2D OT interpolation sequence on
    the unit circle, with distance from origin equal to the PSD value.

    Parameters
    ----------
    psd_seq : ndarray, shape (N^2, T)
        Sequence of interpolated distributions (flattened grid).
    N : int
        Number of samples along each dimension (sampling rate = 1).
    theta_seq : array-like, optional
        Interpolation parameters for labeling.
    """
    M, T = psd_seq.shape
    if M != N**2:
        raise ValueError("psd_seq first dimension must equal N^2")
    
    # Normalized frequency coordinates
    f1 = np.fft.fftshift(np.fft.fftfreq(N, d=1))
    f2 = np.fft.fftshift(np.fft.fftfreq(N, d=1))
    F1, F2 = np.meshgrid(f1, f2, indexing="ij")
    
    # Flatten
    F1_flat = F1.ravel()
    F2_flat = F2.ravel()
    
    # Compute angle
    phi = np.arctan2(F2_flat, F1_flat)
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Draw unit circle for reference
    circle_phi = np.linspace(0, 2 * np.pi, 500)
    ax.plot(np.cos(circle_phi), np.sin(circle_phi), 'k--', alpha=0.5)
    
    # Plot each time step with distance proportional to PSD value
    for t in range(T):
        mass = psd_seq[:, t]
        nonzero = mass > 0
        x = mass[nonzero] * np.cos(phi[nonzero])
        y = mass[nonzero] * np.sin(phi[nonzero])
        ax.scatter(x, y, s=20,
                   label=f"t={theta_seq[t]:.2f}" if theta_seq is not None else None)
    
    ax.set_xlabel("X (PSD * cos φ)")
    ax.set_ylabel("Y (PSD * sin φ)")
    ax.set_title("2D PSD Directions on Unit Circle (magnitude = PSD)")
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    
    # Legend outside
    if theta_seq is not None:
        ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
    
    plt.tight_layout()
    plt.show()

def plot_psd_seq_unit_sphere(psd_seq, N, theta_seq=None):
    """
    Plots 3D PSD sequence on a unit sphere with distance from origin = PSD magnitude.
    Legend is placed outside and shows the magnitude summary.
    
    Parameters
    ----------
    psd_seq : ndarray, shape (N^3, T)
        Sequence of interpolated distributions (flattened 3D grid).
    N : int
        Number of samples along each dimension (sampling rate = 1).
    theta_seq : array-like, optional
        Interpolation parameters for labeling.
    """
    M, T = psd_seq.shape
    if M != N**3:
        raise ValueError("psd_seq first dimension must equal N^3")
    
    # Normalized frequency coordinates
    f1 = np.fft.fftshift(np.fft.fftfreq(N, d=1))
    f2 = np.fft.fftshift(np.fft.fftfreq(N, d=1))
    f3 = np.fft.fftshift(np.fft.fftfreq(N, d=1))
    
    F1, F2, F3 = np.meshgrid(f1, f2, f3, indexing="ij")
    
    # Flatten
    F1_flat = F1.ravel()
    F2_flat = F2.ravel()
    F3_flat = F3.ravel()
    
    # Compute unit direction vectors
    vectors = np.vstack([F1_flat, F2_flat, F3_flat]).T
    norms = np.linalg.norm(vectors, axis=1)
    nonzero_mask = norms > 0
    unit_vectors = np.zeros_like(vectors)
    unit_vectors[nonzero_mask] = vectors[nonzero_mask] / norms[nonzero_mask, None]
    
    # Plot setup
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Reference unit sphere
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='k', alpha=0.1)
    
    # Plot each time step
    handles = []
    labels = []
    for t in range(T):
        mass = psd_seq[:, t]
        nonzero_mass = mass > 0
        x = mass[nonzero_mass] * unit_vectors[nonzero_mass, 0]
        y = mass[nonzero_mass] * unit_vectors[nonzero_mass, 1]
        z = mass[nonzero_mass] * unit_vectors[nonzero_mass, 2]
        scatter = ax.scatter(x, y, z, s=20)
        handles.append(scatter)
        if theta_seq is not None:
            labels.append(f"t={theta_seq[t]:.2f}, max={mass.max():.2f}")
        else:
            labels.append(f"t={t}, max={mass.max():.2f}")
    
    ax.set_xlabel("X (PSD * dir)")
    ax.set_ylabel("Y (PSD * dir)")
    ax.set_zlabel("Z (PSD * dir)")
    ax.set_title("3D PSD Directions on Unit Sphere (magnitude = PSD)")
    
    # Legend outside
    ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.show()

def print_nonzero_indices(psd_seq):
    """
    Prints indices of nonzero entries for all interpolation steps.

    Parameters
    ----------
    psd_seq : ndarray, shape (N^D, T)
        Interpolated distributions sequence.
    """
    N_flat, T = psd_seq.shape
    for t in range(T):
        nonzero_idx = np.flatnonzero(psd_seq[:, t])
        print(f"Step {t}: {len(nonzero_idx)} nonzero entries")
        print(nonzero_idx)



if __name__ == "__main__":
    main()

