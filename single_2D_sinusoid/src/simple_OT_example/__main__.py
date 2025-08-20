import simple_OT_example.find_OT_plan as opt
import simple_OT_example.interpolate_OT_plan as itp
import simple_OT_example.simulate_signal as sim
import utils.plots as plot
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import os
from pathlib import Path

def main():

    #### Initialise ####

    # Number of samples and sampling frequency (same everywhere) --> Source sapce and target space are equal (and saquare)
    N = 200
    fs = 10

    # Signal frequencies
    fc_1 = np.array([0.5 , 0.08])                                   # Source signal
    fc_2 = np.array([[0.15 , -0.4],[0.4 , 0.33]])                   # Target signal

    # Optimisation library to use for finding the OT plan
    solver = 'sinkhorn'

    # Simulate the source and target signals and get their psds (mu, nu)
    mu, nu, f_row, f_col = sim.simulate_signals(N,fs,fc_1,fc_2,include_plots=False)

    plot.plot_src_and_target_psds(mu,nu,f_row,f_col)

    #### Optimisation ####

    # Downsample the problem if cvxpy is used
    if solver == 'cvxpy' and N > 30:
        print('downsample problem')
        mu, nu, N = downsample_problem(30/N, mu, nu, N)

    # Calculate and save the OT plan if not already done
    OT_data_file_path = Path(f'single_2D_sinusoid/src/simple_OT_example/result_data/OT_plan_{solver}.npy')
    if OT_data_file_path.exists():
        print("File already exists.")
        T = np.load(OT_data_file_path, allow_pickle=True)
    else:
        print('compute OT')
        T = solve_OT_problem(mu,nu,N,solver = solver)

    # 4D shape before interpolation
    if np.shape(T) == (N*N,N*N):
        # plot.block_plot_transport_plan(T,sigma = 2)
        T_4D = T.reshape(N,N,N,N)
    else:
        T_4D = T

    #### Interpolation ####

    N_theta = 21
    theta_seq = np.linspace(0,1,N_theta)
    itp_data_file_path = Path(f'single_2D_sinusoid/src/simple_OT_example/result_data/interpolation_{solver}.npz')

    if itp_data_file_path.exists():
        print("File already exists.")
    else:
        print('compute interpolation')
        itp_seq,itp_seq_smooth = itp.displacement_interpolation_2d(T_4D, theta_seq)
        np.savez(itp_data_file_path, itp_seq=itp_seq, itp_seq_smooth = itp_seq_smooth)

    data = np.load(itp_data_file_path, allow_pickle=True)

    itp_data = data['itp_seq']
    itp_data_smooth = data['itp_seq_smooth']

    #### Interpolation Plot ####

    # Select 9 evenly spaced indices
    indices = np.linspace(0, N_theta - 1, 9, dtype=int)

    # Create plots
    fig, plots = plt.subplots(3, 3, figsize=(14, 14))
    plots = plots.flatten()

    # Set colour scale
    vmin = itp_data.min()
    vmax = itp_data.max()

    # Plot the plots
    for idx, k in enumerate(indices):
        p = plots[idx]
        im = p.imshow(
            itp_data[:, :, k],  
            extent=[f_row[0], f_row[-1], f_col[0], f_col[-1]],
            origin='lower',
            cmap='viridis',
            vmin=vmin,
            vmax=vmax
        )
        p.set_title(f'Interpolated Spectrum θ={theta_seq[k]:.2f}')
        p.set_xlabel('Frequency col (Hz)')
        p.set_ylabel('Frequency row (Hz)')
        p.set_xlim(-0.6, 0.6)
        p.set_ylim(-0.6,0.6)
        fig.colorbar(im, ax=p, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    # plot.plot_interpolation_sequence_3d(itp_data,theta_seq)

    print("done")

# ------ Internal ------ #

def downsample_problem(downsample_factor, mu, nu, N):
    """
    Reduces the size of the problem by by (1 - downsample_factor) in each dimension. 

    Parameters
    ----------
    - downsample_factor: Each dimension size is reduced to N * downsample_factor. The psd arrays are zoomed using scipy.zoom by this factor
    - mu: original source distribution
    - nu: original target distribution
    - N: original problem size (dimension wise)

    Returns
    -------
    - mu_coarse: rezised source distribution with lower resolution
    - nu_coarse: rezised source distribution with lower resolution
    - N_coarse: new, reduced promlem size
    """

    # Zoom psd arrays using bilinerar interpolation
    # Computes each output pixel as a weighted average of its four nearest neighbors in the input array
    mu_coarse = zoom(mu,downsample_factor,order=1)                      
    nu_coarse = zoom(nu,downsample_factor,order=1)

    # Re-normalise the psds
    mu_coarse = mu_coarse / np.sum(mu_coarse)
    nu_coarse = nu_coarse / np.sum(nu_coarse)

    # Adjust dimension size according to the downsample_factor
    N_coarse = int(N * downsample_factor)

    return mu_coarse, nu_coarse, N_coarse


def solve_OT_problem(mu , nu, N, solver = 'sinkhorn', return_2D = True):
    """
    Find the optimal transport plan by solving the OT problem on a square 2D domain. Assumes the source and target distributions are defined over a square grid of shape (N, N).
    The plan is saved as a .npy file named 'OT_plan_sinkhorn.npy' or 'OT_plan_cvxpy.npy' depending on the solver used.

    Parameters
    ----------
    - mu: source mass distribution of shape (N,N)
    - nu: target mass distribution of shape (N,N)
    - N: grid dimension
    - solver: library to be used for the optimisation ('sinkhorn' or 'cvxpy'). Default of 'sinkhorn'
    - bin_indices: If True, return the transport plan in flattened 2D (bin index) form of shape (N², N²). If False, return the transport plan reshaped to 4D (grid index) form (N, N, N, N).

    Returns
    -------
    - T: Optimal transport plan
    """ 

    # Assure distribution metric spaces are of the correct dimensions
    if mu.shape != (N, N) or nu.shape != (N, N):
        raise ValueError(f"mu and nu must have shape ({N}, {N})")
   
    # Find the optimal transport plan (T_2D defined over "bin indices" (p,q))
    if solver == 'sinkhorn':
        T_2D , _ = opt.sinkhorn_2d_OT(mu,nu)
        file_path = 'single_2D_sinusoid/src/simple_OT_example/result_data/OT_plan_sinkhorn.npy'
    elif solver == 'cvxpy':
        T_2D , _ = opt.cvx_2d_OT(mu,nu)
        file_path = 'single_2D_sinusoid/src/simple_OT_example/result_data/OT_plan_cvxpy.npy'
    else:
        raise ValueError(f"Unknown solver '{solver}'. Choose 'sinkhorn' or 'cvxpy'.")
    
    # Return either flat or reshaped version
    if return_2D:
        T = T_2D
    else: 
        T = T_2D.reshape((N, N, N, N))                                                          # Map with "grid indices" (i,j,k,l)

    # Create directory if it doesn't exist and save the plan
    os.makedirs('single_2D_sinusoid/src/simple_OT_example/result_data', exist_ok=True)
    np.save(file_path, T)
    
    return T                                        


def create_transport_list(T_2D, N, f_row ,f_col):
    """
    Converts a 2D transport plan in the bin index form into a sparce list where each entry holds source and target grid index pairs as well as the mass transported between those points.

    Parameters
    ----------
    - T_2D: OT plan in 2D bin index form of shape (N*N,N*N)
    - N: grid dimension size 
    - f_row: grid row axis in the frequency domain
    - f_col: grid column axis in the frequency domain

    Returns
    -------
    - transport_list: sparce list format of the original transport plan
    
    """

    # Create gridspace
    F_row, F_col = np.meshgrid(f_row, f_col, indexing='ij')     # Grid of frequency axes
    points = np.vstack([F_row.ravel(), F_col.ravel()]).T        # Convert grid to a 2D array where points[b] gives the corresnding "grid indices" for the "bin index" b
    

    # Create list of mass transported between points
    transport_list = []                                         # Sparse list representation of transport plan. Each entry is [ [ (i,j) , (j,k) ] , T_{i,j,k,l} ] 
    for src_idx in range(N**2):
        for tgt_idx in range(N**2):
            mass = T_2D[src_idx, tgt_idx]
            if mass > 1e-6:                                     # Threshold to ignore neglable transports
                src = points[src_idx]
                tgt = points[tgt_idx]
                transport_list.append([[tuple(src), tuple(tgt)], mass])

    return transport_list


if __name__ == "__main__":
    main()