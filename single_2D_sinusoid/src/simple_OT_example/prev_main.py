import simple_OT_example.find_OT_plan as opt
import simple_OT_example.interpolate_OT_plan as itp
import utils.signals as signals
import utils.plots as plot
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom

def main():

    # Number of samples and sampling frequency (same everywhere) --> Source sapce and target space are equal (and saquare)
    N = 100
    fs = 10

    # Signal frequencies
    fc_x = np.array([0.5 , 0.08])                   # Source signal
    fc_y = np.array([[0.15 , -0.4],[0.4 , 0.33]])   # Target signal

    # Define axis ranges (time domain)
    t_row = np.arange(N) / fs
    t_col = np.arange(N) / fs

    # Generate sinusoids
    signal1 = signals.create_2d_sinusoid(t_row, t_col, fc_x[0], fc_x[1])
    signal2 = signals.create_2d_sinusoid(t_row, t_col, fc_y[0,0], fc_y[0,1]) + signals.create_2d_sinusoid(t_row,t_col,fc_y[1,0],fc_y[1,1])

    # FFT2 generated power spectrums
    psd1 = signals.produce_2d_psd(signal1)
    psd2 = signals.produce_2d_psd(signal2)

    # Generate zero-centered frequency axes
    f_row,f_col = signals.generate_frequency_axes(N,fs) 

    # Plot signals and spectrums
    plot.plot_src_and_target_signals(signal1,signal2,t_row,t_col)
    plot.plot_src_and_target_psds(psd1,psd2,f_row,f_col)

    # # Plot the transport plan in the "bin index space"
    # plot.plot_transport_plan(T_2D)
    # plot.block_plot_transport_plan(T_2D,sigma = 2)

    N_theta = 21
    theta_seq = np.linspace(0,1,N_theta)

    # run_computations(N, psd1, psd2, theta_seq)

    data = np.load('/Users/klarawahlden/Documents/multidimentional_OT/results.npz')
    mu_seq = data['mu_seq']
    
    # plot.plot_displacement_interpolation(mu_seq, mu_seq_smooth, theta_seq)


    # Select 9 evenly spaced indices
    indices = np.linspace(0, N_theta - 1, 9, dtype=int)

    # === Plot ===
    fig, plots = plt.subplots(3, 3, figsize=(14, 14))
    plots = plots.flatten()

    # Get consistent color scale
    vmin = mu_seq.min()
    vmax = mu_seq.max()

    for idx, k in enumerate(indices):
        p = plots[idx]
        im = p.imshow(
            mu_seq[:, :, k],  # shift zero-freq to center
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


    print("done")

def run_computations(N, psd1, psd2, theta_seq):


    T_4D = solve_problem(psd1,psd2,N,solver = 'sinkhorn', bin_indices=False)

    mu_seq,mu_seq_smooth = itp.displacement_interpolation_2d(T_4D, theta_seq)
    
    np.savez('results', T_4D = T_4D, mu_seq = mu_seq, mu_seq_smooth = mu_seq_smooth)

def downsample_problem(downsample_factor, psd1, psd2, N):

    psd1_coarse = zoom(psd1,downsample_factor,order=1)
    psd2_coarse = zoom(psd2,downsample_factor,order=1)
    N_coarse = int(N * downsample_factor)

    return psd1_coarse, psd2_coarse, N_coarse


def solve_problem(psd1 , psd2, N, solver = 'sinkhorn', bin_indices = True):    
   
    # Find the optimal transport plan (T_2D defined over "bin indices" (p,q))
    if solver == 'sinkhorn':
        T_2D , _ = opt.sinkhorn_2d_OT(psd1,psd2)
    elif solver == 'cvxpy':
        T_2D , _ = opt.cvx_2d_OT(psd1,psd2)
    else:
        raise ValueError(f"Unknown solver '{solver}'. Choose 'sinkhorn' or 'cvxpy'.")
    
    # Translate T_2D to be defined over "grid indices" (i,j,k,l)
    T_4D = T_2D.reshape((N, N, N, N))       

    if bin_indices:
        return T_2D  
    else: 
        return T_4D                                        


def create_transport_list(T_2D, N, f_row ,f_col):

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

if __name__ == "__main__":
    main()