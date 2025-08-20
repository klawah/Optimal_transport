import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize

def plot_src_and_target_signals(signal1, signal2, t_row, t_col): 
    
    fig, (s1,s2) = plt.subplots(1,2,num = 'signal')

    s1.imshow(np.real(signal1), extent = (t_row[0],t_row[-1],t_col[0],t_col[-1]), cmap="gray")
    s1.set_xlabel('t_row')
    s1.set_ylabel('t_col')
    s1.set_xlim(0,10)
    s1.set_ylim(0,10)
    s1.set_title('Signal 1')

    s2.imshow(np.real(signal2), extent = (t_row[0],t_row[-1],t_col[0],t_col[-1]), cmap="gray")
    s2.set_xlabel('t_row')
    s2.set_ylabel('t_col')
    s2.set_xlim(0,10)
    s2.set_ylim(0,10)
    s2.set_title('Signal 2')

    plt.tight_layout()
    plt.show()

def plot_src_and_target_psds(psd1,psd2,f_row,f_col):

    # Plot the signals in frequency domain (separate source and target) 
    fig, (p1, p2) = plt.subplots(1, 2, num='psd', figsize=(10,5))
    vmin = min(np.min(psd1), np.min(psd2))
    vmax = max(np.max(psd1), np.max(psd2))
    im1 = p1.imshow(
        psd1,
        extent=[f_row[0], f_row[-1], f_col[0], f_col[-1]],
        origin='lower',
        vmin=vmin,
        vmax=vmax,
        aspect = 'equal'
    )
    p1.set_xlim(-0.6, 0.6)
    p1.set_ylim(-0.6, 0.6)
    p1.set_title('PSD 1')
    p2.imshow(psd2, extent=[f_row[0], f_row[-1], f_col[0], f_col[-1]], origin='lower', vmin=vmin, vmax=vmax, aspect = 'equal')
    p2.set_xlim(-0.6, 0.6)
    p2.set_ylim(-0.6, 0.6)
    p2.set_title('PSD 2')
    fig.colorbar(im1, ax=[p1, p2], orientation='vertical')
    plt.show()

def plot_transport_plan(T_2D, sigma = 20, downsample_factor = None, title = 'Transport Matrix', xlabel = 'Target Bin Index', ylabel = 'Source Bin Index'):
    
    # Sigma > 0 --> gaussian blurring, downsample_factor -->
    if sigma > 0 and downsample_factor == None:
        T = gaussian_filter(T_2D, sigma = sigma)
    elif sigma > 0 and downsample_factor != None:
        T = gaussian_filter(zoom(T_2D, downsample_factor, order=1),sigma)
    elif sigma < 0 and downsample_factor != None:
        T = zoom(T_2D, downsample_factor, order=1)
    else:
        T = T_2D.copy()

    plt.figure(figsize=(8, 8))
    plt.imshow(T, cmap='viridis')
    plt.colorbar(label='Transported Mass')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def block_plot_transport_plan(T_2D, nbr_blocks = 20, sigma = None, title = 'Block Transport Matrix', xlabel = 'Target Block Index', ylabel = 'Source Block Index'):

    T = T_2D.copy()
    shape = (T.shape[0]//nbr_blocks, nbr_blocks, T.shape[1]//nbr_blocks, nbr_blocks)
    T = T.reshape(shape).mean(axis=(1,3))

    if sigma != None and sigma > 0:
        T = gaussian_filter(T, sigma = sigma)

    plt.figure(figsize=(8, 8))
    plt.imshow(T, cmap='viridis')
    plt.colorbar(label='Transported Mass')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

    
########################################

def plot_interpolation_sequence_3d(itp_data, theta_seq, threshold=1e-8, z_spacing=0.1):

    _ , N, N_theta = itp_data.shape
    x = np.arange(N)
    y = np.arange(N)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Use a colormap with enough distinct colors for each step
    base_colors = cm.get_cmap('tab20', N_theta)
    colors = [base_colors(i) for i in range(N_theta)]

    for i, theta in enumerate(theta_seq):
        Z = itp_data[i]

        # Remove near-zero values
        Z_cleaned = np.where(Z > threshold, Z, 0)

        # Offset in Z-direction to separate layers
        Z_offset = Z_cleaned + i * z_spacing

        # Only plot where Z is positive
        mask = Z_cleaned > 0
        Z_masked = np.where(mask, Z_offset, np.nan)

        # Create constant facecolor array
        rgba_color = colors[i]
        facecolor_array = np.tile(rgba_color, (N, N, 1))

        ax.plot_surface(X, Y, Z_masked,
                        facecolors=facecolor_array,
                        rstride=1, cstride=1,
                        linewidth=0.2, antialiased=True, alpha=0.9)

    ax.set_title("Displacement Interpolation Sequence (3D)", fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Interpolated Intensity')
    ax.view_init(elev=30, azim=135)

    plt.tight_layout()
    plt.show()

def plot_interpolated_spectrums(mu_seq, f_row, f_col, time_indices):
    fig, plots = plt.subplots(3, 3, figsize=(14, 14))
    plots = plots.flatten()

    vmin = mu_seq.min()
    vmax = mu_seq.max()


