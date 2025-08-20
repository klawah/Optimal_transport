import utils.plots as plot
import numpy as np

def simulate_signals(N, fs, fc_1, fc_2, include_plots = False):
    """
    Simulate two sinusoidal signals in two dimentions and generate their respective power spectral densities.

    Parameters
    ----------
    - N: number of samples
    - fs: sampling frequency
    - fc_1: frequency components of the first signal
    - fc_2: frequency components of the second signal
    - include_plots: Optional plot of signals and spectrums. Default as False

    Returns
    -------
    - psd1: power spectral desity corresponding to the first signal 
    - psd2: power spectral density corresponding to the second signal
    - f_row: related frequency axis, vertical direction
    - f_col: related frequency axis, horizontal direction

    """
    # Convert frequency inputs to 2D arrays
    fc_1 = np.atleast_2d(fc_1)
    fc_2 = np.atleast_2d(fc_2)

    # Define axis ranges (time domain)
    t_row = np.arange(N) / fs
    t_col = np.arange(N) / fs

    # Initialise signals
    signal1 = np.zeros((N,N), dtype=complex)
    signal2 = np.zeros((N,N), dtype=complex)

    # Generate first signal
    for f in fc_1:
        if len(f) != 2:
            raise ValueError("Each frequency component in fc_1 must be a pair (f_row, f_col)")
        signal1 += create_2d_sinusoid(t_row, t_col, f[0], f[1])

    # Generate second signal
    for f in fc_2:
        if len(f) != 2:
            raise ValueError("Each frequency component in fc_2 must be a pair (f_row, f_col)")
        signal2 += create_2d_sinusoid(t_row, t_col, f[0], f[1])

    # FFT2 generated power spectrums
    psd1 = produce_2d_psd(signal1)
    psd2 = produce_2d_psd(signal2)

    # Generate zero-centered frequency axes
    f_row,f_col = generate_frequency_axes(N,fs) 

    # Optional plots
    if include_plots:
        plot.plot_src_and_target_signals(signal1,signal2,t_row,t_col)
        plot.plot_src_and_target_psds(psd1,psd2,f_row,f_col)

    return psd1, psd2, f_row, f_col

# ------ Internal ------ #

def create_2d_sinusoid(x_axis,y_axis,f_x,f_y):

    X, Y = np.meshgrid(x_axis, y_axis, indexing = 'ij')
    
    sinusoid = np.exp(2j*np.pi*(f_x*X+f_y*Y))

    return sinusoid

def produce_2d_psd(signal):
    psd = abs(np.fft.fftshift(np.fft.fft2(signal)))**2
    psd_norm = psd / np.sum(psd)
    return psd_norm

def generate_frequency_axes(N,fs):
    if type(N) != int:
        N = int(N)
        
    f_row = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    f_col = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    return f_row, f_col


def generate_custom_spectrum(shape=(256, 256), peak_params=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    N, M = shape
    fy = np.fft.fftshift(np.fft.fftfreq(N))
    fx = np.fft.fftshift(np.fft.fftfreq(M))
    FX, FY = np.meshgrid(fx, fy, indexing='ij')

    spectrum_magnitude = np.zeros_like(FX)

    # Default: manually defined Gaussian peaks
    if peak_params is None:
        peak_params = [
            {'center': (0.1, 0.2), 'sigma': 0.03, 'amplitude': 1.0},
            {'center': (-0.15, -0.1), 'sigma': 0.05, 'amplitude': 0.8},
            {'center': (0.25, -0.15), 'sigma': 0.02, 'amplitude': 0.6},
        ]


    # Sum multiple 2D Gaussians in frequency domain
    for peak in peak_params:
        cx, cy = peak['center']
        sigma = peak['sigma']
        amp = peak['amplitude']
        gaussian = amp * np.exp(-((FX - cx)**2 + (FY - cy)**2) / (2 * sigma**2))
        spectrum_magnitude += gaussian

    # Add random phase
    random_phase = np.exp(1j * 2 * np.pi * np.random.rand(*shape))
    spectrum_complex = spectrum_magnitude * random_phase

    # Inverse FFT to get spatial signal
    spatial_signal = np.fft.ifft2(np.fft.ifftshift(spectrum_complex)).real

    return spatial_signal, spectrum_magnitude, fx, fy