import numpy as np
import matplotlib.pyplot as plt

def generate_multi_dimensional_sinusoid(space_dim,axis_dim,signal_parameters):
    """
    Simulates a complex sinusoidal signal according to the given parameters set in the specified space dimensions.
    The corresponding power spectrum (shifted to zero center) is computed using numpys fftn function.  

    Parameters
    ----------
    - Space_dim: The dimension of the signal domain
    - axis_dim: Length of each dimension axis (assumed equal in all dimensions)
    - signal_parameters: A list of dictionaries where an entry should specify amplitude ('amplitude'), center frequency ('fc') and damping ('damping') for a component of the complete signal.

    Returns
    -------
    - psd: The power spectrum (normalised to one) for the signal

    """
    
    # Define the spatial space 
    N = axis_dim
    spatial_axes = [np.linspace(0,N-1,N) for _ in range(space_dim)]
    spatial_grid = np.meshgrid(*spatial_axes, indexing = 'ij')

    # Define the signal
    signal = np.zeros(spatial_grid[0].shape, dtype=complex)
    
    for component_parameters in signal_parameters:
        amp = component_parameters['amplitude']
        fc = np.array(component_parameters['fc'])
        damp = component_parameters['damping']

        if len(fc) != space_dim:
            raise ValueError("For each center frequency, the dimension must match the space dimension")
        
        # Compute phase of sinusoid
        phase = np.zeros_like(spatial_grid[0])
        for d in range(space_dim):
            phase += fc[d] * spatial_grid[d]
        carrier = np.exp(2j * np.pi * phase)

        center = (N - 1) / 2
        grid_norm_sq = sum([(grid - center)**2 for grid in spatial_grid])
        envelope = np.exp(-(damp / N) * grid_norm_sq)

        signal += amp  * envelope * carrier

    

    # Transform signal to frequency domain
    spectrum = np.fft.fftshift(np.fft.fftn(signal))

    # Compute the magnitude and normalise 
    spectrum_mag = abs(spectrum) ** 2
    psd = spectrum_mag / np.sum(spectrum_mag)

    return psd


        

