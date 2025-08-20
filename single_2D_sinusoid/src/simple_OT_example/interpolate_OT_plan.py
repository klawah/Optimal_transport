import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter

def displacement_interpolation_1d(T,theta_seq):
    
    P,Q = T.shape

    # Ensure T is a square matrix
    if P != Q:
        raise ValueError("T should be square")

    # Create an index-based grid
    ggrid = np.arange(0,P).reshape(-1,1)                               # The -1 sets the number of rows that match the array given the number of columns is 1
    
    # Ensure the interpolation steps lie within limits [0,1]
    theta_seq = np.asarray(theta_seq).flatten()

    if np.max(theta_seq) > 1:
        raise ValueError("Maximum theta should be <= 1")
    if min(theta_seq)<0: 
        raise ValueError("Minimum theta should be >= 0")


    Theta = len(theta_seq)

    # Create a matrix of interpolation step distributions
    mu_seq = np.zeros((P,Theta))
    t_vec = T.flatten(order ='F')                                   # Flattened by putting columns together   
    for i in range(Theta):
        theta = theta_seq[i]
        point_mat = (1-theta) * ggrid + theta * ggrid.T
        point_quant = np.round(point_mat)
        point_quant = point_quant.flatten(order='F')

        J = np.zeros((P,P**2))
        for j in range(P):
            J[j, point_quant == j] = 1

        mu_seq[:,i] = J @ t_vec

        print(i)

    mu_seq_smooth = None

    if P > 10:
        mu_seq_smooth =np.zeros_like(mu_seq)
        for i in range(Theta):
            mu_seq_smooth[:,i] = gaussian_filter1d(mu_seq[:,i], sigma = 10)

    return mu_seq, mu_seq_smooth


def displacement_interpolation_2d(T, theta_seq, smoothing_sigma = None):
    
    N_x,M_x,N_y,M_y = T.shape

    # Ensure all dimensions of T are equal
    if N_x != M_x or N_x != N_y or N_x != M_y:
        raise ValueError("T should be a 4D matrix of equal dimensions")
    
    # Ensure the interpolation steps lie within limits [0,1]
    theta_seq = np.asarray(theta_seq).flatten()
    if np.max(theta_seq) > 1:
        raise ValueError("Maximum theta should be <= 1")
    if min(theta_seq)<0: 
        raise ValueError("Minimum theta should be >= 0")
    
    N = N_x
    N_theta = len(theta_seq)

    # Find all nonzero transport entries
    i, j, k, l = np.nonzero(T)
    mass = T[i, j, k, l]                    

    # Create a matrix of interpolation step distributions
    itp_seq = np.zeros((N,N,N_theta))
    for q, theta in enumerate(theta_seq):                                       # q = index, theta = value of theta_seq
        # Linear interpolation between bin positions
        i_interp = (1 - theta) * i + theta * k
        j_interp = (1 - theta) * j + theta * l

        # Round to nearest bin
        i_interp_round = np.clip(np.round(i_interp).astype(int), 0, N - 1)
        j_interp_round = np.clip(np.round(j_interp).astype(int), 0, N - 1)

        # Accumulate mass
        for idx in range(len(mass)):
            i_idx = i_interp_round[idx]
            j_idx = j_interp_round[idx]
            m = mass[idx]
            itp_seq[i_idx, j_idx, q] += m

    # Optional smoothing
    mu_seq_smooth = None
    if smoothing_sigma is not None:
        mu_seq_smooth = np.zeros_like(itp_seq)
        for q in range(N_theta):
            mu_seq_smooth[:, :, q] = gaussian_filter(itp_seq[:, :, q], sigma=smoothing_sigma)


    return itp_seq, mu_seq_smooth