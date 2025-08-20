import numpy as np
import sys
import os
import ot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))  # adds 'src' parent
import src.utils.signal_simulation as signal
import src.utils.optimal_transport_computation as opt_t
import src.utils.OT_interpolation as itp
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.image as mpimg
import datetime as dt
import os
import json
import hashlib



def main():

    # Space dimension
    D = 2
    
    # Axis dimension (equal for all axes)
    N = 100

    # Define regularisation factor
    reg = 3e-1

    # Choose if a sparce or a dense optimisation should be run
    run_full = False

    # Interpolation parameters
    N_theta = 21

    # Choose a smoothing factor for interpolation
    if N > 10:
        sigma = 5
    else:
        sigma = 0

    run_setup = {'space_dim': D, 'axis_dim': N, 'reg': reg, 'run_full': run_full, 'N_theta': N_theta, 'smoothing_sigma': sigma}
    
    # Source and target signal parameters
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
    

    # Convert parameters to hash code for comparison later
    mu_hash = parameter_hash(mu_parameters)
    nu_hash = parameter_hash(nu_parameters)

    run_setup['mu_hash'] = mu_hash
    run_setup['nu_hash'] = nu_hash


    ####### Signal ######

    # Compute the spectral densities corresponding to the source and target signal parameters
    mu = signal.generate_multi_dimensional_sinusoid(D, N, mu_parameters)
    nu = signal.generate_multi_dimensional_sinusoid(D, N, nu_parameters)

    # plot_imshow(mu)

    ###### Optimal Transport and Interpolation ######

    run_setup['run_name'] = f"run_N{N}_D{D}"
    
    P, logs = run(mu,nu,run_setup, rerun = True)

    coords = coord_list(N,D)
    time = logs['time']
    cost = get_cost(P,coords)
    print('Time elapsed: ', time, '. Total cost: ',cost)


    input("Press Enter to exit...")


def coord_list(N,D):
    axes = [np.linspace(0,N-1,N) for _ in range(D)] # Define each axis of the grid
    grid = np.meshgrid(*axes, indexing = 'ij') # Create a meshgrid object with the given axis length
    coords = np.stack([g.ravel() for g in grid], axis=-1) # List of coordinates in the grid (row major order)
    return coords.astype(int)

def get_cost(P, source_coords, target_coords = None, cost_metric='sqeuclidean'):
        if target_coords is None:
            cost_matrix =  ot.dist(source_coords,source_coords,cost_metric)
        
        else:
            cost_matrix = ot.dist(source_coords,target_coords,cost_metric)

        return np.sum(P * cost_matrix)

import numpy as np
import matplotlib.pyplot as plt

def plot_transport_distributions(P, grid_shape):
    """
    Visualize where mass starts and ends based on the transport plan P.

    Parameters
    ----------
    P : ndarray, shape (N^2, N^2)
        Transport plan matrix
    grid_shape : tuple
        Shape of the spatial grid (N, N)
    """
    source_mass = np.sum(P, axis=1).reshape(grid_shape)  # Source marginal
    target_mass = np.sum(P, axis=0).reshape(grid_shape)  # Target marginal

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(
        source_mass,
        extent=[-0.5, 0.5, 0.5, -0.5],
        origin='upper',
        aspect='auto',
        cmap='viridis'
    )
    axes[0].set_title('Source Mass Distribution')
    axes[0].set_xlabel('Frequency axis 1')
    axes[0].set_ylabel('Frequency axis 2')
    plt.colorbar(im0, ax=axes[0], label='Mass')

    im1 = axes[1].imshow(
        target_mass,
        extent=[-0.5, 0.5, 0.5, -0.5],
        origin='upper',
        aspect='auto',
        cmap='viridis'
    )
    axes[1].set_title('Target Mass Distribution')
    axes[1].set_xlabel('Frequency axis 1')
    axes[1].set_ylabel('Frequency axis 2')
    plt.colorbar(im1, ax=axes[1], label='Mass')

    plt.tight_layout()
    plt.show()


def plot_interpolations_2d(psd_seq, theta_seq, N):
    N_theta = psd_seq.shape[1]
    nrows = 3
    ncols = int(np.ceil(N_theta / nrows))
    
    vmax = np.percentile(psd_seq, 99.9)
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
    plt.show(block=False)

    return fig



def plot_overlayed_interpolations_2d(psd_seq, N):
    combined = np.sum(psd_seq, axis=1).reshape(N, N)
    extent = [-0.5, 0.5, 0.5, -0.5]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(combined, origin='upper', extent=extent, cmap='viridis', aspect='equal')  
    ax.set_title("Overlayed OT Interpolations (Sum)")
    plt.colorbar(im, ax=ax, label='Normalized Power')
    ax.set_xlabel('Frequency axis 1')
    ax.set_ylabel('Frequency axis 2')
    plt.tight_layout()
    plt.show(block=False)

    return fig




def plot_imshow(psd, name = 'spectrum'):

    plt.figure(figsize=(6, 5))
    plt.imshow(psd, extent=[-0.5, 0.5, 0.5, -0.5], origin='upper', aspect='auto', cmap='viridis')
    plt.colorbar(label='Normalized Power')
    plt.title(f'Power Spectral Density ({name})')
    plt.xlabel('Frequency axis 1')
    plt.ylabel('Frequency axis 2')
    plt.tight_layout()
    plt.show()


def run(mu, nu, run_setup, rerun = True):

    D, reg = run_setup['space_dim'], run_setup['reg']

    result_path = 'multi_OT_trial/src/multi_OT/result_data'
    base_dir = os.path.join(result_path,run_setup['run_name'])

    OT = opt_t.OptimalTransport(mu,nu,reg,D)

    if os.path.exists(base_dir):
        if rerun:
            print('The space has been explored but a rerun is ordered. Running optimisation...')
            P,logs = compute_all_and_save(OT,run_setup,base_dir)
        else:
            print('The space has been explored - chaecking old results...')
            keys =['mu_hash', 'nu_hash','run_full','reg']
            found, folders = check_for_match(base_dir,run_setup,keys)
            if found:
                print('Match found for the transport plan. Checking the interpolation...')
                itp_found, itp_folders = check_for_match(base_dir,run_setup,['smoothing_sigma'],folders)
                
                if itp_found:
                    print('A full match was found in ', itp_folders[0],'. No comutations needed.')
                    P = np.load(itp_folders[0]/"transport_plan.npy")
                    with open(itp_folders[0] / "logs.json", "r") as f:
                        logs = json.load(f)   
                    show_old_result(folders[0], D)
                else:
                    print('The interpolation did not match.')
                    P, logs = compute_interpolation_and_save(folders[0],run_setup,base_dir)
            else:
                ('No matching OT log found. Running optimisation...')
                P,logs = compute_all_and_save(OT,run_setup,base_dir)
    else:
        print('Running the optimisation...')
        P, logs = compute_all_and_save(OT,run_setup,base_dir)

    return P, logs

def compute_interpolation_and_save(folder,run_setup,base_dir):
    N,D = run_setup['axis_dim'], run_setup['space_dim']
    N_theta = run_setup['N_theta']
    sigma = run_setup['smoothing_sigma']
    theta_seq = np.linspace(0,1,N_theta)
    P = np.load(folder/"transport_plan.npy")
    with open(folder/"logs.json", "r") as f:
        full_run_log = json.load(f)   
    
    full_run_log['N_theta'] = N_theta

    print('Computing interpolation for existing transport map...')

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    itp_seq, sigma_used = itp.interpolate_multi_dimensional_OT(P, D, theta_seq, sigma)

    full_run_log['smoothing_sigma'] = sigma_used

    print('Done with interpolation. Saving results.')

    save_dir = os.path.join(base_dir,timestamp)
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "transport_plan.npy"), P)
    np.save(os.path.join(save_dir, "interpolations.npy"), itp_seq)  
    with open(os.path.join(save_dir, "logs.json"), "w") as f:
        json.dump(full_run_log, f, indent=2)
    if D == 2:
        figure1 = plot_interpolations_2d(itp_seq,theta_seq,N)
        figure2 = plot_overlayed_interpolations_2d(itp_seq,N)
        figure1.savefig(os.path.join(save_dir, "interpolation_plot.png"))

    return P, full_run_log


def compute_all_and_save(OT,run_setup,base_dir):
    N,D = run_setup['axis_dim'], run_setup['space_dim']
    run_full = run_setup['run_full']
    N_theta = run_setup['N_theta']
    sigma = run_setup['smoothing_sigma']
    theta_seq = np.linspace(0,1,N_theta)

    if run_full:
        P, log = OT.compute_full_OT()
    else:
        min_threshold = 1e-3 / (N**D)  
        adaptive_threshold = _compute_adaptive_threshold(OT.get_mu())
        threshold = max(min_threshold,adaptive_threshold)    
        P, log, total_cost = OT.compute_thresholded_OT(threshold, return_cost = True)
    P = P.reshape((N**D,) * 2)

    full_run_log = {**run_setup,**log}

    full_run_log['total_cost'] = total_cost

    timestamp = full_run_log['timestamp']

    print('Done with optimisation. Starting interpolation...')

    itp_seq, sigma_used = itp.interpolate_multi_dimensional_OT(P, D, theta_seq, sigma)

    full_run_log['smoothing_sigma'] = sigma_used

    print('Done with interpolation. Saving results.')

    save_dir = os.path.join(base_dir,timestamp)
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "transport_plan.npy"), P)
    np.save(os.path.join(save_dir, "interpolations.npy"), itp_seq)  
    with open(os.path.join(save_dir, "logs.json"), "w") as f:
        json.dump(full_run_log, f, indent=2)
    if D == 2:
        figure1 = plot_interpolations_2d(itp_seq,theta_seq,N)
        figure2 = plot_overlayed_interpolations_2d(itp_seq,N)
        figure1.savefig(os.path.join(save_dir, "interpolation_plot.png"))

    return P, full_run_log


def parameter_hash(param):
    # Sort keys to get consistent JSON, then encode
    json_str = json.dumps(param, sort_keys=True)
    return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

def check_for_match(base_dir,run_setup,keys,folders_to_check = None):
    config_dir = Path(base_dir)
    found = False

    matching_folders = []
    for subfolder in config_dir.iterdir():
        if subfolder.is_dir():
            if folders_to_check == None or subfolder in folders_to_check:
                log_path = subfolder / "logs.json"
                if log_path.exists():
                    try:
                        with open(log_path, 'r') as f:
                            old_logs = json.load(f)

                        # Comparison
                        all_match = True
                        for key in keys:
                            if not old_logs.get(key) == run_setup[key]:
                                all_match = False
                        if all_match:
                            print(f"Match found in {subfolder}")
                            found = True
                            matching_folders.append(subfolder)

                    except json.JSONDecodeError:
                        print(f"[Warning] Could not decode JSON in {log_path}. Skipping.")
                    except Exception as e:
                        print(f"[Warning] Error while reading {log_path}: {e}")
        

    return found, matching_folders

def show_old_result(folder,D):
    if(D == 2):
        fig_path = folder / "interpolation_plot.png"  
        img = mpimg.imread(fig_path)

        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title("Interpolated OT Result")
        plt.show()
    else:
        print('No plot for higher dimensions.')

def _compute_adaptive_threshold(P, top_percentile=80, min_ratio=0.05):
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
