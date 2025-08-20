import ot 
import warnings
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import time

class OptimalTransport:

    def __init__(self,mu,nu,reg,space_dim):
        self.mu = mu
        self.nu = nu
        self.reg = reg
        self.space_dim = space_dim 
    
    def get_mu(self):
        return self.mu
        
    def compute_full_OT(self,return_cost = False):
        
        # Ensure no zero divisions
        epsilon = 1e-12
        mu_nonzero = self.mu + epsilon 
        nu_nonzero = self.nu + epsilon
        mu_nonzero /= np.sum(mu_nonzero)
        nu_nonzero /= np.sum(nu_nonzero)

        N = np.shape(self.mu)[0]

        axes = [np.linspace(0,N-1,N) for _ in range(self.space_dim)]
        grid = np.meshgrid(*axes, indexing = 'ij')
        coords = np.stack([g.ravel() for g in grid], axis=-1)

        source_coords = coords.copy()
        target_coords = coords.copy()

        C = ot.dist(source_coords, target_coords, metric = 'sqeuclidean')
        P, log = self._safe_sinkhorn(mu_nonzero.flatten(), nu_nonzero.flatten(), C, reg = self.reg, numItermax = 500000)

        log['threshold'] = None

        if return_cost:
            total_cost = np.sum(P * C)
            return P, log, total_cost
        else:
            return P, log

    def compute_thresholded_OT(self, threshold, return_cost = False):

        N = np.shape(self.mu)[0]
        spatial_axes = [np.linspace(0, N-1, N) for _ in range(self.space_dim)]
        spatial_grid = np.meshgrid(*spatial_axes, indexing='ij')
        positions = np.stack([g.flatten() for g in spatial_grid], axis=-1)

        mu_flat = self.mu.flatten()
        nu_flat = self.nu.flatten()


        mu_nonzero = mu_flat > threshold
        nu_nonzero = nu_flat > threshold

        mu_sparse = mu_flat[mu_nonzero]
        nu_sparse = nu_flat[nu_nonzero]

        mu_sparse /= mu_sparse.sum()
        nu_sparse /= nu_sparse.sum()

        X = positions[mu_nonzero]
        Y = positions[nu_nonzero]

        mu_reconstructed = np.zeros_like(self.mu.flatten())  # Flat array of same size
        mu_reconstructed[mu_nonzero] = mu_sparse             # Insert non-zero values

        # Reshape to original 2D shape
        mu_reconstructed = mu_reconstructed.reshape(self.mu.shape)

        # # Plot
        # self._plot_imshow(mu_reconstructed, name='mu_sparse')

        C = ot.dist(X, Y, metric='sqeuclidean')
        P, log = self._safe_sinkhorn(mu_sparse, nu_sparse, C, reg = self.reg, numItermax = 50000)  

        log['threshold'] = threshold

        N_total = mu_flat.shape[0]  
        P_full = np.zeros((N_total, N_total))
        P_full[np.ix_(mu_nonzero, nu_nonzero)] = P

        if return_cost:
            total_cost = np.sum(P * C)
            return P_full, log, total_cost
        else:
            return P_full, log
    
    def _plot_imshow(self, psd, name = 'spectrum'):
        plt.figure(figsize=(6, 5))
        plt.imshow(psd, extent=[-0.5, 0.5, 0.5, -0.5], origin='upper', aspect='auto', cmap='viridis')
        plt.colorbar(label='Normalized Power')
        plt.title(f'Power Spectral Density ({name})')
        plt.xlabel('Frequency axis 1')
        plt.ylabel('Frequency axis 2')
        plt.tight_layout()
        plt.show()
    
    def _safe_sinkhorn(self, mu, nu, C, reg, numItermax, try_standard = False):

        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        if try_standard:
            method = 'sinkhorn'
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')  # Turn warnings into exceptions
                    start_time = time.time()
                    print('Running standard sinkhorn')
                    P, log = ot.bregman.sinkhorn(mu, nu, C, log = True, reg = reg, numItermax = numItermax)
                    run_time = time.time() - start_time
                    run_log = {'niter': log['niter'], 'time': run_time, 'opt_method': method, 'timestamp': timestamp, 'cost_shape': C.shape}
            except(Warning, FloatingPointError, ValueError, RuntimeError) as e:
                print(f"[Warning] Standard Sinkhorn failed due to: {e}. Falling back to stabilized version.")
                method = 'sinkhorn_stabilized'
                start_time = time.time()
                P, log = ot.bregman.sinkhorn_stabilized(mu, nu, C, log = True, reg = reg, numItermax = numItermax)
                run_time = time.time() - start_time
                run_log = {'niter': log['n_iter'], 'time': run_time, 'opt_method': method, 'timestamp': timestamp, 'cost_shape': C.shape}
            
        else:
            # print('Running sinkhorn_log')
            method = 'sinkhorn_log'
            start_time = time.time()
            P, log = ot.bregman.sinkhorn_log(mu, nu, C, log = True, reg = reg, numItermax = numItermax)
            run_time = time.time() - start_time
            run_log = {'niter': log['niter'], 'time': run_time, 'opt_method': method, 'timestamp': timestamp, 'cost_shape': C.shape}
        
        return P, run_log


