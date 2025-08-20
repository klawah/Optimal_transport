
import cvxpy as cp 
import numpy as np
import matplotlib as plt
import ot


def cvx_2d_OT(mu, nu):
    """
    Compute the optimal transport plan between source and target distributions using cvxpy optimisation.

    Parameters
    ----------
    - mu: source distribution
    - nu: target distribution 

    Returns
    -------
    - T_2D: optimal transport map in two dimensions (bin index domain)
    - cost: total cost of moving mass between source and target distributions according to the optimal transport plan
    """

    N_x, M_x = mu.shape
    N_y, M_y = nu.shape

    p = mu.flatten() 
    q = nu.flatten() 

    p = p / p.sum()
    q = q / q.sum()

    P = N_x * M_x
    Q = N_y * M_y

    C = compute_cost_matrix(N_x,N_y,M_x,M_y)
    T = cp.Variable((P,Q), nonneg = True)


    objective = cp.Minimize(cp.sum(cp.multiply(C, T)))
    constraints = [
        cp.sum(T, axis=1) == p,
        cp.sum(T, axis=0) == q
    ]
    
    prob = cp.Problem(objective,constraints)
    prob.solve()

    T_2D = T.value
    cost = prob.value

    return T_2D, cost

def sinkhorn_2d_OT(mu, nu, reg = 4e-1):
    """
    Compute the optimal transport plan between source and target distributions using cvxpy optimisation.

    Parameters
    ----------
    - mu: source distribution
    - nu: target distribution 
    - reg: regularisation factor used in the sinkhorn algorithm. Default as 4e-1

    Returns
    -------
    - T_2D: optimal transport map in two dimensions (bin index domain)
    - cost: total cost of moving mass between source and target distributions according to the optimal transport plan
    """
    
    # Ensure no zero divisions
    epsilon = 1e-12
    p = mu.flatten() + epsilon
    q = nu.flatten() + epsilon
    p /= p.sum()
    q /= q.sum()

    N_x, M_x = mu.shape
    N_y, M_y = nu.shape

    if (N_x != N_y) or (M_x != M_y): 
        raise ValueError("Source and target must have the same shape")

    N = N_x
    M = M_x
    x = np.arange(N)
    y = np.arange(M)
    X, Y = np.meshgrid(x, y)

    source_coords = np.stack([X.ravel(), Y.ravel()], axis=1)
    target_coords = np.stack([X.ravel(), Y.ravel()], axis=1)

    C = ot.dist(source_coords, target_coords, metric='euclidean')**2

    T_2D = ot.sinkhorn(p.ravel(), q.ravel(), C, reg)
    cost = np.sum(T_2D * C)

    return T_2D, cost

# ------ Internal ------ #

def compute_cost_matrix(N_x,N_y=None,M_x= None,M_y=None):

    if N_y == None:
        N_y = N_x
    if M_x == None:
        M_x = N_x
    if M_y == None:
        M_y = N_x
        
    source_grid = np.array([(i, j) for i in range(N_x) for j in range(M_x)])
    target_grid = np.array([(k, l) for k in range(N_y) for l in range(M_y)])
    
    return np.sum((source_grid[:, None, :] - target_grid[None, :, :]) ** 2, axis=2)

