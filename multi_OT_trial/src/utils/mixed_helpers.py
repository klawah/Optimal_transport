import numpy as np

def coarse_to_fine(coarse_coords,scaling_factor):
    fine_coords = []
    for coarse_coord in coarse_coords:
        for i in coarse_coord:
            fine_coord = []
            for j in scaling_factor:
                fine_coord.append(i**scaling_factor+j) 
            fine_coords.append(fine_coord)

    return fine_coords


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