"""Utility functions for network training and evaluation."""

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
import os
from functools import partial

def define_neighbors(num_scans_1d, num_neighbors_1d):
    """Return array of neighbors for a scan."""
    position_indices = np.arange(num_scans_1d**2).reshape(num_scans_1d,num_scans_1d)
    neighbors_exp = []
    num_neighbors_1d = 5
    neigh_onedir_exp = num_neighbors_1d//2
    centers = range(neigh_onedir_exp, num_scans_1d - neigh_onedir_exp)
    for i in centers:
        for j in centers:
            neighbors_exp.append(position_indices[i-neigh_onedir_exp:i+neigh_onedir_exp+1,
                                            j-neigh_onedir_exp:j+neigh_onedir_exp+1])
    neighbors_exp = np.array(neighbors_exp)
    return neighbors_exp.reshape((num_scans_1d-2*neigh_onedir_exp)**2,num_neighbors_1d**2)

def process_exp_path(file_path, n_scans, n_k, scan_step_size, d_x, num_neighbors_1d):
    """Process an experimental raw file."""
    try:
        data_original = np.fromfile(file_path+f"/scan_x{n_scans}_y{n_scans}.raw", '<f4')
        data = data_original.reshape((n_scans, n_scans, n_k+2, n_k))
        data = data[:, :, :n_k, :]
        # Adjust data to match simulated data structure.
        data = np.flip(data,-2)
        cbeds = data.reshape(n_scans**2, n_k, n_k)
        cbeds /= cbeds.max(axis=(1, 2), keepdims=True)
        # Create the grid
        scan_locs = np.arange(0, n_scans * scan_step_size, scan_step_size)
        grid_pts = np.round(scan_locs/d_x).astype(int)
        grid_pts_2d = np.stack(np.meshgrid(grid_pts,grid_pts))
        scan_pts = np.reshape(np.flip(grid_pts_2d, axis=0), (2,n_scans**2)).T
        # Create the neighbors
        neighbors_exp = define_neighbors(n_scans, num_neighbors_1d)
        return cbeds, neighbors_exp, scan_pts
    except Exception as e:
        print(f"[ERROR] Failed to process file: {file_path} | Reason: {e}")
        return 0

@partial(jit, static_argnames=['model'])#@jit(static_argnames=['model'])
def eval_exp_batch(model, model_state, inputs):
    """Evaluate model on a batch of data."""
    outputs = model.apply(model_state, inputs, training=False)
    return outputs