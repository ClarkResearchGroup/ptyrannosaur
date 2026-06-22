"""Load 4D-STEM data for training."""

import h5py
import numpy as np
import jax.numpy as jnp
import ptyrannosaur.engine.utils as utils
from tqdm.auto import tqdm

def batch_exp(norm_dps, neighbors, scan_pts):
    """Create batch of data for experiment."""
    center_id = neighbors.shape[1]//2
    input_dps = jnp.moveaxis(norm_dps[neighbors], 1, -1)
    return input_dps, scan_pts[neighbors[:,center_id]]

def loop_batch_exp(dps, neighbors, all_scan_pts, batch_size, model, model_state):
    """Run a batch of experimental data."""
    num_evals = neighbors.shape[0]
    num_batches = num_evals // batch_size
    assert(num_evals % batch_size == 0)
    outputs = []
    scan_pts = []
    for n in tqdm(range(num_batches), desc="Processing batches"):
        batch_neighbors = neighbors[n*batch_size:(n+1)*batch_size]
        batch_dps, batch_scan_pts = batch_exp(dps, batch_neighbors, all_scan_pts)
        batch_outputs = utils.eval_exp_batch(model, model_state, batch_dps)
        outputs.append(batch_outputs)
        scan_pts.append(batch_scan_pts)
    return jnp.concatenate(outputs), jnp.concatenate(scan_pts)