"""Stiching functions for recombining patches into the full object."""

import numpy as np
from scipy.fft import fftn, ifftn, fftfreq
from scipy import ndimage as ndi
from scipy import signal
import networkx as nx
import torch
import torch.nn as nn
from dataclasses import dataclass
import itertools

def grid_stitch(patches, scan_positions):
    """Stitch patches according to their exact scan positions faster."""
    N, H, W, L = patches.shape
    shift = H // 2
    # Offset positions once
    offset = np.min(scan_positions, axis=0) - shift
    scan_positions = scan_positions - offset
    # Compute output size
    max_pos = np.max(scan_positions, axis=0) + shift
    out_shape = (max_pos[0], max_pos[1], L)
    full_obj = np.zeros(out_shape, dtype=patches.dtype)
    count_obj = np.zeros(out_shape, dtype=patches.dtype)
    for s in range(N):
        x, y = scan_positions[s]
        xs = slice(x-shift, x+shift)
        ys = slice(y-shift, y+shift)
        full_obj[xs, ys] += patches[s]
        count_obj[xs, ys] += 1  # broadcasting instead of allocating count array
    return full_obj / np.maximum(count_obj, 1)

def learn_stitch(patches, scan_positions):
    N, H, W, L = patches.shape
    n_1d = int(np.sqrt(N))
    data = patches.reshape(n_1d, n_1d, H, H)
    pos = scan_positions.reshape(n_1d,n_1d,2)
    nr, nc = n_1d, n_1d
    # make shifts
    shift_xn_init = pos[:, 1:, ::-1] - pos[:, :-1, ::-1]
    shift_yn_init = pos[1:, :, ::-1] - pos[:-1, :, ::-1]
    shift_xn_guess = np.mean(shift_xn_init, axis=(0, 1))
    shift_yn_guess = np.mean(shift_yn_init, axis=(0, 1))
    # construct result
    res = masked_correlation_patch_stitching(data, shift_yn_guess, shift_xn_guess)
    return res.stitched_image, res.support, res.patch_pos

def masked_correlation_patch_stitching(data: np.ndarray, 
        shift_xn_guess: np.ndarray, shift_yn_guess: np.ndarray,
        correlation_peak_penalty_factor: float = 2.0, 
        shift_vector_mismatch_cutoff: float = 2.2,
        upsample_factor: int = 10,
        no_shift_weight: float = 0.3, 
        diagonal_weight: float = 0.5,
        learning_rates: tuple[float, int] | list[tuple[float, int]] = [(3e-1, 1000), (1e-1, 1000), (3e-2, 500)],
        support_threshold: float = 0.1):
    """Stitch image patches using cross correlation with masking.

    Requires an initial guess of the shift vectors between rows and columns of patches.
    Tries to find a set of patch positions that is most consistent with the cross correlation
    between neighboring pairs of patches. Some image patches may be excluded from the final stitched image.

    Parameters
    ----------
    data : ndarray
        nr x nc x p x p array containing the image patches to be stitched.
        There are nr x nc patches each of size p x p.  The 1st and 2nd axes of this array are assumed to be along
        the y and x directions, respectively.
    shift_xn_guess : np.ndarray
        Vector of length 2 in the form [dy, dx]. Initial guess of the shift vector in the x direction (the 1st axis of `data`).
    shift_yn_guess : np.ndarray
        Vector of length 2 in the form [dy, dx]. Initial guess of the shift vector in the y direction (the 2nd axis of `data`).
    correlation_peak_penalty_factor : float, optional
        Controls how much to penalize peaks in the cross-correlation that are farther from the initial guess.
        The greater this value, the greater the penalty. A value of 1 does not penalize at all.
    shift_vector_mismatch_cutoff : float, optional
        This value is a distance in units of pixels. Shift vectors that are more than this distance from the
        initial guess are excluded from the patch position refinement.
    upsample_factor : int, optional
        How much to upsample the cross correlation by when calculating more refined shift vectors.
    no_shift_weight : float, optional
        In the patch position refinement, when a x- or y-shift-vector value from cross correlation is not available,
        the initial guess is used instead. This controls how much such vectors are weighted.
    diagonal_weight : float, optional
        Weight for the shift vector between pairs of diagonally adjacent patches.
    learning_rates : tuple of (float, int) or list of tuple of (float, int), optional
        Contains one tuple or a list of tuples. Each tuple is (lr, niter) where lr is the learning rate for the Adam optimizer
        and niter is the number of iterations.
    support_threshold: float, optional
        Pixels in the final image with total support less than this threshold are masked out.

    Returns
    -------
    result : `CorrelationStitchResult` instance
        This is an object with the following attributes:
        
        stitched_image : ndarray
            Image stitched from the patches.  Pixels not covered by enough patches are set to NaN.
        patch_sum : ndarray
            Array summed from the patches, not normalized by the number of patches covering each pixel.
        support : ndarray
            Weighted total number of patches ("support") covering each pixel. A Hamming window is applied to the patches 
            before summing so the weight of a patch is smaller at the edges and corners of the patch.
        patch_pos : ndarray
            2xN array of optimized patch coordinates where patch_pos[0] corresponds to y-coordinates
        r_patch : ndarray
            Vectors of the row indices of the N patches
        c_patch : ndarray
            Vectors of the column indices of the N patches
        num_patches_excluded : int
            Number of patches not included in the final stitched image.
    
    Notes
    -----
    This stitching code is based on the phase cross correlation approach implemented 
    in scikit-image, but with several modifications to try to mitigate issues that
    arise when trying to stitch patches of limited field of view containing only a few
    atoms each with inconsistent contrast.

    The process is as follows: 1) Apply a Hamming window to all the patches.
    2) Compute the normalized cross-correlation between all pairs of patches
    that are neighbors in the x, y, diagonal, or anti-diagonal direction.
    This step is just like in phase cross correlation. 3) Find all peaks in
    the cross correlation arrays and penalize the peaks based on how far they
    are from what is expected given the initial shift vector guess. 4) For
    each pair of neighboring patches, pick the largest peak after the penalty
    factor and use the peak position as the shift vector between the pair of
    patches. 5) Filter the shift vectors and exclude all shift vectors that
    are more than `shift_vector_mismatch_cutoff` from the position expected
    given the initial shift vector guess. 6) Build a graph from the patches
    and remaining shift vectors and disconnect patches that are not well-
    connected to other patches. 7) Gather the patches and shift vectors of
    the largest connected component of the graph. 8) Refine the shift vectors
    to subpixel precision by upsampling the cross correlation arrays. 9) Use
    cost function optimization to find a set of patch positions that is most
    consistent with the refined shift vectors. 10) Shift the patches
    according to the patch positions. Calculate the final image as a weighted
    sum of these shifted patches.
    """
    
    nr, nc = data.shape[:2]
    shift_dn_guess = shift_yn_guess + shift_xn_guess
    shift_an_guess = shift_yn_guess - shift_xn_guess

    # window the data
    window = signal.windows.hamming(data.shape[2])
    windowed_data = data * window[np.newaxis, np.newaxis, :, np.newaxis]
    windowed_data *= window[np.newaxis, np.newaxis, np.newaxis, :]

    data_freq = fftn(windowed_data, axes=(2, 3))
    im_shape = np.array(data_freq.shape[2:])

    prod_yn = data_freq[:-1, :, :, :] * data_freq[1:, :].conj()
    prod_xn = data_freq[:, :-1, :, :] * data_freq[:, 1:, :, :].conj()
    prod_dn = data_freq[:-1, :-1, :, :] * data_freq[1:, 1:, :, :].conj()
    prod_an = data_freq[:-1, 1:, :, :] * data_freq[1:, :-1, :, :].conj()

    eps = np.finfo(prod_yn.real.dtype).eps

    prod_yn /= np.maximum(np.abs(prod_yn), 100*eps)
    prod_xn /= np.maximum(np.abs(prod_xn), 100*eps)
    prod_dn /= np.maximum(np.abs(prod_dn), 100*eps)
    prod_an /= np.maximum(np.abs(prod_an), 100*eps)

    cross_corr_y = ifftn(prod_yn, axes=(2,3))
    cross_corr_x = ifftn(prod_xn, axes=(2,3))
    cross_corr_d = ifftn(prod_dn, axes=(2,3))
    cross_corr_a = ifftn(prod_an, axes=(2,3))

    float_dtype = prod_yn.real.dtype

    def find_peaks_in_cross_corr_abs(cross_corr_abs, expected_shift=None):
        if expected_shift is not None:
            is_peak = np.ones(cross_corr_abs.shape, dtype=bool)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr != 0 or dc != 0:
                        is_peak &= cross_corr_abs >= np.roll(cross_corr_abs, (dr, dc), axis=(2, 3))

            ny = cross_corr_abs.shape[2]
            nx = cross_corr_abs.shape[3]
            xarr, yarr = np.meshgrid(np.arange(nx), np.arange(ny))
            y_exp, x_exp = expected_shift

            # because the cross correlation array is periodic
            dy = np.minimum(np.minimum(np.abs(yarr - y_exp), np.abs(yarr - y_exp - ny)), np.abs(yarr - y_exp + ny))
            dx = np.minimum(np.minimum(np.abs(xarr - x_exp), np.abs(xarr - x_exp - nx)), np.abs(xarr - x_exp + nx))
            dr = np.sqrt(dx**2 + dy**2)

            # penalize peaks that are too far from what is expected
            penalty = dr * correlation_peak_penalty_factor + 1

            arr = cross_corr_abs * is_peak / penalty[np.newaxis, np.newaxis, :, :]
        else:
            arr = cross_corr_abs

        flat_indices = np.argmax(np.reshape(np.abs(arr), (arr.shape[0], arr.shape[1], -1)), axis=2)
        maxima = np.unravel_index(flat_indices, im_shape)
        midpoint = np.array([np.trunc(axis_size / 2) for axis_size in im_shape])
        shift_n = np.stack(maxima).astype(float_dtype, copy=False)
        for d in range(2):
            shift_n[d][shift_n[d] > midpoint[d]] -= im_shape[d]
        return shift_n

    shift_yn = find_peaks_in_cross_corr_abs(np.abs(cross_corr_y), shift_yn_guess)
    shift_xn = find_peaks_in_cross_corr_abs(np.abs(cross_corr_x), shift_xn_guess)
    shift_dn = find_peaks_in_cross_corr_abs(np.abs(cross_corr_d), shift_dn_guess)
    shift_an = find_peaks_in_cross_corr_abs(np.abs(cross_corr_a), shift_an_guess)

    # how far is the computed image shift from what we would expect with perfect probe positions?
    mismatch_yn = np.linalg.norm(shift_yn - shift_yn_guess[:, np.newaxis, np.newaxis], axis=0)
    mismatch_xn = np.linalg.norm(shift_xn - shift_xn_guess[:, np.newaxis, np.newaxis], axis=0)
    mismatch_dn = np.linalg.norm(shift_dn - shift_dn_guess[:, np.newaxis, np.newaxis], axis=0)
    mismatch_an = np.linalg.norm(shift_an - shift_an_guess[:, np.newaxis, np.newaxis], axis=0)

    connect_yn = mismatch_yn < shift_vector_mismatch_cutoff
    connect_xn = mismatch_xn < shift_vector_mismatch_cutoff
    connect_dn = mismatch_dn < shift_vector_mismatch_cutoff
    connect_an = mismatch_an < shift_vector_mismatch_cutoff

    region_id_map, connect_yn, connect_xn, connect_dn, connect_an, \
        component_sizes = build_shift_vector_graph(connect_yn, connect_xn, connect_dn, connect_an, nr, nc)
    
    
    # refine shift vectors and produce a set of patch positions (time-consuming)
    upsample_factor = np.array(upsample_factor, dtype=float_dtype)
    upsampled_region_size = np.ceil(upsample_factor * 1.5)
    dftshift = np.trunc(upsampled_region_size / 2.0) # Center of output array at dftshift + 1

    def get_region_patch_pos(region_id):
        region_defn = region_id_map == region_id

        direction_suffix = 'xyda'
        same_region_xn = (region_id_map[:, :-1]   == region_id) & (region_id_map[:, 1:] == region_id)
        same_region_yn = (region_id_map[:-1, :]   == region_id) & (region_id_map[1:, :] == region_id)
        same_region_dn = (region_id_map[:-1, :-1] == region_id) & (region_id_map[1:, 1:] == region_id)
        same_region_an = (region_id_map[:-1, 1:]  == region_id) & (region_id_map[1:, :-1] == region_id)
        region_connect_xn = connect_xn & same_region_xn
        region_connect_yn = connect_yn & same_region_yn
        region_connect_dn = connect_dn & same_region_dn
        region_connect_an = connect_an & same_region_an

        region_disconnect_xn = np.logical_not(connect_xn) & same_region_xn
        region_disconnect_yn = np.logical_not(connect_yn) & same_region_yn

        all_shifts = {}

        for suffix, connect, shifts, products in zip(direction_suffix,
                    [region_connect_xn, region_connect_yn, region_connect_dn, region_connect_an],
                    [shift_xn, shift_yn, shift_dn, shift_an],
                    [prod_xn, prod_yn, prod_dn, prod_an]):
            sr, sc = np.nonzero(connect)
            region_shifts = shifts[:, sr, sc]
            refined_shifts = np.empty(region_shifts.shape, dtype=float)
            for pidx, (r, c) in enumerate(zip(sr, sc)):
                refined_shifts[:, pidx] = refine_shift(shifts[:, r, c], products[r, c, :, :], upsample_factor, upsampled_region_size, dftshift)
            all_shifts[suffix] = ShiftVectors(sr, sc, refined_shifts)

        bad_xn_indices = NoShiftVectors(*np.nonzero(region_disconnect_xn))
        bad_yn_indices = NoShiftVectors(*np.nonzero(region_disconnect_yn))

        r_patch, c_patch, patch_pos, _ = reconcile_shift_vectors(region_defn, all_shifts['x'], all_shifts['y'], all_shifts['d'], all_shifts['a'],
                                                            bad_xn_indices, bad_yn_indices,
                                                            shift_xn_guess, shift_yn_guess, 
                                                            no_shift_weight, diagonal_weight, learning_rates)
        return r_patch, c_patch, patch_pos
    
    r_patch, c_patch, patch_pos = get_region_patch_pos(0)

    stitched_region = stitch_patches(data, r_patch, c_patch, patch_pos)
    stitched_image = stitched_region.masked_image(support_threshold)
    num_patches_excluded = nr*nc - component_sizes[0].item()
    print(f'Image stitched with {num_patches_excluded} patch(es) excluded')

    return StitchResult(stitched_image,
        stitched_region.image, stitched_region.support, 
        patch_pos, r_patch, c_patch, num_patches_excluded)


@dataclass
class StitchResult:
    stitched_image: np.ndarray
    patch_sum: np.ndarray
    support: np.ndarray
    patch_pos: np.ndarray
    r_patch: np.ndarray
    c_patch: np.ndarray
    num_patches_excluded: int


def build_shift_vector_graph(connect_yn, connect_xn, connect_dn, connect_an, nr, nc):
    print("proportion of shift vectors within threshold:")
    for dir_str, arr in zip(['xn', 'yn', 'dn', 'an'], [connect_xn, connect_yn, connect_dn, connect_an]):
        prop = np.sum(arr) / np.size(arr)
        print(f'{dir_str}: {prop*100:.2f}%')

    def coord_to_idx(r, c):
        return r*nc + c

    def idx_to_coord(idx):
        return idx // nc, idx % nc

    # build initial graph
    graph = nx.Graph()
    graph.add_nodes_from(range(nr*nc))
    for r in range(nr):
        for c in range(nc-1):
            p1 = coord_to_idx(r, c)
            p2 = coord_to_idx(r, c+1)
            if connect_xn[r, c]:
                graph.add_edge(p1, p2)

    for r in range(nr-1):
        for c in range(nc):
            p1 = coord_to_idx(r, c)
            p2 = coord_to_idx(r+1, c)
            if connect_yn[r, c]:
                graph.add_edge(p1, p2)

    for r in range(nr-1):
        for c in range(nc-1):
            p1 = coord_to_idx(r, c)
            p2 = coord_to_idx(r+1, c+1)
            if connect_dn[r, c]:
                graph.add_edge(p1, p2)

    for r in range(nr-1):
        for c in range(nc-1):
            p1 = coord_to_idx(r, c+1)
            p2 = coord_to_idx(r+1, c)
            if connect_an[r, c]:
                graph.add_edge(p1, p2)


    def straight_and_diagonal_edges(nr, nc):
        for r in range(nr):
            for c in range(nc-1):
                p1 = coord_to_idx(r, c)
                p2 = coord_to_idx(r, c+1)
                yield r, c, p1, p2

        for r in range(nr-1):
            for c in range(nc):
                p1 = coord_to_idx(r, c)
                p2 = coord_to_idx(r+1, c)
                yield r, c, p1, p2

        for r in range(nr-1):
            for c in range(nc-1):
                p1 = coord_to_idx(r, c)
                p2 = coord_to_idx(r+1, c+1)
                yield r, c, p1, p2

        for r in range(nr-1):
            for c in range(nc-1):
                p1 = coord_to_idx(r, c+1)
                p2 = coord_to_idx(r+1, c)
                yield r, c, p1, p2

    # try to disconnect some patches that are not connected by enough shift vectors to the rest of the graph
    n_edge_remove = 0
    n_iterations = 0
    for _ in range(8):
        changed = False

        # remove articulation points
        comm_ids = np.full(nr*nc, -1, dtype=int)
        components = list(sorted(nx.biconnected_components(graph), key=len, reverse=True))
        large_id = 124*124
        for k, c in enumerate(components):
            for node in c:
                if comm_ids[node] == -1:
                    comm_ids[node] = k
                else: # articulation points belonging to multiple components shall be removed from both
                    comm_ids[node] = large_id
                    large_id += 1
        for r, c, p1, p2 in straight_and_diagonal_edges(nr, nc):
            if graph.has_edge(p1, p2) and (comm_ids[p1] != comm_ids[p2]):
                graph.remove_edge(p1, p2)
                n_edge_remove += 1
                changed = True

        # remove bridges
        bridge_list = list(nx.bridges(graph))
        if len(bridge_list) > 0:
            n_edge_remove += len(bridge_list)
            graph.remove_edges_from(bridge_list)
            changed = True
        if not changed:
            break
        else:
            n_iterations += 1
    print(f'{n_edge_remove} edge(s) removed over {n_iterations} iteration(s)')
    # print('graph changed after last iteration: ', changed)

    # update_connectivity_arrays
    for r in range(nr):
        for c in range(nc-1):
            p1 = coord_to_idx(r, c)
            p2 = coord_to_idx(r, c+1)
            connect_xn[r, c] = graph.has_edge(p1, p2)

    for r in range(nr-1):
        for c in range(nc):
            p1 = coord_to_idx(r, c)
            p2 = coord_to_idx(r+1, c)
            connect_yn[r, c] = graph.has_edge(p1, p2)

    for r in range(nr-1):
        for c in range(nc-1):
            p1 = coord_to_idx(r, c)
            p2 = coord_to_idx(r+1, c+1)
            connect_dn[r, c] = graph.has_edge(p1, p2)

    for r in range(nr-1):
        for c in range(nc-1):
            p1 = coord_to_idx(r, c+1)
            p2 = coord_to_idx(r+1, c)
            connect_an[r, c] = graph.has_edge(p1, p2)


    components = list(sorted(nx.connected_components(graph), key=len, reverse=True))
    component_sizes = np.array([len(c) for c in components])

    region_ids = np.full(nr*nc, -1, dtype=int)
    components = [np.array(list(c)) for c in components]
    for k, c in enumerate(components):
        region_ids[c] = k
        # print(k, len(c))
    # print(len(components), 'regions', 'min id', np.min(region_ids), 'max id', np.max(region_ids))

    region_id_map = np.reshape(region_ids, (nr, nc))

    return region_id_map, connect_yn, connect_xn, connect_dn, connect_an, component_sizes


def _upsampled_dft(data, upsampled_region_size, upsample_factor=1, axis_offsets=None):
    """
    Upsampled DFT by matrix multiplication.

    This function was taken from scikit-image:
    https://github.com/scikit-image/scikit-image/blob/v0.26.0/src/skimage/registration/_phase_cross_correlation.py

    This code is intended to provide the same result as if the following
    operations were performed:
        - Embed the array "data" in an array that is ``upsample_factor`` times
          larger in each dimension.  ifftshift to bring the center of the
          image to (1,1).
        - Take the FFT of the larger array.
        - Extract an ``[upsampled_region_size]`` region of the result, starting
          with the ``[axis_offsets+1]`` element.

    It achieves this result by computing the DFT in the output array without
    the need to zeropad. Much faster and memory efficient than the zero-padded
    FFT approach if ``upsampled_region_size`` is much smaller than
    ``data.size * upsample_factor``.

    Parameters
    ----------
    data : array
        The input data array (DFT of original data) to upsample.
    upsampled_region_size : integer or tuple of integers, optional
        The size of the region to be sampled.  If one integer is provided, it
        is duplicated up to the dimensionality of ``data``.
    upsample_factor : integer, optional
        The upsampling factor.  Defaults to 1.
    axis_offsets : tuple of integers, optional
        The offsets of the region to be sampled.  Defaults to None (uses
        image center)

    Returns
    -------
    output : ndarray
            The upsampled DFT of the specified region.
    """
    # if people pass in an integer, expand it to a list of equal-sized sections
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [
            upsampled_region_size,
        ] * data.ndim
    else:
        if len(upsampled_region_size) != data.ndim:
            raise ValueError(
                "shape of upsampled region sizes must be equal "
                "to input data's number of dimensions."
            )

    if axis_offsets is None:
        axis_offsets = [
            0,
        ] * data.ndim
    else:
        if len(axis_offsets) != data.ndim:
            raise ValueError(
                "number of axis offsets must be equal to input "
                "data's number of dimensions."
            )

    im2pi = 1j * 2 * np.pi

    dim_properties = list(zip(data.shape, upsampled_region_size, axis_offsets))

    for n_items, ups_size, ax_offset in dim_properties[::-1]:
        kernel = (np.arange(ups_size) - ax_offset)[:, None] * fftfreq(
            n_items, upsample_factor
        )
        kernel = np.exp(-im2pi * kernel)
        # use kernel with same precision as the data
        kernel = kernel.astype(data.dtype, copy=False)

        # Equivalent to:
        #   data[i, j, k] = kernel[i, :] @ data[j, k].T
        data = np.tensordot(kernel, data, axes=(1, -1))
    return data


def refine_shift(shift, image_product, upsample_factor, upsampled_region_size, dftshift):
    '''
    Given a shift vector and the correlation between two image patches, refine the shift to subpixel precision.

    This function was adapted from scikit-image:
    https://github.com/scikit-image/scikit-image/blob/v0.26.0/src/skimage/registration/_phase_cross_correlation.py
    '''

    shift = np.round(shift * upsample_factor) / upsample_factor

    # Matrix multiply DFT around the current shift estimate
    sample_region_offset = dftshift - shift * upsample_factor
    cross_correlation = _upsampled_dft(
        image_product.conj(),
        upsampled_region_size,
        upsample_factor,
        sample_region_offset,
    ).conj()

    # Locate maximum and map back to original pixel grid
    maxima = np.unravel_index(
        np.argmax(np.abs(cross_correlation)), cross_correlation.shape
    )

    float_dtype = image_product.real.dtype
    maxima = np.stack(maxima).astype(float_dtype, copy=False)
    maxima -= dftshift

    shift += maxima / upsample_factor

    return shift


@dataclass
class ShiftVectors:
    r: np.ndarray
    c: np.ndarray
    shifts: np.ndarray

@dataclass
class NoShiftVectors:
    r: np.ndarray
    c: np.ndarray

def reconcile_shift_vectors(region_defn: np.ndarray, 
        xn: ShiftVectors, yn: ShiftVectors, 
        dn: ShiftVectors, an: ShiftVectors,
        bad_xn: NoShiftVectors, bad_yn: NoShiftVectors,
        shift_xn_init: np.ndarray, shift_yn_init: np.ndarray,
        no_shift_weight: float = 0.3, diagonal_weight: float = 0.5,
        learning_rates: tuple[float, int] | list[tuple[float, int]] = [(3e-1, 1000), (1e-1, 1000), (3e-2, 500)]):
    """
    Returns
    -------
    r_patch : ndarray
    c_patch : ndarray
        Vectors of the row and column indices of the N patches
    optimized_pos : ndarray
        2xN array of optimized patch coordinates where optimized_pos[0] corresponds to y-coordinates
    """
    
    npos = np.sum(region_defn)

    # two kinds of patch indices:
    #   1) 2D index (row, column) into original array of patches
    #   2) 1D index into list of patches for this region
    r_patch, c_patch = np.nonzero(region_defn)
    idx_2D_to_1D = {}
    for k, (r, c) in enumerate(zip(r_patch, c_patch)):
        idx_2D_to_1D[(r, c)] = k

    xn_indices = list(itertools.chain(zip(xn.r, xn.c), zip(bad_xn.r, bad_xn.c)))
    yn_indices = list(itertools.chain(zip(yn.r, yn.c), zip(bad_yn.r, bad_yn.c)))

    idx1_xn = np.array([idx_2D_to_1D[r  , c]   for r, c in xn_indices])
    idx2_xn = np.array([idx_2D_to_1D[r  , c+1] for r, c in xn_indices])
    idx1_yn = np.array([idx_2D_to_1D[r  , c]   for r, c in yn_indices])
    idx2_yn = np.array([idx_2D_to_1D[r+1, c]   for r, c in yn_indices])
    idx1_dn = np.array([idx_2D_to_1D[r  , c]   for r, c in zip(dn.r, dn.c)])
    idx2_dn = np.array([idx_2D_to_1D[r+1, c+1] for r, c in zip(dn.r, dn.c)])
    idx1_an = np.array([idx_2D_to_1D[r  , c+1] for r, c in zip(an.r, an.c)])
    idx2_an = np.array([idx_2D_to_1D[r+1, c]   for r, c in zip(an.r, an.c)])

    n_xn_good = len(xn.r)
    n_xn_bad = len(bad_xn.r)
    n_yn_good = len(yn.r)
    n_yn_bad = len(bad_yn.r)

    fixed_pos_weight = 1/npos

    xn_shifts = np.zeros((2, n_xn_good + n_xn_bad), dtype=float)
    xn_weights = np.ones((2, n_xn_good + n_xn_bad), dtype=float)
    xn_shifts[:, :n_xn_good] = xn.shifts
    xn_shifts[:, n_xn_good:] = shift_xn_init[:, np.newaxis]
    xn_weights[:, n_xn_good:] = no_shift_weight

    yn_shifts = np.zeros((2, n_yn_good + n_yn_bad), dtype=float)
    yn_weights = np.zeros((2, n_yn_good + n_yn_bad), dtype=float)
    yn_shifts[:, :n_yn_good] = yn.shifts
    yn_shifts[:, n_yn_good:] = shift_yn_init[:, np.newaxis]
    yn_weights[:, n_yn_good:] = no_shift_weight

    # pos[0, :] are y-coordinates
    # pos[1, :] are x-coordinates
    pos = r_patch[np.newaxis, :] * shift_yn_init[:, np.newaxis] + \
        c_patch[np.newaxis, :] * shift_xn_init[:, np.newaxis]
    avg_pos = np.mean(pos, axis=1)
    pos -= avg_pos[:, np.newaxis]

    idx1_xn_t = torch.LongTensor(idx1_xn)
    idx2_xn_t = torch.LongTensor(idx2_xn)
    idx1_yn_t = torch.LongTensor(idx1_yn)
    idx2_yn_t = torch.LongTensor(idx2_yn)
    idx1_dn_t = torch.LongTensor(idx1_dn)
    idx2_dn_t = torch.LongTensor(idx2_dn)
    idx1_an_t = torch.LongTensor(idx1_an)
    idx2_an_t = torch.LongTensor(idx2_an)
    xn_weights_t = torch.from_numpy(xn_weights)
    yn_weights_t = torch.from_numpy(yn_weights)

    opt_targets = (torch.from_numpy(pos[:, 0]),
                torch.from_numpy(xn_shifts),
                torch.from_numpy(yn_shifts),
                torch.from_numpy(dn.shifts),
                torch.from_numpy(an.shifts))

    class PatchPositions(nn.Module):
        def __init__(self, pos):
            super().__init__()
            self.pos = nn.parameter.Parameter(torch.from_numpy(pos).clone())

        def forward(self):
            loss_shifts_xn = self.pos[:, idx2_xn_t] - self.pos[:, idx1_xn_t]
            loss_shifts_yn = self.pos[:, idx2_yn_t] - self.pos[:, idx1_yn_t]
            loss_shifts_dn = self.pos[:, idx2_dn_t] - self.pos[:, idx1_dn_t]
            loss_shifts_an = self.pos[:, idx2_an_t] - self.pos[:, idx1_an_t]
            first_pos = self.pos[:, 0]
            return (first_pos, loss_shifts_xn, loss_shifts_yn, loss_shifts_dn, loss_shifts_an)

    class CustomLoss(nn.Module):
        def __init__(self):
            super(CustomLoss, self).__init__()

        def forward(self, inputs, targets):
            first_pos_loss = torch.sum(torch.square(inputs[0] - targets[0])) * fixed_pos_weight
            loss_x = torch.sum(torch.square(inputs[1] - targets[1]) * xn_weights_t)
            loss_y = torch.sum(torch.square(inputs[2] - targets[2]) * yn_weights_t)
            loss_d = torch.sum(torch.square(inputs[3] - targets[3])) * diagonal_weight
            loss_a = torch.sum(torch.square(inputs[4] - targets[4])) * diagonal_weight
            return first_pos_loss + loss_x + loss_y + loss_d + loss_a
        
    model = PatchPositions(pos)
    loss_fn = CustomLoss()

    loss_values = []

    if isinstance(learning_rates, tuple):
        learning_rates = [learning_rates]
    for lr, niter in learning_rates:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
        for epoch in range(niter):
            # forward pass
            outputs = model()
            loss = loss_fn(outputs, opt_targets)
            loss_values.append(loss.item())

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    optimized_pos = model.pos.detach().cpu().numpy()
    return r_patch, c_patch, optimized_pos, loss_values


def perfect_grid_stitching(data: np.ndarray, 
        shift_xn_guess: np.ndarray, shift_yn_guess: np.ndarray):
    nr, nc = data.shape[:2]

    r_patch = np.arange(nr).repeat(nc)
    c_patch = np.tile(np.arange(nc), nr)
    pos = r_patch[np.newaxis, :] * shift_yn_guess[:, np.newaxis] + \
            c_patch[np.newaxis, :] * shift_xn_guess[:, np.newaxis]
    stitched_region = stitch_patches(data, r_patch, c_patch, pos)
    stitched_image = stitched_region.masked_image()

    return StitchResult(stitched_image,
        stitched_region.image, stitched_region.support, 
        pos, r_patch, c_patch, 0)


@dataclass
class StitchedPatches:
    image: np.ndarray
    support: np.ndarray
    positions: np.ndarray
    r_patch: np.ndarray
    c_patch: np.ndarray

    def masked_image(self, support_threshold=0.1):
        masked_support = self.support.copy()
        masked_support[masked_support < support_threshold] = np.nan
        return self.image / masked_support
    

def stitch_patches(data, r_patch, c_patch, patch_pos):
    """
    Parameters
    ----------
    data : ndarray
        nr x nc x p x p array of image patches
    r_patch : ndarray
    c_patch : ndarray
        Vectors of the row and column indices of the N patches
    patch_pos : ndarray
        2xN array of patch coordinates where optimized_pos[0] corresponds to y-coordinates
    """
    padded_data = np.pad(data, ((0,0), (0,0), (1,1), (1,1)))
    im_shape = data.shape[2:]
    padded_support = np.pad(np.ones(im_shape, dtype=float), 1)
    padded_shape = np.array(padded_support.shape)

    window = signal.windows.hamming(padded_data.shape[2])
    window2 = window[:, np.newaxis] * window[np.newaxis, :]
    windowed_padded_support = padded_support * window2
    windowed_padded_data = padded_data * window2[np.newaxis, np.newaxis, :, :]

    patch_pos -= np.min(patch_pos, axis=1)[:, np.newaxis]
    region_max = np.ceil(np.max(patch_pos, axis=1)).astype(int) + padded_shape

    # positions of the top left corners of the patches
    # +1 to account for padding
    padded_pos = patch_pos + 1

    npos = len(r_patch)

    canvas = np.zeros(region_max, dtype=float)
    canvas_support = np.zeros(region_max, dtype=float)

    for k in range(npos):
        curr_pos = patch_pos[:, k]
        rounded_pos = np.round(curr_pos)
        subpx_shift = curr_pos - rounded_pos
        shift_patch = ndi.shift(windowed_padded_data[r_patch[k], c_patch[k]], subpx_shift, order=1)
        shift_support = ndi.shift(windowed_padded_support, subpx_shift, order=1)

        r, c = rounded_pos.astype(int)
        canvas[r:r+padded_shape[0], c:c+padded_shape[1]] += shift_patch
        canvas_support[r:r+padded_shape[0], c:c+padded_shape[1]] += shift_support

    return StitchedPatches(canvas, canvas_support, padded_pos, r_patch, c_patch)