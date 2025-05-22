import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

def find_landmarks_optimized(mask_xyz, spacing=(1.0, 1.0, 1.0), top_percent=15, min_y_dist_vox=10):
    """
    Improved landmark detection on tibial plateau using skeleton + distance transform.

    Args:
        mask_xyz (ndarray): 3D binary mask in [x, y, z] format.
        spacing (tuple): voxel spacing (x, y, z).
        top_percent (float): top percent of slices in Z to consider as plateau.
        min_y_dist_vox (int): minimum distance from Y-midline to consider a point as medial/lateral.

    Returns:
        medial_xyz, lateral_xyz: landmark coordinates in [x, y, z] format.
    """
    # Transpose to [z, y, x]
    mask_zyx = np.transpose(mask_xyz, (2, 1, 0))
    spacing_zyx = spacing[::-1]

    # Step 1: Identify plateau region
    z_nonzero = np.any(mask_zyx, axis=(1, 2))
    z_indices = np.where(z_nonzero)[0]
    if len(z_indices) == 0:
        raise ValueError("Mask is empty.")
    z_cutoff = int(np.percentile(z_indices, 100 - top_percent))
    plateau_mask = np.zeros_like(mask_zyx, dtype=bool)
    plateau_mask[z_cutoff:] = mask_zyx[z_cutoff:]

    if np.sum(plateau_mask) == 0:
        raise ValueError("Empty plateau region. Increase top_percent.")

    # Step 2: Distance transform
    dist = distance_transform_edt(plateau_mask, sampling=spacing_zyx)

    # Step 3: Skeletonization (2D slice-wise)
    skeleton_points = []
    for z in range(z_cutoff, plateau_mask.shape[0]):
        slice_mask = plateau_mask[z]
        if np.any(slice_mask):
            skel2d = skeletonize(slice_mask)
            coords = np.argwhere(skel2d)  # [y, x]
            for y, x in coords:
                skeleton_points.append([z, y, x])
    skeleton_points = np.array(skeleton_points)

    if len(skeleton_points) < 2:
        raise ValueError("Too few skeleton points detected.")

    # Step 4: Combine with distance + y-spread + z-weight
    zyx_coords = skeleton_points
    values = dist[zyx_coords[:, 0], zyx_coords[:, 1], zyx_coords[:, 2]]

    # Filter near midline
    y_mid = np.median(zyx_coords[:, 1])
    y_dists = np.abs(zyx_coords[:, 1] - y_mid)
    keep_idx = np.where(y_dists > min_y_dist_vox)[0]
    if len(keep_idx) < 2:
        raise ValueError("Not enough points after Y-distance filtering.")
    zyx_coords = zyx_coords[keep_idx]
    values = values[keep_idx]
    y_dists = y_dists[keep_idx]

    # Weight score by Y distance and Z height (closer to plateau top = better)
    y_weight = 1.0 + (y_dists / (np.max(y_dists) + 1e-5))
    z_scores = 1.0 - (zyx_coords[:, 0] - z_cutoff) / (plateau_mask.shape[0] - z_cutoff + 1e-5)
    values = values * y_weight * z_scores

    # Step 5: Split medial/lateral
    medial_idx = np.where(zyx_coords[:, 1] < y_mid)[0]
    lateral_idx = np.where(zyx_coords[:, 1] >= y_mid)[0]

    if len(medial_idx) == 0 or len(lateral_idx) == 0:
        raise ValueError("Could not split into medial/lateral groups.")

    medial_zyx = zyx_coords[medial_idx[np.argmax(values[medial_idx])]]
    lateral_zyx = zyx_coords[lateral_idx[np.argmax(values[lateral_idx])]]

    # Convert back to [x, y, z]
    medial_xyz = medial_zyx[::-1]
    lateral_xyz = lateral_zyx[::-1]

    return medial_xyz, lateral_xyz
