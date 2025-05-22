import nibabel as nib
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from plots import show_landmarks

def find_landmarks(tibia_3d, spacing, top_percent=15, min_x=10):

    # Identify plateau region
    z_nonzero = np.any(tibia_3d, axis=(0, 1))
    z_indices = np.where(z_nonzero)[0]
    z_cutoff = int(np.percentile(z_indices, 100 - top_percent))
    plateau_3d = np.zeros_like(tibia_3d, dtype=bool)
    plateau_3d[:, :, z_cutoff:] = tibia_3d[:, :, z_cutoff:]

    # Distance transform
    dist = distance_transform_edt(plateau_3d, sampling=spacing)

    # Skeletonization (2D slice-wise)
    skeleton_points = []
    for z in range(z_cutoff, plateau_3d.shape[2]):
        slice_mask = plateau_3d[:, :, z]
        if np.any(slice_mask):
            skel_2d = skeletonize(slice_mask)
            coords = np.argwhere(skel_2d)  # [x, y]
            for x, y in coords:
                skeleton_points.append([x, y, z])
    skeleton_points = np.array(skeleton_points)

    # Step 4: Combine with distance + x-spread + z-weight
    coords_3d = skeleton_points
    values = dist[coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2]]

    # Filter near X-midline (now used for medial-lateral)
    x_mid = np.median(coords_3d[:, 0])
    x_dists = np.abs(coords_3d[:, 0] - x_mid)
    keep_idx = np.where(x_dists > min_x)[0]
    coords_3d = coords_3d[keep_idx]
    values = values[keep_idx]
    x_dists = x_dists[keep_idx]

    # Weight score by X distance and Z height (closer to plateau top = better)
    x_weight = 1.0 + (x_dists / np.max(x_dists))
    z_scores = 1.0 - (coords_3d[:, 2] - z_cutoff) / (plateau_3d.shape[2] - z_cutoff)
    values = values * x_weight * z_scores

    # Step 5: Split medial/lateral using X-axis
    medial_idx = np.where(coords_3d[:, 0] < x_mid)[0]
    lateral_idx = np.where(coords_3d[:, 0] >= x_mid)[0]

    medial = coords_3d[medial_idx[np.argmax(values[medial_idx])]]
    lateral = coords_3d[lateral_idx[np.argmax(values[lateral_idx])]]

    return medial, lateral

if __name__ == "__main__":

    segmented_volume = nib.load("../results/3702_left_knee_mask_segmented.nii.gz")
    labels = segmented_volume.get_fdata()
    spacing = segmented_volume.header.get_zooms()
    tibia_3d = (labels == 1).astype(np.float32)
    medial, lateral = find_landmarks(tibia_3d, spacing)

