# Standard library imports
import os

# Third party imports
import nibabel as nib
import numpy as np

# Relative imports
from skimage import measure
from scipy.ndimage import distance_transform_edt

# Local import
from plots import show_contours, show_distance_transform, visualize_expansion, visualize_randomness

def calculate_contours(mask):
    # Find contours in the segmented slice
    contours = measure.find_contours(mask)
    return contours

def expand_mask(mask_data, spacing, expansion=2.0):
    # Prepare output
    expanded_mask = np.zeros_like(mask_data)
    
    # Expand each 2D slice independently
    for i in range(0, mask_data.shape[2]):
        # Invert the mask: foreground = False, background = True
        inverted = ~mask_data[:, :, i].astype(bool)
    
        # Distance transform
        dist_map = distance_transform_edt(inverted, sampling=spacing[0:2])

        # Threshold at desired expansion distance
        expanded_mask[:, :, i] = (dist_map <= expansion) | mask_data[:, :, i].astype(bool)
    
    return expanded_mask

def random_mask(mask_data, spacing, expansion=2.0, randomness_level=0.5, seed=None):

    if seed is not None:
        np.random.seed(seed)

    random_mask = np.zeros_like(mask_data)

    for i in range(0, mask_data.shape[2]):
        binary_mask = mask_data[:, :, i].astype("bool")
        outside = distance_transform_edt(~binary_mask, sampling=spacing[0:2])
        inside = distance_transform_edt(binary_mask, sampling=spacing[0:2])
        sdf = outside.copy()
        sdf[binary_mask] = -inside[binary_mask]

        # Create a boolean mask for the expanded region
        expanded_mask = sdf <= expansion

        # Difference between expanded mask and binary mask
        shell = expanded_mask & ~binary_mask

        # Find the indices of the shell
        shell_indices = np.argwhere(shell)

        # Number of indices to select
        n_select = int(randomness_level * len(shell_indices))

        selected_indices = np.random.choice(len(shell_indices), n_select, replace=False)

        original_mask = binary_mask.copy()
        for idx in selected_indices:
            original_mask[tuple(shell_indices[idx])] = True
        random_mask[:, :, i] = original_mask
        
    return random_mask


if __name__ == "__main__":

    mask_volume = nib.load(os.environ["MASK_DATA_PATH"])
    mask_data = mask_volume.get_fdata()
    spacing = mask_volume.header.get_zooms()

    # expanded_mask_1 = expand_mask(mask_data, spacing, expansion=2.0)
    # visualize_expansion(expanded_mask_1, expansion=2.0)

    # expanded_mask_2 = expand_mask(mask_data, spacing, expansion=4.0)
    # visualize_expansion(expanded_mask_2, expansion=4.0)

    randomized_mask_1 = random_mask(mask_data, spacing, expansion=2.0, randomness_level=1.0, seed=42)
    visualize_randomness(randomized_mask_1, expansion=2.0)

    randomized_mask_2 = random_mask(mask_data, spacing, expansion=2.0, randomness_level=0.8, seed=50)
    visualize_randomness(randomized_mask_2, expansion=4.0)

    