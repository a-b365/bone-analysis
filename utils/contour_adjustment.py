# Standard library imports
import os

# Third party imports
import nibabel as nib
import numpy as np

# Relative imports
from skimage import measure
from scipy.ndimage import distance_transform_edt

# Local import
# from plots import show_mask, show_contours, show_distance_transform

def calculate_contours(mask):
    # Find contours in the segmented slice
    contours = measure.find_contours(mask, 0.8)
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
    
    return expanded_mask, dist_map

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
        
    return random_mask, inside, outside


if __name__ == "__main__":

    mask_volume = nib.load(os.environ["MASK_DATA_PATH"])
    mask_data = mask_volume.get_fdata()
    # mask_data = mask_data.transpose(2, 1, 0)
    spacing = mask_volume.header.get_zooms()

    expanded_mask_1, inside = expand_mask(mask_data, spacing, expansion=2.0)
    nib.save(nib.Nifti1Image(expanded_mask_1, affine=mask_volume.affine), os.environ["STORE_LOCATION"]+"left_knee_expanded_mask_1.nii.gz")
    
    # In the same way save the second expanded mask to .nii format 
    expanded_mask_2, inside = expand_mask(mask_data, spacing, expansion=4.0)
    nib.save(nib.Nifti1Image(expanded_mask_2, affine=mask_volume.affine), os.environ["STORE_LOCATION"] + "left_knee_expanded_mask_2.nii.gz")

    # # Calculate contours for slice number 110
    # contours = calculate_contours(expanded_mask_1[110]) + calculate_contours(mask_data[110])
    # show_contours(expanded_mask_1[110], contours)   # Visualize the congittour
    # show_distance_transform(inside) # Visualize the distance map

    randomized_mask_1, inside, outside = random_mask(mask_data, spacing, expansion=2.0, randomness_level=0.6, seed=42)
    nib.save(nib.Nifti1Image(randomized_mask_1, affine=mask_volume.affine), os.environ["STORE_LOCATION"]+"left_knee_randomized_mask_1.nii.gz")
    # contours = calculate_contours(randomized_mask_1[0]) + calculate_contours(mask_data[0])
    # show_contours(randomized_mask_1[0], contours)
    # show_distance_transform(inside)
    # show_distance_transform(outside)
    randomized_mask_2, inside, outside = random_mask(mask_data, spacing, expansion=4.0, randomness_level=0.4, seed=50)
    nib.save(nib.Nifti1Image(randomized_mask_2, affine=mask_volume.affine), os.environ["STORE_LOCATION"]+"left_knee_randomized_mask_2.nii.gz")


    