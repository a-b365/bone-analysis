"""
Contour Adjustment Module

This module provides functionality for expanding and randomizing contours in 
medical image masks using distance transform techniques. It enables uniform
expansion and controlled randomization for data augmentation purposes.

Author: Amir Bhattarai
Date: 2025-05-23
Version: 1.0
"""

# Standard library imports
import os
from typing import List, Tuple, Optional

# Third-party imports
import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage import measure

# Local imports
from plots import (show_contours, show_distance_transform, 
                   visualize_expansion, visualize_randomness)


def calculate_contours(mask: np.ndarray, level: float = 0.5) -> List[np.ndarray]:
    """
    Calculate contours in a 2D binary mask.
    
    Args:
        mask (np.ndarray): 2D binary mask
        level (float): Contour level. Default: 0.5
        
    Returns:
        List[np.ndarray]: List of contour coordinate arrays
        
    Example:
        >>> mask = np.zeros((100, 100))
        >>> mask[25:75, 25:75] = 1
        >>> contours = calculate_contours(mask)
        >>> print(f"Found {len(contours)} contours")
    """
    contours = measure.find_contours(mask, level=level)
    return contours


def expand_mask(mask_data: np.ndarray, 
                spacing: Tuple[float, ...], 
                expansion: float = 2.0) -> np.ndarray:
    """
    Expand binary mask contours uniformly using distance transform.
    
    This function uses Euclidean distance transform to achieve uniform expansion
    in all directions, overcoming limitations of coordinate-based methods.
    
    Args:
        mask_data (np.ndarray): 3D binary mask data
        spacing (Tuple[float, ...]): Voxel spacing in mm (x, y, z)
        expansion (float): Expansion distance in mm. Default: 2.0
        
    Returns:
        np.ndarray: Expanded binary mask with same shape as input
        
    Raises:
        ValueError: If mask_data is not 3D or expansion is negative
        
    Example:
        >>> mask = np.random.randint(0, 2, (64, 64, 32))
        >>> spacing = (1.0, 1.0, 2.0)
        >>> expanded = expand_mask(mask, spacing, expansion=3.0)
    """
    if mask_data.ndim != 3:
        raise ValueError("Input mask must be 3D")
    
    if expansion < 0:
        raise ValueError("Expansion distance must be non-negative")
    
    # Initialize output array
    expanded_mask = np.zeros_like(mask_data, dtype=bool)
    
    # Process each 2D slice independently
    for slice_idx in range(mask_data.shape[2]):
        # Get current slice
        current_slice = mask_data[:, :, slice_idx]
        
        # Skip empty slices
        if not np.any(current_slice):
            continue
        
        # Invert mask: foreground becomes background for distance calculation
        inverted_mask = ~current_slice.astype(bool)
        
        # Calculate distance from background pixels to nearest foreground pixel
        distance_map = distance_transform_edt(
            inverted_mask, 
            sampling=spacing[:2]  # Use only x,y spacing for 2D
        )
        
        # Create expanded mask by thresholding distance map
        # Include original foreground and pixels within expansion distance
        expanded_slice = (distance_map <= expansion) | current_slice.astype(bool)
        expanded_mask[:, :, slice_idx] = expanded_slice
    
    return expanded_mask


def create_random_mask(mask_data: np.ndarray, 
                      spacing: Tuple[float, ...], 
                      expansion: float = 2.0,
                      randomness_level: float = 0.5, 
                      seed: Optional[int] = None) -> np.ndarray:
    """
    Create randomized mask variations using signed distance transform.
    
    This function creates controlled randomization by randomly selecting pixels
    within the expansion shell around the original mask boundary.
    
    Args:
        mask_data (np.ndarray): 3D binary mask data
        spacing (Tuple[float, ...]): Voxel spacing in mm (x, y, z)
        expansion (float): Maximum expansion distance in mm. Default: 2.0
        randomness_level (float): Fraction of shell pixels to include (0.0-1.0). Default: 0.5
        seed (Optional[int]): Random seed for reproducibility. Default: None
        
    Returns:
        np.ndarray: Randomized binary mask
        
    Raises:
        ValueError: If randomness_level is not between 0 and 1
        
    Example:
        >>> mask = np.random.randint(0, 2, (64, 64, 32))
        >>> spacing = (1.0, 1.0, 2.0)
        >>> random_mask = create_random_mask(mask, spacing, randomness_level=0.8, seed=42)
    """
    if not 0.0 <= randomness_level <= 1.0:
        raise ValueError("Randomness level must be between 0.0 and 1.0")
    
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize output array
    random_mask = np.zeros_like(mask_data)
    
    # Process each 2D slice independently
    for slice_idx in range(mask_data.shape[2]):
        current_slice = mask_data[:, :, slice_idx]
        
        # Skip empty slices
        if not np.any(current_slice):
            continue
        
        # Convert to boolean
        binary_mask = current_slice.astype(bool)
        
        # Calculate distance transforms for signed distance function
        # Distance from background to foreground (positive outside)
        outside_distance = distance_transform_edt(
            ~binary_mask, 
            sampling=spacing[:2]
        )
        
        # Distance from foreground to background (for inside calculation)
        inside_distance = distance_transform_edt(
            binary_mask, 
            sampling=spacing[:2]
        )
        
        # Create signed distance function
        # Positive values outside object, negative values inside
        signed_distance = outside_distance.copy()
        signed_distance[binary_mask] = -inside_distance[binary_mask]
        
        # Create expanded region mask
        expanded_region = signed_distance <= expansion
        
        # Find expansion shell (expanded region minus original mask)
        expansion_shell = expanded_region & ~binary_mask
        
        # Get indices of shell pixels
        shell_indices = np.argwhere(expansion_shell)
        
        if len(shell_indices) == 0:
            # No shell pixels found, keep original mask
            random_mask[:, :, slice_idx] = binary_mask
            continue
        
        # Calculate number of pixels to randomly select
        num_to_select = int(randomness_level * len(shell_indices))
        
        # Randomly select shell pixels
        if num_to_select > 0:
            selected_indices = np.random.choice(
                len(shell_indices), 
                size=num_to_select, 
                replace=False
            )
            
            # Start with original mask
            randomized_slice = binary_mask.copy()
            
            # Add randomly selected shell pixels
            for idx in selected_indices:
                x, y = shell_indices[idx]
                randomized_slice[x, y] = True
                
            random_mask[:, :, slice_idx] = randomized_slice
        else:
            # No pixels to add, keep original
            random_mask[:, :, slice_idx] = binary_mask
    
    return random_mask


def save_mask(mask_data: np.ndarray, 
              reference_volume: nib.Nifti1Image, 
              output_path: str) -> None:
    """
    Save processed mask to NIfTI file with original metadata.
    
    Args:
        mask_data (np.ndarray): Processed mask data
        reference_volume (nib.Nifti1Image): Reference volume for metadata
        output_path (str): Output file path
    """
    # Create new NIfTI image with original header
    new_image = nib.Nifti1Image(
        mask_data.astype(np.float32), 
        reference_volume.affine, 
        reference_volume.header
    )
    
    # Save to file
    nib.save(new_image, output_path)
    print(f"Saved processed mask to: {output_path}")


def main() -> None:
    """
    Main execution function for contour adjustment module.
    
    Demonstrates mask expansion and randomization functionality.
    """
    try:
        # Get mask data path from environment variable
        mask_path = os.environ.get("MASK_DATA_PATH")
        if not mask_path:
            raise ValueError("MASK_DATA_PATH environment variable not set")
        
        # Load medical volume
        print("Loading mask data...")
        mask_volume = nib.load(mask_path)
        mask_data = mask_volume.get_fdata()
        spacing = mask_volume.header.get_zooms()
        
        print(f"Loaded mask with shape: {mask_data.shape}")
        print(f"Voxel spacing: {spacing}")
        
        # Example 1: Mask expansion
        # expanded_mask_2mm = expand_mask(mask_data, spacing, expansion=2.0)
        # visualize_expansion(expanded_mask_2mm, expansion=2.0)
        
        # expanded_mask_4mm = expand_mask(mask_data, spacing, expansion=4.0)
        # visualize_expansion(expanded_mask_4mm, expansion=4.0)
        
        # Example 2: Randomized masks
        print("Creating randomized masks...")
        
        # High randomness (100% of shell pixels)
        randomized_mask_high = create_random_mask(
            mask_data, spacing, 
            expansion=2.0, 
            randomness_level=1.0, 
            seed=42
        )
        visualize_randomness(randomized_mask_high, expansion=2.0)
        
        # Medium randomness (80% of shell pixels)
        randomized_mask_medium = create_random_mask(
            mask_data, spacing, 
            expansion=2.0, 
            randomness_level=0.8, 
            seed=50
        )
        visualize_randomness(randomized_mask_medium, expansion=4.0)
        
        print("Contour adjustment complete!")
        
    except Exception as e:
        print(f"Error in contour adjustment: {str(e)}")
        raise


if __name__ == "__main__":
    main()