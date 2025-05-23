"""
Landmark Detection Module

This module provides functionality for detecting anatomical landmarks in tibial
plateau region using skeletonization and distance transform techniques.

Author: Amir Bhattarai
Date: 2025-05-23
Version: 1.0
"""

# Standard library imports
import os
from typing import Tuple, Optional

# Third-party imports
import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

# Local imports
from plots import show_landmarks


def find_landmarks(tibia_3d: np.ndarray, 
                   spacing: Tuple[float, float, float],
                   top_percent: float = 15.0,
                   min_x_distance: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect medial and lateral landmarks in tibial plateau region.
    
    This function uses a combination of skeletonization and distance transform
    to identify anatomically relevant landmark points in the tibia bone.
    
    Args:
        tibia_3d (np.ndarray): 3D binary mask of tibia bone
        spacing (Tuple[float, float, float]): Voxel spacing in mm (x, y, z)
        top_percent (float): Percentage of top slices to consider as plateau. Default: 15.0
        min_x_distance (int): Minimum distance from midline in pixels. Default: 10
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Coordinates of medial and lateral landmarks
                                     as [x, y, z] arrays
        
    Raises:
        ValueError: If tibia_3d is not 3D or contains no foreground pixels
        
    Example:
        >>> tibia = np.random.randint(0, 2, (128, 128, 64))
        >>> spacing = (1.0, 1.0, 2.0)
        >>> medial, lateral = find_landmarks(tibia, spacing)
        >>> print(f"Medial: {medial}, Lateral: {lateral}")
    """
    if tibia_3d.ndim != 3:
        raise ValueError("Input tibia volume must be 3D")
    
    if not np.any(tibia_3d):
        raise ValueError("Input tibia volume contains no foreground pixels")
    
    # Step 1: Identify tibial plateau region
    # Find slices containing tibia data
    z_nonzero = np.any(tibia_3d, axis=(0, 1))
    z_indices = np.where(z_nonzero)[0]
    
    if len(z_indices) == 0:
        raise ValueError("No valid slices found in tibia volume")
    
    # Calculate plateau region (top percentage of slices)
    z_cutoff = int(np.percentile(z_indices, 100 - top_percent))
    
    # Create plateau mask focusing on superior region
    plateau_3d = np.zeros_like(tibia_3d, dtype=bool)
    plateau_3d[:, :, z_cutoff:] = tibia_3d[:, :, z_cutoff:]
    
    print(f"Plateau region: slices {z_cutoff} to {tibia_3d.shape[2]-1}")
    
    # Step 2: Calculate distance transform for depth weighting
    # Distance from each point to nearest background pixel
    distance_map = distance_transform_edt(plateau_3d, sampling=spacing)
    
    # Step 3: Skeletonization to find structural centerlines
    skeleton_points = []
    
    # Process each slice in plateau region
    for z in range(z_cutoff, plateau_3d.shape[2]):
        slice_mask = plateau_3d[:, :, z]
        
        # Skip empty slices
        if not np.any(slice_mask):
            continue
        
        # Apply 2D skeletonization to current slice
        skeleton_2d = skeletonize(slice_mask)
        
        # Extract skeleton coordinates
        skeleton_coords = np.argwhere(skeleton_2d)  # Returns [x, y] coordinates
        
        # Add z-coordinate and convert to [x, y, z] format
        for x, y in skeleton_coords:
            skeleton_points.append([x, y, z])
    
    if len(skeleton_points) == 0:
        raise ValueError("No skeleton points found in plateau region")
    
    skeleton_points = np.array(skeleton_points)
    print(f"Found {len(skeleton_points)} skeleton points")
    
    # Step 4: Calculate weighted scores for landmark selection
    # Extract coordinates for processing
    coords_3d = skeleton_points
    
    # Get distance values at skeleton points
    distance_values = distance_map[coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2]]
    
    # Step 5: Establish medial-lateral separation using x-coordinate median
    x_median = np.median(coords_3d[:, 0])
    x_distances = np.abs(coords_3d[:, 0] - x_median)
    
    # Filter points that are sufficiently far from midline
    # This ensures clear medial/lateral separation
    valid_indices = np.where(x_distances > min_x_distance)[0]
    
    if len(valid_indices) == 0:
        # If no points are far enough, use all points
        print("Warning: No points found far from midline, using all skeleton points")
        valid_indices = np.arange(len(coords_3d))
    
    # Filter coordinates and values
    filtered_coords = coords_3d[valid_indices]
    filtered_distances = distance_values[valid_indices]
    filtered_x_distances = x_distances[valid_indices]
    
    # Step 6: Calculate composite weighting score
    # Weight by distance from x-midline (larger separation = better)
    x_weight = 1.0 + (filtered_x_distances / np.max(filtered_x_distances))
    
    # Weight by z-depth (closer to plateau top = better)
    z_scores = 1.0 - (filtered_coords[:, 2] - z_cutoff) / (plateau_3d.shape[2] - z_cutoff)
    
    # Combine distance transform value with positional weights
    composite_scores = filtered_distances * x_weight * z_scores
    
    # Step 7: Select medial and lateral landmarks
    # Split points based on x-coordinate relative to median
    medial_indices = np.where(filtered_coords[:, 0] < x_median)[0]
    lateral_indices = np.where(filtered_coords[:, 0] >= x_median)[0]
    
    if len(medial_indices) == 0 or len(lateral_indices) == 0:
        raise ValueError("Could not find both medial and lateral points")
    
    # Select best points from each side
    medial_best_idx = medial_indices[np.argmax(composite_scores[medial_indices])]
    lateral_best_idx = lateral_indices[np.argmax(composite_scores[lateral_indices])]
    
    medial_landmark = filtered_coords[medial_best_idx]
    lateral_landmark = filtered_coords[lateral_best_idx]
    
    print(f"Selected landmarks:")
    print(f"Medial: [{medial_landmark[0]}, {medial_landmark[1]}, {medial_landmark[2]}]")
    print(f"Lateral: [{lateral_landmark[0]}, {lateral_landmark[1]}, {lateral_landmark[2]}]")
    
    return medial_landmark, lateral_landmark


def save_landmarks(medial: np.ndarray, 
                   lateral: np.ndarray, 
                   output_path: str,
                   spacing: Optional[Tuple[float, float, float]] = None) -> None:
    """
    Save landmark coordinates to text file.
    
    Args:
        medial (np.ndarray): Medial landmark coordinates [x, y, z]
        lateral (np.ndarray): Lateral landmark coordinates [x, y, z]
        output_path (str): Output file path
        spacing (Optional[Tuple[float, float, float]]): Voxel spacing for mm conversion
    """
    with open(output_path, 'a') as f:
        f.write("# Tibial Plateau Landmark Coordinates\n")
        f.write("# Format: Landmark, X_pixel, Y_pixel, Z_pixel")
        
        if spacing:
            f.write(", X_mm, Y_mm, Z_mm")
        f.write("\n")
        
        # Write medial landmark
        f.write(f"Medial, {medial[0]:.1f}, {medial[1]:.1f}, {medial[2]:.1f}")
        if spacing:
            x_mm = medial[0] * spacing[0]
            y_mm = medial[1] * spacing[1]
            z_mm = medial[2] * spacing[2]
            f.write(f", {x_mm:.2f}, {y_mm:.2f}, {z_mm:.2f}")
        f.write("\n")
        
        # Write lateral landmark
        f.write(f"Lateral, {lateral[0]:.1f}, {lateral[1]:.1f}, {lateral[2]:.1f}")
        if spacing:
            x_mm = lateral[0] * spacing[0]
            y_mm = lateral[1] * spacing[1]
            z_mm = lateral[2] * spacing[2]
            f.write(f", {x_mm:.2f}, {y_mm:.2f}, {z_mm:.2f}")
        f.write("\n")
    
    print(f"Landmarks saved to: {output_path}")


def main() -> None:
    """
    Main execution function for landmark detection module.
    
    Loads segmented tibia data and detects medial/lateral landmarks.
    """
    try:
        # Get segmented data path from environment variable
        segmented_path = os.environ.get("SEGMENTED_DATA_PATH")
        if not segmented_path:
            raise ValueError("SEGMENTED_DATA_PATH environment variable not set")
        
        print("Loading segmented volume...")
        segmented_volume = nib.load(segmented_path)
        labels = segmented_volume.get_fdata()
        spacing = segmented_volume.header.get_zooms()
        
        print(f"Volume shape: {labels.shape}")
        print(f"Voxel spacing: {spacing}")
        
        # Extract tibia mask (assuming label 1 is tibia)
        tibia_3d = (labels == 1).astype(np.float32)
        tibia_voxels = np.sum(tibia_3d)
        
        if tibia_voxels == 0:
            raise ValueError("No tibia voxels found with label 1")
        
        print(f"Tibia contains {tibia_voxels} voxels")
        
        # Detect landmarks
        print("Detecting landmarks...")
        medial, lateral = find_landmarks(tibia_3d, spacing)
        
        # Save results
        output_path = os.environ["STORE_LOCATION"] + "landmarks.txt"
        # save_landmarks(medial, lateral, output_path, spacing)
        
        # Visualize results (optional)
        show_landmarks(tibia_3d, medial, lateral)
        
        print("Landmark detection complete!")
        
    except Exception as e:
        print(f"Error in landmark detection: {str(e)}")
        raise


if __name__ == "__main__":
    main()