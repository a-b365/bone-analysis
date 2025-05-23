"""
Visualization Module

This module provides visualization functions for medical image analysis tasks
including contour plotting, distance transforms, mask expansions, segmentation
results, and landmark detection visualization.

Author: Amir Bhattarai
Date: 2025-05-23
Version: 1.0
"""

# Standard library imports
from typing import List, Tuple, Union

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab


def show_contours(mask: np.ndarray, contours: List[np.ndarray]) -> None:
    """
    Display 2D contours overlaid on a mask image.
    
    This function plots detected contours on top of the original mask image
    using matplotlib for 2D visualization.
    
    Args:
        mask (np.ndarray): 2D binary mask array to display as background
        contours (List[np.ndarray]): List of contour arrays, each containing 
                                   (x, y) coordinates of contour points
    
    Returns:
        None: Displays the plot using matplotlib.pyplot.show()
    
    Example:
        >>> mask = np.zeros((100, 100))
        >>> contours = [np.array([[10, 10], [20, 10], [20, 20]])]
        >>> show_contours(mask, contours)
    """
    # Plot each contour on the image
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0])
    
    # Display the mask as background in grayscale
    plt.imshow(mask, cmap="gray")
    plt.show()


def show_distance_transform(dist_matrix: np.ndarray) -> None:
    """
    Visualize a distance transform matrix using a heat map.
    
    This function displays the distance transform as a color-coded image
    where intensity represents distance values.
    
    Args:
        dist_matrix (np.ndarray): 2D array containing distance transform values
    
    Returns:
        None: Displays the plot using matplotlib.pyplot.show()
    
    Example:
        >>> dist_matrix = np.random.rand(50, 50)
        >>> show_distance_transform(dist_matrix)
    """
    plt.imshow(dist_matrix, cmap="hot")
    plt.colorbar(label="Distance")
    plt.title("Distance Transform")
    plt.show()


def visualize_expansion(expanded_mask: np.ndarray, expansion: float) -> None:
    """
    Visualize 3D expanded mask using Mayavi contour plotting.
    
    This function creates a 3D visualization of an expanded binary mask,
    typically used to show the results of morphological expansion operations.
    
    Args:
        expanded_mask (np.ndarray): 3D binary array representing the expanded mask
        expansion (float): Expansion distance in millimeters for title display
    
    Returns:
        None: Displays the 3D plot using mlab.show()
    
    Example:
        >>> expanded_mask = np.random.randint(0, 2, (50, 50, 50))
        >>> visualize_expansion(expanded_mask, 2.0)
    """
    mlab.contour3d(expanded_mask, color=(1, 1, 1))
    mlab.title(f"{expansion}mm Expanded Mask")
    mlab.show()


def visualize_randomness(randomized_mask: np.ndarray, expansion: float) -> None:
    """
    Visualize 3D randomized mask using Mayavi contour plotting.
    
    This function creates a 3D visualization of a randomized binary mask,
    typically used to show the results of stochastic mask generation.
    
    Args:
        randomized_mask (np.ndarray): 3D binary array representing the randomized mask
        expansion (float): Expansion distance in millimeters for title display
    
    Returns:
        None: Displays the 3D plot using mlab.show()
    
    Example:
        >>> randomized_mask = np.random.randint(0, 2, (50, 50, 50))
        >>> visualize_randomness(randomized_mask, 2.0)
    """
    mlab.contour3d(randomized_mask, color=(1, 1, 1))
    mlab.title(f"{expansion}mm Randomized Mask")
    mlab.show()


def visualize_segments(labels: np.ndarray) -> None:
    """
    Visualize 3D segmentation results showing different anatomical structures.
    
    This function displays segmented anatomical structures (tibia and femur)
    in different colors using 3D contour visualization.
    
    Args:
        labels (np.ndarray): 3D integer array where different values represent
                           different segmented structures:
                           - Label 1: Femur (displayed in red)
                           - Label 2: Tibia (displayed in green)
    
    Returns:
        None: Displays the 3D plot using mlab.show()
    
    Example:
        >>> labels = np.random.randint(0, 3, (50, 50, 50))
        >>> visualize_segments(labels)
    """
    # Extract individual structures from labeled volume
    tibia = (labels == 1).astype(np.float32)
    femur = (labels == 2).astype(np.float32)
    
    # Visualize structures in different colors
    mlab.contour3d(tibia, color=(0, 1, 0))  # Green for tibia
    mlab.contour3d(femur, color=(1, 0, 0))  # Red for femur
    mlab.title("Segmentation Results", size=1)
    mlab.show()


def show_landmarks(volume: np.ndarray, 
                  medial: Union[np.ndarray, Tuple[float, float, float]], 
                  lateral: Union[np.ndarray, Tuple[float, float, float]]) -> None:
    """
    Visualize anatomical landmarks on a 3D volume.
    
    This function displays detected landmark points (medial and lateral)
    overlaid on a semi-transparent 3D volume visualization.
    
    Args:
        volume (np.ndarray): 3D binary array representing the anatomical structure
        medial (Union[np.ndarray, Tuple[float, float, float]]): 3D coordinates 
                                                               of medial landmark
        lateral (Union[np.ndarray, Tuple[float, float, float]]): 3D coordinates 
                                                                of lateral landmark
    
    Returns:
        None: Displays the 3D plot using mlab.show()
    
    Example:
        >>> volume = np.random.randint(0, 2, (50, 50, 50))
        >>> medial = [25, 15, 30]
        >>> lateral = [25, 35, 30]
        >>> show_landmarks(volume, medial, lateral)
    """
    # Display the volume with transparency
    mlab.contour3d(volume, color=(1, 1, 1), opacity=0.3)
    
    # Display landmark points
    mlab.points3d(medial[0], medial[1], medial[2], 
                  color=(1, 0, 0),  # Red for medial
                  scale_factor=4, 
                  resolution=20)
    
    mlab.points3d(lateral[0], lateral[1], lateral[2], 
                  color=(0, 0, 1),  # Blue for lateral
                  scale_factor=4, 
                  resolution=20)
    
    mlab.title("Anatomical Landmarks")
    mlab.show()


# Module-level constants
DEFAULT_COLORMAP = "gray"
DEFAULT_SCALE_FACTOR = 4
DEFAULT_OPACITY = 0.3

# Color definitions for consistent visualization
COLORS = {
    'white': (1, 1, 1),
    'red': (1, 0, 0),
    'green': (0, 1, 0),
    'blue': (0, 0, 1)
}


if __name__ == "__main__":
    """
    Module test and demonstration code.
    
    This section provides basic testing functionality when the module
    is run directly. It creates sample data and demonstrates the
    visualization functions.
    """
    print("Medical Image Analysis Visualization Module")
    print("==========================================")
    print("This module provides visualization functions for:")
    print("- 2D contour visualization")
    print("- Distance transform visualization") 
    print("- 3D mask expansion visualization")
    print("- 3D segmentation visualization")
    print("- 3D landmark visualization")
    print("\nImport this module to use the visualization functions.")