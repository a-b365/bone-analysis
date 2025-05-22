from mayavi import mlab
import numpy as np
import matplotlib.pyplot as plt

def show_contours(mask, contours):
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0])
    plt.imshow(mask, cmap="gray")
    plt.show()

def show_distance_transform(dist_matrix):
    plt.imshow(dist_matrix, cmap="hot")
    plt.show()

def visualize_expansion(expanded_mask, expansion):
    
    mlab.contour3d(expanded_mask, color=(1, 1, 1))
    mlab.title(f"{expansion}mm Expanded Mask")
    mlab.show()

def visualize_randomness(expanded_mask, expansion):
    
    mlab.contour3d(expanded_mask, color=(1, 1, 1))
    mlab.title(f"{expansion}mm Randomized Mask")
    mlab.show()

def visualize_segments(labels):

    tibia = (labels == 2).astype(np.float32)
    femur = (labels == 1).astype(np.float32)

    mlab.contour3d(tibia, color=(0, 1, 0))
    mlab.contour3d(femur, color=(1, 0, 0))
    mlab.title("Segmentation", size=1)
    mlab.show()


def show_landmarks(volume, medial, lateral):

    mlab.contour3d(tibia_3d, color=(1, 1, 1), opacity=0.3)
    mlab.points3d(medial[0], medial[1], medial[2], color=(1, 0, 0), scale_factor=4, resolution=20)
    mlab.points3d(lateral[0], lateral[1], lateral[2], color=(0, 0, 1), scale_factor=4, resolution=20)
    mlab.show()
