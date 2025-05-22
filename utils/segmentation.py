# Standard library imports
import os

# Third party libraries
import nibabel as nib
import numpy as np

# Relative imports
from mayavi import mlab
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max

# Local imports
from plots import visualize_segments

def watershed_segmentation(volume_3d):
    distance = distance_transform_edt(volume_3d.astype(bool))
    coords = peak_local_max(distance, footprint=np.ones((100,100,100)), labels=volume_3d.astype(bool))
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=volume_3d.astype(bool))
    return labels


if __name__ == "__main__":

    mask_volume = nib.load(os.environ["MASK_DATA_PATH"])
    mask_data = mask_volume.get_fdata()
    volume_3d = np.stack(mask_data, axis=0)
    labels = watershed_segmentation(volume_3d)
    visualize_segments(labels)



