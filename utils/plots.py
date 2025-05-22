from mayavi import mlab
import numpy as np

def visualize_tibia_landmarks(mask_xyz, medial_xyz, lateral_xyz):
    """
    Visualize tibia 3D mask with medial and lateral landmarks using Mayavi.

    Args:
        mask_xyz (ndarray): 3D binary mask in [x, y, z] format.
        medial_xyz (list or ndarray): [x, y, z] of medial point.
        lateral_xyz (list or ndarray): [x, y, z] of lateral point.
    """

    # Get surface coordinates from the mask
    verts = np.argwhere(mask_xyz)

    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]

    # Plot tibia surface
    mlab.figure(size=(800, 800), bgcolor=(1, 1, 1))
    mlab.points3d(x, y, z,
                  mode='point',
                  color=(0.6, 0.6, 0.6),
                  opacity=0.3,
                  scale_factor=1)

    # Plot medial point (red)
    mlab.points3d(medial_xyz[0], medial_xyz[1], medial_xyz[2],
                  color=(1, 0, 0), scale_factor=4, resolution=20)
    mlab.text3d(medial_xyz[0], medial_xyz[1], medial_xyz[2], 'Medial',
                scale=2, color=(1, 0, 0))

    # Plot lateral point (blue)
    mlab.points3d(lateral_xyz[0], lateral_xyz[1], lateral_xyz[2],
                  color=(0, 0, 1), scale_factor=4, resolution=20)
    mlab.text3d(lateral_xyz[0], lateral_xyz[1], lateral_xyz[2], 'Lateral',
                scale=2, color=(0, 0, 1))

    # Finalize
    mlab.orientation_axes()
    mlab.view(azimuth=180, elevation=90, distance='auto')
    mlab.show()
