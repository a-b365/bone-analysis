
# Medical Image Processing: Knee CT Scan Analysis

## Overview

This project applies classical image processing techniques to knee CT scans, focusing on bone segmentation, contour manipulation, and landmark detection. It extracts critical anatomical insights from medical images, specifically targeting femur and tibia structures.

---

## Features

- **Automated Bone Segmentation** – Isolate femur and tibia from axial CT slices  
- **Contour Expansion** – Expand bone masks (e.g., +2mm, +4mm) using distance transform  
- **Mask Randomization** – Introduce controlled randomness (e.g. 20%, 80%)
- **Landmark Detection** – Extract medial and lateral tibial plateau points  
- **3D Visualization** – Render segmentation results with interactive tools  
- **Medical File Support** – Native handling of `.nii` (NIfTI) imaging format

---

## Technical Approach

**Core Methods:**
- Watershed segmentation (2D slice-wise)
- Distance transform for expansion/randomization
- Skeletonization for landmark localization
- Morphological preprocessing (hole filling, noise cleanup)

**Stack:**
- `scikit-image`, `scipy`, `opencv`, `nibabel`
- Visualization: `matplotlib`, `mayavi`
- Helper libs: `numpy`, `pandas`

---

## Project Structure

```
bone_analysis/
├── utils/
│   ├── segmentation.py           # Segmentation & 3D rendering
│   ├── contour_adjustment.py     # Expansion and randomization
│   ├── landmark_detection.py     # Landmark extraction
│   └── plots.py                  # 2D & 3D visualization helpers
├── notebooks/
│   └── notebook.ipynb            # Interactive Jupyter notebook
├── results/
│   ├── *.nii                     # Processed segmentations
│   ├── landmark_coordinates.txt  # Landmark coordinates
│   └── codebase_description.txt  # Includes the description of the files
├── images/
│   └── *.png                     # Example rendered visualizations
├── docs/
│   └── report.pdf                # Methodology and technical documentation
├── requirements.txt              # Python dependencies
├── env.ps1                       # Windows setup script
└── readme.md                     # Project overview and instructions
```

---

## 🚀 Setup Instructions

### 🔧 Prerequisites

- Python 3.8+
- Git (to clone the repo)
- PowerShell (for Windows users)

### 💻 Installation

1. Clone the repository

    ```bash
    git clone https://github.com/a-b365/bone-analysis.git
    cd bone_analysis
    ```

2. Run the powershell script to add the environment variables:

    ```powershell
    .\env.ps1
    ```

3. Create and activate a virtual environment:

  - On macOS/Linux:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

  - On Windows:
    ```powershell
    .\venv\Scripts\activate
    ```

4. Install Dependencies

    Run:
    ```
    pip install -r requirements.txt
    ```
---

## Example Usage

```python
from segmentation import watershed_segmentation
from contour_adjustment import expand_mask
from landmark_detection import find_landmarks

# Load CT and segment bones
labels = watershed_segmentation(volume_3d)

# Expand mask
expanded_mask = expand_mask(mask_data, spacing, expansion=2.0)

# Identify landmarks
medial_landmark, lateral_landmark = find_landmarks(tibia_3d, spacing)

# Visualize 3D
from mayavi import mlab
def visualize_segments(labels):
    
    tibia = (labels == 2).astype(np.float32)
    femur = (labels == 1).astype(np.float32)

    mlab.contour3d(tibia, color=(0, 1, 0))
    mlab.contour3d(femur, color=(1, 0, 0))
    mlab.title("Segmentation", size=1)
    mlab.show()
```

---

## 📊 Outputs

- **Segmentation Masks**: Femur/tibia separated
- **Expanded Masks**: 2mm & 4mm expansion
- **Randomized Masks**: 20% & 80% variation
- **Landmarks**: Medial/lateral coordinates in `coordinates.txt`
- **Visuals**: 2D slices and 3D renderings (in `results/images/`)

---

## Methodology

**Preprocessing:**
- HU thresholding
- Morphological Operations
- Binary Hole filling
- Gaussian filtering
- Zoom operations

**Segmentation:**
- Global thresholding
- Watershed + cleanup

**Distance Transform Applications:**
- Contour expansion
- Mask randomization
- Landmark skeletonization

---

## Usage

- Run the desired Python module directly from the command line.
- Upload a document image and click **Process**.
- The app will display one output at a time:
  - Segmentation: Generates masks for femur and tibia.
  - Contour Expansion: Produces expanded and randomized contours of the masks.
  - Landmark Detection: Identifies medial and lateral tibial landmarks.
  - Plots: Visualizes segmentation, contours, distance maps, and landmarks.

---

## Notes

  - A detailed project report is available in the docs/ folder
  - Run the Jupyter notebook inside the notebooks/ folder for an interactive walkthrough of the implementation
  - Visual outputs and intermediate results can be found in the images/ directory.

---

