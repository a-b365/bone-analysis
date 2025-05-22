
# 🦴 Medical Image Processing: Knee CT Scan Analysis

## 🏥 Overview

This project applies classical image processing techniques to knee CT scans, focusing on bone segmentation, contour manipulation, and landmark detection. Designed for those who are new to medical imaging, it extracts critical anatomical insights from medical images, specifically targeting femur and tibia structures.

---

## 🎯 Features

- **Automated Bone Segmentation** – Isolate femur and tibia from axial CT slices  
- **Contour Expansion** – Expand bone masks (e.g., +2mm, +4mm) using distance transform  
- **Mask Randomization** – Introduce 20% and 80% controlled variations  
- **Landmark Detection** – Extract medial and lateral tibial plateau points  
- **3D Visualization** – Render segmentation results with interactive tools  
- **Medical File Support** – Native handling of `.nii` (NIfTI) imaging format

---

## 🔬 Technical Approach

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

## 📁 Project Structure

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
│   ├── coordinates.txt           # Landmark coordinates
│   └── description.txt           # Includes the description of the files
├── images/
│   └── *.png                     # Example rendered visualizations
├── docs/
│   └── report.pdf                # Methodology and technical documentation
├── requirements.txt              # Python dependencies
├── env.ps1                       # Windows setup script
└── README.md                     # Project overview and instructions
---

## 🚀 Getting Started

### 🔧 Requirements

- Python 3.8+
- Git
- PowerShell (for Windows users)

### 💻 Installation

Clone and set up the environment:

```bash
git clone https://github.com/a-b365/bone-analysis.git
cd bone_analysis
```

For Windows:
```powershell
.\env.ps1
```

Manual (cross-platform):
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## 🧪 Example Usage

```python
from segmentation import watershed_segmentation
from contour_adjustment import expand_mask
from landmark_detection import find_landmarks

# Load CT and segment bones
labels = watershed_segmentation(volume_3d)

# Expand mask
expanded_mask = expand_mask(mask, spacing, expansion=2.0)

# Identify landmarks
medial, lateral = find_landmarks(tibia_3d, spacing)

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

## 🧠 Methodology

**Preprocessing:**
- HU thresholding
- Morphological Operations
- Binary Hole filling
- Connected component filtering

**Segmentation:**
- Adaptive thresholding
- Watershed + cleanup

**Distance Transform Applications:**
- Contour expansion
- Mask randomization
- Landmark skeletonization

---

## 🤝 Contributing

1. Fork the repo
2. Create a branch (`git checkout -b feature/new-algorithm`)
3. Commit your changes
4. Push to GitHub
5. Open a Pull Request

---

## 🙏 Acknowledgments

- Nepal Applied Mathematics and Informatics Institute for research (NAAMII)
- `scikit-image`, `nibabel`, and open-source communities
- AI tools including ChatGPT, Claude AI, and Perplexity

---

## 🔍 Citation

```bibtex
@software{medical_image_processing_2024,
  title={Medical Image Processing: Bone Analysis},
  author={Amir Bhattarai},
  year={2024},
  url={https://github.com/a-b365/bone-analysis.git}
}
```
