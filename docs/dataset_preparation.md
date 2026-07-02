# Dataset Preparation Guide

This document explains the steps to acquire, extract, and preprocess the **CASIA-B** dataset to prepare it for model training and gallery operations.

---

## 1. CASIA-B Dataset Overview

CASIA-B is a widely used multi-view gait dataset containing silhouette data from **124 subjects** captured from **11 view angles** ranging from $0^\circ$ to $180^\circ$ in steps of $18^\circ$. Each subject is recorded walking under three different conditions:
- **Normal Walking (nm):** 6 sequences per subject (`nm-01` to `nm-06`).
- **Carrying a Bag (bg):** 2 sequences per subject (`bg-01` to `bg-02`).
- **Wearing a Coat/Jacket (cl):** 2 sequences per subject (`cl-01` to `cl-02`).

---

## 2. Preprocessing & Extraction Pipeline

The preprocessing workflow translates raw frame silhouettes into a single normalized **Gait Energy Image (GEI)** for each sequence. A GEI represents the pixel-wise average intensity of silhouettes captured over a complete gait cycle, condensing spatial-temporal gait parameters into a single $128 \times 64$ grayscale image.

The preprocessing steps are managed by [preprocessing/dataset_builder.py](file:///e:/ARGUS_AI/preprocessing/dataset_builder.py) and [preprocessing/gei_builder.py](file:///e:/ARGUS_AI/preprocessing/gei_builder.py):

```
       +---------------------------------------------+
       |   Raw CASIA-B Silhouette Zip File (ZIP)     |
       +----------------------┬----------------------+
                              │ (Extract Zip)
                              ▼
       +---------------------------------------------+
       |   Sequential Binary Silhouette PNG Frames   |
       +----------------------┬----------------------+
                              │ (Crop & Morphological Clean)
                              ▼
       +---------------------------------------------+
       |       Normalized 128x64 Grayscale Frames    |
       +----------------------┬----------------------+
                              │ (Accumulate & Average)
                              ▼
       +---------------------------------------------+
       |     Final Gait Energy Image (GEI) PNG       |
       +---------------------------------------------+
```

1. **Extraction:** The script reads the raw ZIP file in a streaming fashion using `ZipStreamer` to avoid filling up the disk.
2. **Segmentation & Crop:** Silhouettes are resized, cropped to bounding box margins, and centered into a $128 \times 64$ binary image.
3. **Temporal Averaging (GEI Build):** For each walk sequence (e.g. `nm-01`), the silhouette frames are averaged:
   $$GEI(x, y) = \frac{1}{N} \sum_{t=1}^{N} I(x, y, t)$$
   where $I(x, y, t)$ is the silhouette pixel intensity at $(x, y)$ in frame $t$, and $N$ is the total frames in the gait cycle.

---

## 3. Preprocessing Command

To execute the CASIA-B preprocessing script:
```bash
python cli.py --mode preprocess
```
*Alternative:* Use python directly to specify paths:
```bash
python scripts/preprocess_casia.py --zip-path data/GaitDatasetB-silh.zip --output-dir data/casia_processed
```

---

## 4. Directory Structure Outcomes

Once preprocessing completes, the preprocessed database compiles under the following structure:
```
data/casia_processed/
└── gei/
    ├── 001/
    │   ├── 001_nm-01_108.png
    │   ├── 001_nm-02_090.png
    │   ├── 001_bg-01_072.png
    │   └── 001_cl-01_054.png
    ├── 002/
    └── ...
```
Each file is formatted as: `{subject_id}_{walking_condition}-{sequence_id}_{view_angle}.png`
- Example: `034_nm-01_126.png` represents Subject `034`, Normal Walking sequence `01`, filmed at a $126^\circ$ view angle.
- The output images are 8-bit grayscale PNGs of size $128 \times 64$.
- These compiled files serve as the inputs for Model Training and Vector Gallery creation.
