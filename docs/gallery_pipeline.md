# Biometric Gallery Pipeline

This document explains the compilation, storage, and runtime update mechanisms for the biometric vector gallery database.

---

## 1. Gallery Concept

For gait recognition, the system does not compare raw videos in real-time. Instead, it operates on a **Biometric Gallery**.
A gallery consists of:
- **Features (`.npy`):** A matrix of shape $[N, 256]$ containing L2-normalized deep embedding vectors extracted from subjects' GEIs.
- **Labels (`.npy`):** An array of shape $[N]$ storing corresponding subject IDs (e.g. `034`).
- **Metadata (`.json`):** Key-value maps indexing subject IDs to status (`ACTIVE`, `DISABLED`, `ARCHIVED`), source (`GAIT` or `PHOTO`), file pathways, and register timestamps.

The project maintains two distinct gallery folders:
1. **Validation Gallery (`models/gallery/`):** Built from preprocessed CASIA-B directories for offline evaluation sweeps.
2. **Surveillance Gallery (`models/live_gallery/`):** Used during active camera tracking and real-time auto-enrollments.

---

## 2. Feature Extraction & Serialization

Feature registration is coordinated by [enrollment/enrollment_manager.py](file:///e:/ARGUS_AI/enrollment/enrollment_manager.py):

1. **Input Reader:** Reads silhouette directories or video files.
2. **Quality Gate:** Checks that the folders contain at least 5 frames of readable resolution.
3. **CNN Embedding:** Passes GEIs through `ByGaitLight` to extract 256-dimensional vectors.
4. **Backup Appearance Path:** If a photo (non-gait image) is registered, [AppearanceFeatureExtractionStep](file:///e:/ARGUS_AI/pipeline/steps/appearance_feature_extraction.py) extracts color histogram and Canny edge shapes instead of CNN features, tagging them as `source: "PHOTO"`.
5. **Persistence:** The [VectorStore](file:///e:/ARGUS_AI/storage/vector_store.py) updates the database by appending new rows to the `.npy` files and writing updated metadata configurations to `gallery_metadata.json`.

---

## 3. Gallery Commands

### Build Initial Evaluation Gallery
```bash
python cli.py --mode gallery
```
*Alternative (Makefile):*
```bash
make gallery
```

### Automatic Folder-Watch Enrollment
To launch background disk monitoring:
```bash
python cli.py --mode auto-enroll --input data/new_input
```
*Alternative (Makefile):*
```bash
make auto-enroll
```
This polls `data/new_input/` for new subject directories. Once images are stable, it registers them, writes them to `models/live_gallery/`, and moves/deletes processed inputs.

### Disable or Archive an Identity
Update metadata states to temporarily disable identification matching:
```bash
python cli.py --mode set-status --person-id 034 --status DISABLED
```

### Remove an Identity from Gallery
Completely purge an identity from numpy datasets:
```bash
python cli.py --mode remove-identity --person-id 034
```
This updates `.npy` files by filtering out the matching labels.
