# Preprocessing

The `preprocessing` package handles video frame preprocessing, silhouette extraction, Gait Energy Image (GEI) synthesis, CASIA-B dataset processing, and data augmentation in ARGUS AI.

## Responsibilities

- Extracting clean binary silhouettes from cropped person images using Otsu thresholding and morphology.
- Synthesizing 64x128 Gait Energy Images (GEIs) over temporal silhouette sequence windows.
- Parsing and structuring raw benchmark datasets (such as CASIA-B) into standard directory layouts.
- Applying data augmentation (horizontal flip, random cropping, rotation) for model training.
- Boundaries: Does not run deep CNN inference or execute vector similarity matching.

## Key Modules

<!-- BEGIN SYNC: KEY_MODULES -->
| Module | Purpose |
|---|---|
| [augmentation.py](file:///E:/ARGUS_AI/preprocessing/augmentation.py) | Data augmentation transformations for GEI images during model training |
| [casia_extractor.py](file:///E:/ARGUS_AI/preprocessing/casia_extractor.py) | Dataset parser extracting silhouettes and metadata from raw CASIA-B structure |
| [dataset_builder.py](file:///E:/ARGUS_AI/preprocessing/dataset_builder.py) | Builds train/test dataset splits and pre-computes GEI feature caches |
| [gei_builder.py](file:///E:/ARGUS_AI/preprocessing/gei_builder.py) | Synthesizes Gait Energy Images by temporal averaging over binary silhouette frames |
| [silhouette_extractor.py](file:///E:/ARGUS_AI/preprocessing/silhouette_extractor.py) | Extracts binary human silhouettes using Otsu thresholding, morphological filters, and contour sizing |
| [skeleton_extractor.py](file:///E:/ARGUS_AI/preprocessing/skeleton_extractor.py) | Experimental skeleton pose estimation helper module |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

Person Crop BGR Image → `preprocessing/silhouette_extractor.py` → 64×128 Binary Silhouette Mask → `preprocessing/gei_builder.py` → 64×128 GEI Image.

## Configuration

- [configs/gei.yaml](file:///e:/ARGUS_AI/configs/gei.yaml): silhouette size, Otsu blur parameters, GEI temporal window length

## Public Interfaces

- `SilhouetteExtractor`: Silhouette generation engine in [preprocessing/silhouette_extractor.py](file:///e:/ARGUS_AI/preprocessing/silhouette_extractor.py).
- `GEIBuilder`: GEI temporal averaging generator in [preprocessing/gei_builder.py](file:///e:/ARGUS_AI/preprocessing/gei_builder.py).
- `DatasetBuilder`: Dataset pre-processor in [preprocessing/dataset_builder.py](file:///e:/ARGUS_AI/preprocessing/dataset_builder.py).
- `CASIABExtractor`: CASIA-B dataset parser in [preprocessing/casia_extractor.py](file:///e:/ARGUS_AI/preprocessing/casia_extractor.py).

## Tests

- [tests/test_silhouette.py](file:///e:/ARGUS_AI/tests/test_silhouette.py)
- [tests/test_gei_stream.py](file:///e:/ARGUS_AI/tests/test_gei_stream.py)

## Related Documentation

- [Root README](file:///e:/ARGUS_AI/README.md)
- [Training Documentation](file:///e:/ARGUS_AI/training/README.md)
