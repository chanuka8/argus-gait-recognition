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
| --- | --- |
| [augmentation.py](augmentation.py) | Data augmentation transformations for GEI images during model training |
| [casia_extractor.py](casia_extractor.py) | Dataset parser extracting silhouettes and metadata from raw CASIA-B structure |
| [dataset_builder.py](dataset_builder.py) | Builds train/test dataset splits and pre-computes GEI feature caches |
| [gei_builder.py](gei_builder.py) | Synthesizes Gait Energy Images by temporal averaging over binary silhouette frames |
| [image_enhancement.py](image_enhancement.py) | Module/resource file image_enhancement.py |
| [silhouette_extractor.py](silhouette_extractor.py) | Extracts binary human silhouettes using Otsu thresholding, morphological filters, and contour sizing |
| [skeleton_extractor.py](skeleton_extractor.py) | Experimental skeleton pose estimation helper module |
| [video_quality_gate.py](video_quality_gate.py) | Module/resource file video_quality_gate.py |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

Person Crop BGR Image → `preprocessing/silhouette_extractor.py` → 64×128 Binary Silhouette Mask → `preprocessing/gei_builder.py` → 64×128 GEI Image.

## Configuration

- [configs/gei.yaml](../configs/gei.yaml): silhouette size, Otsu blur parameters, GEI temporal window length

## Public Interfaces

- `SilhouetteExtractor`: Silhouette generation engine in [preprocessing/silhouette_extractor.py](silhouette_extractor.py).
- `GEIBuilder`: GEI temporal averaging generator in [preprocessing/gei_builder.py](gei_builder.py).
- `DatasetBuilder`: Dataset pre-processor in [preprocessing/dataset_builder.py](dataset_builder.py).
- `CASIABExtractor`: CASIA-B dataset parser in [preprocessing/casia_extractor.py](casia_extractor.py).

## Tests

- [tests/test_silhouette.py](../tests/test_silhouette.py)
- [tests/test_gei_stream.py](../tests/test_gei_stream.py)

## Related Documentation

- [Root README](../README.md)
- [Training Documentation](../training/README.md)
