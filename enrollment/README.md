# Enrollment

The `enrollment` package handles target identity registration, gallery database management, background folder watching for auto-enrollment, and image quality validation in ARGUS AI.

## Responsibilities

- Enrolling new target identities into gait and appearance gallery databases.
- Watching configured directories for automated target image/video drop-in enrollment.
- Validating candidate enrollment samples for GEI quality, resolution, and format compliance.
- Boundaries: Does not handle real-time streaming video capture or live threat alert dispatching.

## Key Modules

<!-- BEGIN SYNC: KEY_MODULES -->
| Module | Purpose |
| --- | --- |
| [appearance_gallery_updater.py](appearance_gallery_updater.py) | Updates appearance feature embeddings in the enrollment gallery |
| [auto_enrollment_service.py](auto_enrollment_service.py) | Automated background service for target identity auto-registration |
| [enrollment_lifecycle.py](enrollment_lifecycle.py) | Module/resource file enrollment_lifecycle.py |
| [enrollment_manager.py](enrollment_manager.py) | Main enrollment facade handling target identity addition, deletion, and sync |
| [enrollment_queue.py](enrollment_queue.py) | Thread-safe asynchronous queue for batch enrollment tasks |
| [enrollment_validator.py](enrollment_validator.py) | Quality inspector verifying resolution, silhouette suitability, and GEI validity |
| [folder_watcher.py](folder_watcher.py) | Filesystem observer monitoring designated directories for new enrollment files |
| [gallery_updater.py](gallery_updater.py) | Computes and persists gait GEI feature embeddings into `models/gallery/` |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

Enrollment Files → `enrollment/folder_watcher.py` → `enrollment/enrollment_validator.py` → `enrollment/gallery_updater.py` → `models/gallery/` (`.npy` & `.json`).

## Configuration

- [configs/system.yaml](../configs/system.yaml): `recognition.gallery_dir`
- [configs/inference.yaml](../configs/inference.yaml): matching policy thresholds

## Public Interfaces

- `EnrollmentManager`: Primary facade for identity registration in [enrollment/enrollment_manager.py](enrollment_manager.py).
- `AutoEnrollmentService`: Background registration worker in [enrollment/auto_enrollment_service.py](auto_enrollment_service.py).
- `GalleryUpdater`: Feature extraction and matrix persistence engine in [enrollment/gallery_updater.py](gallery_updater.py).

## Tests

- [tests/integration/test_dual_modal_pipeline.py](../tests/integration/test_dual_modal_pipeline.py)
- [tests/test_watchlist_integration.py](../tests/test_watchlist_integration.py)

## Related Documentation

- [Root README](../README.md)
- [Intelligence Documentation](../intelligence/README.md)
