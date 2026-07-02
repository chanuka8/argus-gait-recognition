# ARGUS Gait Module: Photo-Only Recognition Path Removal Report

**Date:** 2026-06-13  
**Status:** Completed & Audited

This report details the deactivation and removal of all photo-only recognition components and pathways from the active runtime of the ARGUS Biometric Surveillance & Gait Recognition Module.

---

## 1. Rationale for Removal
True gait recognition depends on walking motion characteristics extracted from a temporal silhouette sequence (GEI). Photo-only data does not contain motion patterns and is inadequate for contactless, long-range gait matching. Under a hybrid configuration, utilizing static photos for gait recognition results in false matches or invalid states. For professional and research validity, the active recognition pipeline has been decoupled from the photo/appearance database. Photo/appearance recognition is now explicitly marked as future work requiring a deep Person Re-Identification (Person ReID) network.

---

## 2. Modified Runtime Files

### 1. [live_recognition.py](file:///e:/ARGUS_AI/pipeline/live_recognition.py)
- **Modifications:** 
  - Removed imports for `AppearanceFeatureExtractionStep`, `AppearanceMatchingStep`, and `HumanReviewDecisionEngine`.
  - Removed appearance-specific attributes (`self.appearance_matcher`, `self.appearance_extractor`, and `self.review_engine`) and appearance gallery directory loading from the constructor.
  - Rewrote `_recognize_track()` to perform gait-only identification using `models/live_gallery` and apply prediction smoothing.
  - Decoupled `_final_identity()` from the hybrid decision layer, binding validation to the `self.threshold` (0.85) check.
  - Aligned the OpenCV overlay color mapper (`_get_label_color`) to:
    - `COLLECTING` → Yellow `(0, 255, 255)`
    - `UNKNOWN` → Green `(0, 255, 0)`
    - Confirmed Match → Red `(0, 0, 255)`
  - Removed all `PHOTO_REVIEW`/`REVIEW_REQUIRED` references from the screen box drawings.

### 2. [auto_enrollment_service.py](file:///e:/ARGUS_AI/enrollment/auto_enrollment_service.py)
- **Modifications:**
  - Modified `enroll_pending()` to check folders before processing.
  - If a folder contains only images and no videos (photo-only), it prints `"Skipped photo-only folder. Gait enrollment requires video: <person_id>"` and skips enrollment.
  - If a folder contains both images and videos (mixed), it ignores the images and processes only the video files to extract GEIs for live gait enrollment.
  - Removed calls to photo preparation and appearance gallery updater.
  - Set `has_photos=False` when verifying fingerprint-based re-enrollment status.

### 3. [enrollment_manager.py](file:///e:/ARGUS_AI/enrollment/enrollment_manager.py)
- **Modifications:**
  - Modified `enroll_person()` to skip/fail photo-only or mixed folder requests.
  - Photo-only folders fail with a message: `"Skipped photo-only folder. Gait enrollment requires video."`.
  - Mixed folders fail with a message: `"Gait enrollment requires video processing. Use AutoEnrollmentService to generate gait GEIs from video."`.

### 4. [alert_manager.py](file:///e:/ARGUS_AI/utils/alert_manager.py)
- **Modifications:**
  - Restricted alert outputs to `UNKNOWN_PERSON`, `LOW_CONFIDENCE`, and `CONFIRMED_MATCH`.
  - Added logging/alerts for `CONFIRMED_MATCH` events when the matching score exceeds the confidence threshold (0.90) and the identity is not unknown.

### 5. [inference_pipeline.py](file:///e:/ARGUS_AI/pipeline/inference_pipeline.py)
- **Modifications:**
  - Re-routed default `VectorStore` gallery directory load path from `"models/gallery"` to `"models/live_gallery"`.
  - Passed `self.metadata` to `self.matcher.match()` so that offline folder predictions respect identity status constraints (ACTIVE, DISABLED, ARCHIVED).

### 6. [folder_recognition.py](file:///e:/ARGUS_AI/pipeline/folder_recognition.py)
- **Modifications:**
  - Initialized `InferencePipeline` with a `0.0` default threshold inside the constructor.
  - In `run()`, dynamically set `self.pipeline.matcher.threshold = threshold` before processing images to ensure the threshold flag passed via CLI is fully respected.

---

## 3. Verification & Compliance
- Running `python cli.py --mode health` passes successfully.
- Running `python cli.py --mode tests` and `pytest` passes 15/15 unit and integration tests successfully.
- Auto-enrollment directory watcher correctly skips folders containing photos only, and only registers video-derived GEIs.
- Active surveillance overlays draw green boxes for unknown users and red boxes for validated matches.
