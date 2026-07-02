# ARGUS Remaining Known Limitations

**Date:** 2026-06-13  
**Status:** Documented  
**Scope:** Gait Recognition Module Known Constraints

This document lists the remaining known limitations and architectural design assumptions in the ARGUS gait recognition module.

---

## 1. Folder Recognition Discrepancy (InferencePipeline)
* **Description:** `FolderRecognitionPipeline` relies on `InferencePipeline` which defaults to `VectorStore()` pointing to `models/gallery` (the CASIA-B training gallery, not `models/live_gallery`).
* **Impact:** Running `python cli.py --mode recognize-folder` performs matching against CASIA-B numeric IDs instead of live-enrolled names.
* **Metadata Impact:** It does not pass `metadata` to `MatchingStep.match()`, meaning disabled/archived status filtering does not apply during folder-based recognition.
* **Status:** Documented risk. If folder recognition is needed for the live demo, `InferencePipeline` should be modified to load `"models/live_gallery"`.

## 2. No Dynamic Gallery Hot-Reload
* **Description:** The live recognition pipeline (`LiveRecognitionPipeline`) loads the gait and appearance galleries into memory once during initialization (`__init__`).
* **Impact:** If the background auto-enrollment watcher enrolls a new person while the live camera window is open, the live pipeline will not match the new person until the live script is restarted.
* **Workaround:** Restart `--mode system` or `--mode live` to reload the memory-cached vector store.

## 3. ByteTrack Track ID Recyclability
* **Description:** If ByteTrack reuses a track ID for a new person (e.g., person A exits the frame, and person B enters immediately on the same track ID), the `PredictionSmoother`'s history window still contains person A's votes.
* **Impact:** A brief 2–3 frame delay occurs before person B's identity dominates the voting window and gets correctly displayed.

## 4. Silhouette Extraction Sensitivity (Otsu Thresholding)
* **Description:** While CASIA-B silhouettes are extremely clean, real-world webcams produce noisy backgrounds. `SilhouetteStep` relies on basic Otsu thresholding.
* **Impact:** High noise or complex lighting can lead to distorted GEIs and lower embedding similarity scores.
