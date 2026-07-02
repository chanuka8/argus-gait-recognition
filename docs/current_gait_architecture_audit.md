# ARGUS Gait Recognition Module — Architecture Audit

**Date:** 2026-06-13  
**Scope:** Gait Recognition Module only (no Face Recognition, no Frontend/UI)  
**Project Root:** `E:\ARGUS_AI`

---

## 1. Full Architecture Map: Enrollment to Live Recognition

### A. Enrollment Flow

```
data/new_input/<person_id>/
    ├── *.mp4, *.avi, *.mov    → VIDEO path (gait)
    └── *.png, *.jpg, *.jpeg   → PHOTO path (appearance)
         │
         ▼
AutoEnrollmentService.enroll_pending()
    │
    ├─── _person_folders()          skip names starting with _ or .
    ├─── _input_files()             collect all image + video files
    ├─── _image_files() / _video_files()   classify by extension
    ├─── _already_enrolled()        check both live_store + appearance_store
    ├─── _needs_enrollment()        marker + gallery check
    │
    ├─── VIDEO branch ──────────────────────────────────────────────
    │    │
    │    ├── _prepare_gait_folder()
    │    │   └── _process_video()
    │    │       ├── TrackingStep.track(frame)     YOLOv8 + ByteTrack
    │    │       ├── SilhouetteStep.extract_from_crop(crop)
    │    │       ├── LiveGEI.add(silhouette)        accumulate 15 frames
    │    │       └── LiveGEI.build() → save GEI .png to processed dir
    │    │
    │    └── EnrollmentManager.enroll_gait_person()
    │        ├── EnrollmentValidator.validate_person_folder()   ≥5 images
    │        ├── FeatureExtractionStep.extract(gei_image)
    │        │   ├── _read_grayscale()
    │        │   ├── _normalize_to_silhouette()   binary threshold + morphology
    │        │   └── ByGaitLight model → 256-dim embedding
    │        └── GalleryUpdater.add_person()
    │            └── VectorStore.save() → models/live_gallery/
    │
    └─── PHOTO branch ─────────────────────────────────────────────
         │
         ├── _prepare_photo_folder()   copy images to processed dir
         └── EnrollmentManager.enroll_appearance_person()
             ├── AppearanceFeatureExtractionStep.extract()
             │   ├── HSV color histogram
             │   └── Canny edge shape projection
             └── AppearanceGalleryUpdater.add_person()
                 └── VectorStore.save() → models/appearance_gallery/
```

### B. Live Recognition Flow

```
Camera (StreamEngine)
    │
    ▼
LiveRecognitionPipeline.run()
    │
    ├── StreamEngine.read()                read frame from webcam
    ├── TrackingStep.track(frame)           YOLOv8n person detection → ByteTrack
    │
    │   For each tracked person:
    │   ├── _crop_person(frame, box)        extract bounding box crop
    │   ├── SilhouetteStep.extract_from_crop(crop)
    │   │   ├── grayscale → Gaussian blur → Otsu threshold
    │   │   └── morphological open/close cleanup
    │   ├── LiveGEI.add(silhouette)         accumulate into rolling buffer
    │   │
    │   └── When buffer ready (≥15 frames) && recognition_interval:
    │       ├── LiveGEI.build()             compute mean silhouette → GEI
    │       ├── _gei_to_embedding()         GEI → ByGaitLight → 256-dim vector
    │       ├── MatchingStep.match()        cosine similarity against gallery
    │       │   ├── _is_active() filter     skip DISABLED/ARCHIVED
    │       │   └── threshold check (0.85)
    │       ├── _final_identity()           threshold + smoother gate
    │       │   ├── UNKNOWN if raw == UNKNOWN
    │       │   ├── UNKNOWN if score < threshold
    │       │   └── PredictionSmoother.update() → majority vote
    │       ├── EventLogger.log()           CSV audit trail
    │       ├── AlertManager.evaluate()     alert on UNKNOWN/low confidence
    │       └── SecurityEngine.evaluate()   severity + decision
    │
    └── _draw_track()                       overlay on frame
        └── cv2.imshow()                    display
```

### C. Video Recognition Flow

Identical to live recognition but reads from a video file instead of a camera.
Produces frame-level CSV report and per-video summary with identity vote counts.

### D. Folder Recognition Flow

```
GEI folder
    │
    ▼
FolderRecognitionPipeline → InferencePipeline
    ├── FeatureExtractionStep.extract(image)
    └── MatchingStep.match(embedding, gallery_features, gallery_labels)
        ⚠ Uses models/gallery (training gallery), NOT models/live_gallery
        ⚠ Does NOT pass metadata → no status filtering
```

---

## 2. Hybrid Upgrade Analysis

### Videos → Gait Gallery ✅ CORRECT
- Videos are processed via `_process_video()` → detection → silhouette → GEI → saved as PNG
- GEI PNGs are fed through `FeatureExtractionStep` using `ByGaitLight` → 256-dim embeddings
- Stored in `models/live_gallery/` with full metadata (status, enabled, embeddings count)

### Photos → Appearance Gallery ✅ CORRECT
- Photos are routed to `_prepare_photo_folder()` → copied to `data/auto_enrollment/photos/`
- Processed by `AppearanceFeatureExtractionStep` (color histogram + edge shape)
- Stored in separate `models/appearance_gallery/` — NOT in `models/live_gallery/`
- Photos do NOT pollute the gait gallery

### Photo/Gait Separation ✅ CONFIRMED
The `AutoEnrollmentService` correctly splits files:
- `_video_files()` → gait path → `models/live_gallery/`
- `_image_files()` → appearance path → `models/appearance_gallery/`

---

## 3. Specific Audit Findings

### Task 4: Are photos entering the gait gallery? ✅ NO
Photos go exclusively through `enroll_appearance_person()` → `AppearanceGalleryUpdater` → `models/appearance_gallery/`. The gait gallery (`models/live_gallery/`) only receives embeddings from video-derived GEI images.

**Evidence:** Live gallery contains `person_test` (220 embeddings) and `demo_person_001` (10 embeddings). Appearance gallery contains 5 identities (147 embeddings) including photo-only sources like Devhan and Isuru.

### Task 5: Are videos correctly converted to GEI and gait embeddings? ✅ YES
`_process_video()` → detection → silhouette → `LiveGEI` (15 frames) → `build()` → GEI PNG → `FeatureExtractionStep.extract()` → ByGaitLight → 256-dim embedding → `GalleryUpdater.add_person()`.

### Task 6: Does live recognition use `models/live_gallery`? ✅ YES
- `LiveRecognitionPipeline.__init__()` defaults to `gallery_dir="models/live_gallery"` (line 30)
- `VideoRecognitionPipeline.__init__()` defaults to `gallery_dir="models/live_gallery"` (line 31)

**⚠ BUT:** `InferencePipeline` (used by `FolderRecognitionPipeline`) defaults to `VectorStore()` which defaults to `models/gallery` — the CASIA-B training gallery. This is a **different** gallery from live_gallery.

### Task 7: Does matching skip DISABLED/ARCHIVED? ✅ YES (for live/video)
`MatchingStep.match()` accepts `metadata` parameter. The `_is_active()` method checks:
- `status == "ACTIVE"` AND `enabled == True`
- Both `live_recognition.py` and `video_recognition.py` pass `self.metadata` to `match()`

**⚠ BUT:** `InferencePipeline.predict()` calls `self.matcher.match()` WITHOUT the `metadata` argument (line 57-61). This means **folder recognition does NOT filter by status**. This is an inconsistency.

### Task 8: Does auto enrollment prevent duplicate embeddings? ✅ PARTIALLY

The `_already_enrolled()` method checks both galleries:
```python
gait_done = not has_videos or person_id in gait_metadata
appearance_done = not has_photos or person_id in appearance_metadata
return gait_done and appearance_done
```
This works correctly — already-enrolled identities are skipped.

**⚠ BUG in `_needs_enrollment()`:** Line 264-267:
```python
if marker.get("fingerprint") != fingerprint:
    return True
return True    # ← ALWAYS returns True regardless
```
The last `return True` bypasses the fingerprint check. If a person already exists in the gallery but the marker says "enrolled" with matching fingerprint, it should return `False` to skip. Instead it **always** re-enrolls. This means the `_already_enrolled()` check on line 247-252 is the only guard, and it works correctly, making this a redundant but harmless bug — the gallery check catches it first.

### Task 9: Can PredictionSmoother keep old identities after UNKNOWN? ✅ FIXED

The new `_final_identity()` method (added to both `live_recognition.py` and `video_recognition.py`) gates the smoother:
```python
if raw_identity == "UNKNOWN":
    return "UNKNOWN"         # Never feeds UNKNOWN to smoother
if score < self.threshold:
    return "UNKNOWN"         # Below threshold → UNKNOWN
return self.smoother.update(track_id, raw_identity)
```

**This means:** UNKNOWN results bypass the smoother entirely. The smoother history only contains ACCEPTED identities. Once a person walks away and returns, the smoother's deque fills with the new identity. The smoother can only "stick" between two KNOWN identities, which is acceptable behavior.

**Residual concern:** If person A leaves and person B takes the same track_id, the smoother history still contains person A's votes. The new person B needs to accumulate majority votes to override. With `history_size=10`, this means B needs ≥6 consecutive above-threshold matches. This is actually desirable — it prevents momentary flickers.

### Task 10: Are thresholds consistent? ⚠ INCONSISTENCY

| Component | Matching Threshold | Alert Threshold | Security Threshold |
|---|---|---|---|
| `LiveRecognitionPipeline` | **0.85** | 0.90 | 0.90 |
| `VideoRecognitionPipeline` | **0.85** | 0.90 | 0.90 |
| `MatchingStep` (default) | **0.75** | — | — |
| `test_live_recognition.py` | **0.85** | — | — |
| `run_video_recognition.py` (default) | **0.75** | 0.80 | 0.80 |
| `FolderRecognitionPipeline` | **0.75** (via MatchingStep default) | — | — |
| `cli.py recognize-folder` | **0.70** | — | — |
| `cli.py recognize-video` | **0.75** | — | — |
| `AlertManager` (default) | — | **0.75** | — |
| `SecurityEngine` (default) | — | — | **0.80** |

**Key issues:**
1. `run_video_recognition.py` script defaults to 0.75, but `VideoRecognitionPipeline.__init__` now defaults to 0.85. The script's defaults override the class defaults when run via CLI.
2. `cli.py recognize-video` passes threshold=0.75 — different from the live/video pipeline's hardened 0.85.
3. `FolderRecognitionPipeline` uses 0.70 (via CLI) or 0.75 (MatchingStep default) — significantly lower.

### Task 11: Is `python cli.py --mode system` production-safe? ⚠ MOSTLY

The current `system` mode runs:
1. `health` → safe read-only system check
2. `live` → starts auto enrollment watcher + live camera recognition

**Production safety assessment:**
- ✅ Health check is non-destructive
- ✅ Auto enrollment watcher runs in background, skips `_disabled_*` folders
- ✅ Already-enrolled identities are skipped
- ⚠ Live camera opens immediately — may fail silently if no camera
- ✅ Graceful shutdown via Q key or CTRL+C

**Verdict:** Safe for demo. The removal of the one-time auto-enroll step (previously step 2/3) means system mode now only does health + live, which is production-appropriate.

### Task 12: Remaining Architecture Risks

1. **`InferencePipeline` uses wrong gallery** — `models/gallery` (CASIA-B training) instead of `models/live_gallery`. Folder recognition matches against training subjects, not enrolled live identities.

2. **`InferencePipeline` does not pass metadata** — No status filtering for folder recognition. DISABLED identities can still match.

3. **`_needs_enrollment()` always returns True** — Line 267: `return True` is unreachable-logic-dead but semantically wrong. The `_already_enrolled()` guard saves it.

4. **`run_auto_enrollment.py` summary prints wrong key** — Line 98: `result.get('embeddings_added')` but the result dict now uses `gait_embeddings_added` and `appearance_embeddings_added`. This will print `None` in the summary.

5. **No gallery hot-reload** — Live recognition loads gallery once at startup. If auto enrollment adds new identities while live recognition is running, they won't be visible until restart.

6. **LiveGEI buffer never cleared** — After recognition, the buffer keeps accumulating. The same GEI frames contribute to multiple recognitions. This is by design (rolling window) but means recognition events are correlated, not independent.

7. **Silhouette quality** — CASIA-B silhouettes are clean lab data. Real-world silhouettes from Otsu thresholding may be noisy. The `FeatureExtractionStep._normalize_to_silhouette()` adds adaptive thresholding and morphological cleanup, which partially mitigates this.

### Task 13: False-Positive Risks

1. **Low threshold in folder/video CLI paths** — 0.70-0.75 is significantly lower than the live pipeline's 0.85. A demo using `recognize-folder` or `recognize-video` via CLI could show more false positives.

2. **CASIA-B gallery in InferencePipeline** — Folder recognition matches against CASIA-B subjects (numeric IDs like "001"). If enrolled live identities have similar gaits to any CASIA-B subject, folder recognition could return a CASIA-B ID.

3. **Appearance gallery is NOT used for matching** — The appearance gallery exists but no recognition pipeline queries it. It's enrollment-only right now. This means photo-enrolled identities (Devhan, Isuru, person01) are invisible to live/video recognition unless they also have video-based gait enrollments.

4. **PredictionSmoother vote history** — If track_id is reused (person A → gap → person B), the smoother needs majority votes to switch. Short appearances of person B may show person A's identity.

### Task 14: Demo Readiness

**✅ The project IS demo-ready for a final-year presentation.**

**Strengths:**
- End-to-end pipeline works: enrollment → live recognition → security alerts
- Hybrid enrollment (video=gait, photo=appearance) correctly separates galleries
- Status management (ACTIVE/DISABLED/ARCHIVED) integrated throughout
- CLI provides all necessary modes for demo workflow
- Unknown-safe smoothing prevents identity leakage
- Security layer + alert system provides audit trail
- Video recognition produces CSV reports for evaluation evidence

**For demo, the recommended workflow is:**
1. `python cli.py --mode health` — show system readiness
2. Drop a walking video into `data/new_input/<person_name>/`
3. `python cli.py --mode auto-enroll` — show enrollment
4. `python cli.py --mode system` — show live recognition
5. `python cli.py --mode set-status --person-id <name> --status DISABLED` — show status management
6. Show that disabled identity no longer matches

### Task 15: Project Completion Percentage

| Component | Completion | Notes |
|---|---|---|
| ByGaitLight CNN architecture | 100% | 3-layer CNN + embedding, works |
| Training pipeline | 100% | Trainer, TripletLoss, dataset loader |
| CASIA-B preprocessing | 100% | Silhouette + GEI generation |
| Training gallery builder | 100% | `build_gallery.py` |
| Feature extraction | 100% | With silhouette normalization |
| Matching + cosine similarity | 100% | With status filtering |
| Live recognition pipeline | 100% | With all subsystems integrated |
| Video recognition pipeline | 100% | With CSV export |
| Folder recognition pipeline | 90% | Works but uses wrong gallery |
| Auto enrollment service | 95% | Hybrid split works, minor print bug |
| Gallery management | 100% | Add, remove, set-status |
| Prediction smoother | 100% | Unknown-safe gating |
| Security engine | 100% | Severity + decision |
| Alert manager | 100% | CSV alert log |
| Event logger | 100% | CSV event log |
| CLI | 100% | 22 modes |
| Appearance feature extraction | 100% | Color + shape features |
| Appearance gallery | 100% | Separate storage |
| Appearance-based recognition | 0% | Gallery exists but no recognition uses it |

**Overall: ~92%**

The core gait recognition pipeline is complete and functional. The appearance gallery infrastructure exists but is not yet used for actual recognition — it's enrollment-only.

### Task 16: Recommended Next Steps (Priority Order)

1. **Fix `_needs_enrollment()` dead return** — Change line 267 from `return True` to `return False`. This is a 1-line fix that makes the fingerprint logic correct.

2. **Fix `run_auto_enrollment.py` summary print** — Update line 98 to print `gait_embeddings_added` and `appearance_embeddings_added` instead of the old `embeddings_added` key.

3. **Fix `InferencePipeline` gallery source** — Change `VectorStore()` default to `VectorStore(gallery_dir="models/live_gallery")` and pass metadata to `match()`. This ensures folder recognition uses the same gallery and status filtering as live/video.

4. **Standardize thresholds** — Consider aligning `run_video_recognition.py` defaults with the pipeline's 0.85 threshold, or make the CLI default match the pipeline default.

5. **Prepare demo script** — Create a simple `docs/demo_guide.md` with the exact step-by-step commands for the final-year demo presentation.
