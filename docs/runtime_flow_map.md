# ARGUS Runtime Flow Map

**Date:** 2026-06-13  
**Scope:** Gait Recognition Module runtime paths

---

## 1. CLI Entry Points

All runtime paths start from `cli.py` which dispatches to 22 mode functions.
Each mode function executes a script via `subprocess.run()`.

```
python cli.py --mode <MODE>
    │
    └── modes[args.mode](args)
        └── run_command([python, scripts/<script>.py, ...])
```

---

## 2. Runtime Flow: System Mode (Production Entry)

```
cli.py --mode system
    │
    ├── [1/2] health(args)
    │   └── scripts/system_check.py                    read-only checks
    │
    └── [2/2] live(args)
        │
        ├── BACKGROUND: scripts/run_auto_enrollment.py --watch
        │   └── AutoEnrollmentService.watch()
        │       └── loop every 5s:
        │           └── enroll_pending()
        │               ├── scan data/new_input/*
        │               ├── skip _disabled_* and .* folders
        │               ├── check _already_enrolled()
        │               ├── VIDEO → _prepare_gait_folder() → enroll_gait_person()
        │               └── PHOTO → _prepare_photo_folder() → enroll_appearance_person()
        │
        └── FOREGROUND: scripts/test_live_recognition.py
            └── LiveRecognitionPipeline(threshold=0.85).run()
                └── camera loop (see detailed flow below)
```

---

## 3. Runtime Flow: Live Recognition Loop

```
LiveRecognitionPipeline.run()
    │
    ▼ ─── FRAME LOOP ──────────────────────────────────────────
    │
    ├── StreamEngine.read()                     webcam frame
    │
    ├── TrackingStep.track(frame)
    │   ├── YOLO(frame, classes=[0])            person detection
    │   └── ByteTrack.update_with_detections()  persistent tracking
    │
    │   FOR EACH (box, track_id):
    │   │
    │   ├── _crop_person(frame, box)            bounding box crop
    │   │
    │   ├── SilhouetteStep.extract_from_crop(crop)
    │   │   ├── cvtColor → grayscale
    │   │   ├── GaussianBlur(5,5)
    │   │   ├── threshold(OTSU)
    │   │   ├── resize(64, 128)
    │   │   └── morphological open + close
    │   │
    │   ├── LiveGEI.add(silhouette)             rolling buffer (max 15)
    │   │
    │   └── IF buffer.ready() AND _should_recognize(track_id):
    │       │
    │       ├── LiveGEI.build()                 mean of 15 silhouettes → GEI
    │       │
    │       ├── _gei_to_embedding(gei)
    │       │   ├── normalize float32 / 255.0
    │       │   ├── unsqueeze(0).unsqueeze(0)   → [1, 1, 128, 64]
    │       │   └── ByGaitLight(tensor)         → 256-dim embedding
    │       │
    │       ├── MatchingStep.match(embedding, features, labels, metadata)
    │       │   ├── _is_active(label, metadata)     filter DISABLED/ARCHIVED
    │       │   ├── L2-normalize query + gallery
    │       │   ├── cosine similarity (dot product)
    │       │   └── IF best_score < 0.85 → UNKNOWN
    │       │
    │       ├── _final_identity(track_id, raw, score)
    │       │   ├── raw == UNKNOWN → "UNKNOWN"
    │       │   ├── score < 0.85   → "UNKNOWN"
    │       │   └── PredictionSmoother.update() → majority vote (deque size 10)
    │       │
    │       ├── EventLogger.log()               → outputs/events/recognition_log.csv
    │       ├── AlertManager.evaluate()         → outputs/events/alerts.csv
    │       └── SecurityEngine.evaluate()       → severity + decision
    │
    ├── _draw_track(frame, box, track_id)       overlay identity + score
    │
    └── cv2.imshow() + waitKey(1)               display
    │
    ▼ ─── END LOOP (Q to quit) ────────────────────────────────
```

---

## 4. Runtime Flow: Video Recognition

```
cli.py --mode recognize-video --video <path>
    │
    └── scripts/run_video_recognition.py
        │
        └── VideoRecognitionPipeline(threshold=0.85).run(video_path)
            │
            ├── cv2.VideoCapture(path)
            │
            ▼ ─── FRAME LOOP (identical to live) ─────────────
            │   ... same as live recognition ...
            │   ... each recognition → appended to results[] ...
            ▼ ─── END LOOP ───────────────────────────────────
            │
            ├── Counter(identities) → most_common_identity
            ├── _save_csv(results, output_path)     optional CSV export
            └── print summary
```

---

## 5. Runtime Flow: Folder Recognition

```
cli.py --mode recognize-folder --folder <path>
    │
    └── scripts/run_folder_recognition.py
        │
        └── FolderRecognitionPipeline().run(folder_path, threshold=0.70)
            │
            ├── InferencePipeline()
            │   ├── FeatureExtractionStep()      ByGaitLight + normalization
            │   ├── MatchingStep()               threshold=0.75
            │   └── VectorStore()                ⚠ models/gallery (training)
            │
            FOR EACH image:
            │   ├── InferencePipeline.predict(image)
            │   │   ├── extract(image) → embedding
            │   │   └── match(embedding, gallery) → identity, score
            │   └── threshold gate → ACCEPTED or REJECTED
            │
            ├── Counter → most_common_identity
            └── optional CSV export
```

---

## 6. Runtime Flow: Auto Enrollment (One-Shot)

```
cli.py --mode auto-enroll
    │
    └── scripts/run_auto_enrollment.py
        │
        └── AutoEnrollmentService.enroll_pending()
            │
            FOR EACH person_folder in data/new_input/:
            │
            ├── skip if name starts with _ or .
            ├── collect all files (images + videos)
            ├── classify → image_files / video_files
            ├── compute fingerprint (file sizes + timestamps)
            │
            ├── _needs_enrollment()?
            │   ├── _already_enrolled()? → False if both galleries have person
            │   └── marker check (⚠ always returns True due to bug)
            │
            ├── IF has video_files:
            │   ├── _prepare_gait_folder()
            │   │   └── _process_video() → detect → silhouette → GEI → PNG
            │   └── enroll_gait_person(gait_folder)
            │       └── FeatureExtractionStep → GalleryUpdater → live_gallery
            │
            └── IF has image_files:
                ├── _prepare_photo_folder() → copy images
                └── enroll_appearance_person(photo_folder)
                    └── AppearanceFeatureExtractionStep → AppearanceGalleryUpdater → appearance_gallery
```

---

## 7. Runtime Flow: Gallery Maintenance

```
cli.py --mode set-status --person-id <id> --status DISABLED
    └── scripts/set_gallery_identity_status.py
        └── VectorStore.load() → update metadata → VectorStore.save()

cli.py --mode remove-identity --person-id <id>
    └── scripts/remove_gallery_identity.py
        └── VectorStore.load() → filter arrays → VectorStore.save()

cli.py --mode clean-numeric-gallery --confirm
    └── scripts/remove_numeric_gallery_identities.py
        └── Remove all numeric-named identities (CASIA-B IDs)
```

---

## 8. Gallery Data Locations

| Gallery | Path | Purpose | Used By |
|---|---|---|---|
| Training | `models/gallery/` | CASIA-B evaluation | `InferencePipeline` (folder recognition) |
| Live | `models/live_gallery/` | Runtime recognition | `LiveRecognitionPipeline`, `VideoRecognitionPipeline` |
| Appearance | `models/appearance_gallery/` | Photo-based fallback | Enrollment only (not queried for recognition) |

---

## 9. Output Locations

| Output | Path | Written By |
|---|---|---|
| Recognition log | `outputs/events/recognition_log.csv` | `EventLogger` |
| Alert log | `outputs/events/alerts.csv` | `AlertManager` |
| Security log | `outputs/security/security_log.csv` | `SecurityLogger` |
| Video CSV report | User-specified `--output` path | `VideoRecognitionPipeline._save_csv()` |
| Folder CSV report | User-specified `--output` path | `FolderRecognitionPipeline._save_csv()` |
| Enrollment markers | `data/new_input/<person>/.argus_enrolled.json` | `AutoEnrollmentService` |
