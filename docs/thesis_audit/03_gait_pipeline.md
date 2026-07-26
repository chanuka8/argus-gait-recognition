# Phase 2 (Part 2) — Gait Recognition Pipeline Analysis

## 3.1 End-to-End Pipeline Stages

### Stage 1: Video/Camera Input

| Property | Detail |
|---|---|
| **Purpose** | Acquire video frames from various sources |
| **Input** | USB webcam, RTSP stream, or video file |
| **Output** | Raw BGR frame (numpy array, H×W×3) |
| **Main Files** | `streaming/stream_engine.py`, `streaming/multi_stream_engine.py` |
| **Algorithm** | OpenCV `VideoCapture` |
| **Parameters** | `device_index: 0`, `width: 640`, `height: 480`, `target_fps: 15` |
| **Verification** | Tested in live and multi-camera modes |
| **Limitations** | RTSP requires network connectivity; webcams require connected hardware |

### Stage 2: Person Detection

| Property | Detail |
|---|---|
| **Purpose** | Detect human persons in each frame |
| **Input** | Raw BGR frame |
| **Output** | Bounding boxes (xyxy format) + confidence scores |
| **Main Files** | `pipeline/steps/detection.py`, `pipeline/steps/tracking.py` |
| **Algorithm** | YOLOv8n (ultralytics) |
| **Parameters** | `confidence: 0.4`, `classes: [0]` (person class only) |
| **Model** | `models/weights/yolov8n.pt` (6.5 MB pre-trained) |
| **Reason** | Real-time person-only detection; YOLOv8n is the smallest and fastest variant |
| **Alternatives** | YOLOv8s/m/l, Faster R-CNN, SSD |
| **Advantages** | Sub-10ms inference, high recall for persons |
| **Limitations** | May miss occluded or distant persons; pre-trained on COCO, not fine-tuned |
| **Verified by tests** | `tests/test_detector.py` |

### Stage 3: Multi-Object Tracking

| Property | Detail |
|---|---|
| **Purpose** | Assign persistent track IDs to detected persons across frames |
| **Input** | Detections (bounding boxes + confidences) |
| **Output** | Detections with `tracker_id` assigned |
| **Main Files** | `pipeline/steps/tracking.py` |
| **Algorithm** | ByteTrack (via `supervision.ByteTrack`) |
| **Mathematical Operation** | IoU-based association with Kalman filter state prediction |
| **Reason** | Handles occlusions via second-round low-confidence matching |
| **Advantages** | Robust track assignment; handles partial occlusions |
| **Limitations** | Track ID reassignment after extended occlusions; no appearance features used |
| **Verified by tests** | `tests/test_tracker.py` |

### Stage 4: Bounding Box Stabilization

| Property | Detail |
|---|---|
| **Purpose** | Smooth bounding box coordinates to reduce detection jitter |
| **Input** | Raw detections with track IDs |
| **Output** | Stabilized bounding boxes per track ID |
| **Main Files** | `utils/box_stabilizer.py` |
| **Algorithm** | Exponential Moving Average (EMA) |
| **Parameters** | `ema_alpha: 0.35`, `min_iou_keep: 0.25`, `max_missed_frames: 8`, `max_jump_ratio: 0.35` |
| **Mathematical Operation** | `stable_box = α × new_box + (1−α) × old_box` |
| **Reason** | Eliminates detection flicker for silhouette quality |
| **Limitations** | Introduces slight lag in position tracking |

### Stage 5: Person Crop Extraction

| Property | Detail |
|---|---|
| **Purpose** | Crop the detected person region from the frame |
| **Input** | Frame + stabilized bounding box |
| **Output** | Cropped BGR image (person region) |
| **Main Files** | `pipeline/live_recognition.py::_crop_person()` |
| **Algorithm** | Array slicing with boundary clamping |
| **Limitations** | Quality depends on detection accuracy; may include background |

### Stage 6: Silhouette Extraction

| Property | Detail |
|---|---|
| **Purpose** | Extract binary human silhouette from person crop |
| **Input** | BGR person crop |
| **Output** | 64×128 binary silhouette image (uint8: 0 or 255) |
| **Main Files** | `pipeline/steps/silhouette_step.py` |
| **Algorithm** | Grayscale → Gaussian Blur (5×5) → Otsu Thresholding → Morphological Open (3×3, 1 iter) → Morphological Close (3×3, 2 iter) → Contour Detection → Largest Contour Selection → Aspect Ratio Filtering → Canvas Placement |
| **Parameters** | `target_size: (64, 128)`, body height ~85% of canvas, aspect ratio filter: 1.2–6.0 |
| **Mathematical Operation** | Otsu's method minimizes intra-class variance of pixel intensities |
| **Reason** | Separates foreground (person) from background without deep learning |
| **Alternatives** | Semantic segmentation (DeepLabV3), background subtraction (MOG2) |
| **Advantages** | No additional model required; fast execution |
| **Limitations** | Poor quality under complex backgrounds, shadows, illumination changes; relies on good contrast between person and background |
| **Failure Conditions** | Contour area < 50px, area > 95% of crop, width < 5px, height < 15px, abnormal aspect ratio |
| **Verified by tests** | `tests/test_silhouette.py` |

### Stage 7: Gait Energy Image (GEI) Generation

| Property | Detail |
|---|---|
| **Purpose** | Aggregate temporal walking pattern into single representative image |
| **Input** | Sequence of binary silhouette frames |
| **Output** | 64×128 GEI image (uint8) |
| **Main Files** | `pipeline/steps/live_gei.py` |
| **Algorithm** | Rolling-window arithmetic mean of binarized silhouettes |
| **Mathematical Operation** | `GEI(x,y) = (1/N) × Σᵢ Bᵢ(x,y)`, where Bᵢ are binary silhouettes |
| **Parameters** | `max_frames: 15`, `min_frames: 10` (default), `size: (64, 128)` |
| **Reason** | GEI captures average gait pattern, reducing noise from individual frames |
| **Alternatives** | Gait Entropy Image, Period Energy Image, pose-based representations |
| **Advantages** | Simple, compact representation; well-studied in literature |
| **Limitations** | Loses fine-grained temporal dynamics; affected by silhouette quality; requires minimum ~10 walking frames |
| **Verified by tests** | `tests/test_gei_stream.py` |

### Stage 8: CNN Embedding Generation

| Property | Detail |
|---|---|
| **Purpose** | Project GEI into compact embedding vector for similarity matching |
| **Input** | 64×128 GEI image normalized to [0, 1] |
| **Output** | 256-dimensional L2-normalized embedding vector |
| **Main Files** | `models/architectures/bygait_light.py`, `pipeline/live_recognition.py::_gei_to_embedding()` |
| **Algorithm** | 3-block CNN: Conv2d→BN→ReLU→MaxPool (×3), AdaptiveAvgPool, Linear, L2-Normalize |
| **Mathematical Operation** | `emb = L2_normalize(Linear(AvgPool(Conv3(Conv2(Conv1(x))))))` |
| **Parameters** | Input: 1×1×128×64 tensor, Output: 256-dim vector |
| **Reason** | Lightweight architecture suitable for CPU deployment |
| **Limitations** | Limited representational capacity compared to deeper networks |
| **Verified by tests** | Implicitly tested through evaluation pipeline |

### Stage 9: Gallery Search

| Property | Detail |
|---|---|
| **Purpose** | Compare query embedding against enrolled gallery templates |
| **Input** | 256-dim query embedding + gallery matrix |
| **Output** | Ranked list of (identity, similarity_score) tuples |
| **Main Files** | `pipeline/steps/matching_step.py` |
| **Algorithm** | Cosine similarity via matrix dot product |
| **Mathematical Operation** | `similarity(q, g) = (q · g) / (‖q‖ × ‖g‖)` — since both are L2-normalized, this simplifies to `q · g` |
| **Parameters** | Gallery features filtered by metadata `status == "ACTIVE"` |
| **Reason** | Cosine similarity is the standard metric for embedding-based recognition |
| **Alternatives** | Euclidean distance, Mahalanobis distance |
| **Advantages** | Scale-invariant; efficient via matrix multiplication |
| **Limitations** | Assumes embedding space is well-separated; sensitive to embedding quality |

### Stage 10: Adaptive Decision Policy

| Property | Detail |
|---|---|
| **Purpose** | Make tiered identity decisions based on confidence level |
| **Input** | Best match identity + score + embedding |
| **Output** | (identity, score, decision_level) |
| **Main Files** | `pipeline/live_recognition.py::_adaptive_decision()` |
| **Algorithm** | Multi-tier threshold + centroid verification for mid-range scores |
| **Decision Levels** | CONFIRMED_MATCH (≥0.92), VERIFIED_MATCH (0.85-0.92 + centroid agrees), REVIEW_REQUIRED (0.85-0.92 + centroid disagrees), LOW_CONFIDENCE (0.70-0.85), UNKNOWN_PERSON (<0.70) |
| **Centroid Verification** | Builds per-identity centroids, applies margin rule + top-k majority voting |
| **Parameters** | `confirmed_threshold: 0.92`, `verify_low: 0.85`, `verify_high: 0.92`, `margin: 0.05`, `top_k: 5` |
| **Reason** | Reduces false positives by requiring stronger evidence for mid-range scores |
| **Limitations** | Threshold values are manually selected, not learned |

### Stage 11: Prediction Smoothing

| Property | Detail |
|---|---|
| **Purpose** | Stabilize identity predictions over time using temporal voting |
| **Input** | Track ID + raw prediction + score |
| **Output** | Stabilized identity label |
| **Main Files** | `utils/prediction_smoother.py` |
| **Algorithm** | Sliding-window majority voting with confirmation persistence |
| **Parameters** | `history_size: 10`, `min_stable_votes: 3` |
| **Logic** | Count votes for each prediction in the last 10 frames (excluding UNKNOWN below threshold); if best gets ≥3 votes, confirm; otherwise keep previously confirmed identity if it has ≥1 vote |
| **Reason** | Prevents identity flickering between frames |
| **Limitations** | Introduces latency; may persist incorrect identity if initial votes were wrong |

### Stage 12: Security Evaluation & Audit Logging

| Property | Detail |
|---|---|
| **Purpose** | Classify recognition events by severity and log to audit trail |
| **Input** | Track ID, identity, score, camera ID |
| **Output** | Security decision (ALLOW/SECURITY_ALERT/REVIEW_REQUIRED) + CSV log entry |
| **Main Files** | `security_layer/security_engine.py`, `security_layer/security_logger.py` |
| **Algorithm** | Rule-based: UNKNOWN → SECURITY_ALERT, score < threshold → REVIEW_REQUIRED |
| **Log Fields** | timestamp, track_id, identity, score, severity, decision, camera_id |
| **Storage** | Thread-safe CSV at `outputs/security_logs/security_events.csv` |
| **Verified by tests** | `tests/test_audit_verification.py` |

### Stage 13: Detection Reporting

| Property | Detail |
|---|---|
| **Purpose** | Generate structured detection reports with snapshots |
| **Input** | Camera ID, location, track ID, identity, status, score, bounding box, frame |
| **Output** | JSONL + CSV reports, cropped snapshots |
| **Main Files** | `utils/detection_reporter.py` |
| **Parameters** | `output_dir: outputs/detection_reports`, `cooldown_seconds: 10`, report filtering by status |

## 3.2 Pipeline Summary Table

| Stage | Input | Processing Method | Output | Main File | Verification Status | Limitations |
|---|---|---|---|---|---|---|
| 1. Video Input | Camera/file | OpenCV VideoCapture | BGR frame | `streaming/stream_engine.py` | Implemented | Requires hardware/network |
| 2. Person Detection | BGR frame | YOLOv8n (class=0) | Bounding boxes | `pipeline/steps/tracking.py` | **Verified** (test_detector.py) | Pre-trained, not fine-tuned |
| 3. Multi-Object Tracking | Detections | ByteTrack (IoU + Kalman) | Track IDs | `pipeline/steps/tracking.py` | **Verified** (test_tracker.py) | ID reassignment after occlusion |
| 4. Box Stabilization | Raw boxes | EMA smoothing | Stable boxes | `utils/box_stabilizer.py` | Implemented | Slight positional lag |
| 5. Person Crop | Frame + box | Array slicing | BGR crop | `pipeline/live_recognition.py` | Implemented | Background included |
| 6. Silhouette Extraction | BGR crop | Otsu + morphology + contour | 64×128 binary mask | `pipeline/steps/silhouette_step.py` | **Verified** (test_silhouette.py) | Poor in complex backgrounds |
| 7. GEI Generation | Silhouette sequence | Rolling window mean (N=15) | 64×128 GEI | `pipeline/steps/live_gei.py` | **Verified** (test_gei_stream.py) | Needs ≥10 walking frames |
| 8. CNN Embedding | GEI tensor | ByGaitLight forward pass | 256-dim vector | `models/architectures/bygait_light.py` | **Verified** (eval reports) | Limited model capacity |
| 9. Gallery Search | Query embedding | Cosine similarity (dot product) | Ranked matches | `pipeline/steps/matching_step.py` | **Verified** (eval reports) | Sensitive to embedding quality |
| 10. Adaptive Decision | Match results | Multi-tier threshold + centroid | Decision label | `pipeline/live_recognition.py` | Implemented | Manual threshold selection |
| 11. Prediction Smoothing | Raw predictions | Voting (window=10, votes≥3) | Stable identity | `utils/prediction_smoother.py` | Implemented | Latency; may persist errors |
| 12. Security Logging | Decision results | Rule-based severity + CSV | Audit log | `security_layer/security_engine.py` | **Verified** (test_audit) | No encryption or tamper-proofing |
| 13. Detection Reporting | All results | JSONL/CSV + snapshots | Report files | `utils/detection_reporter.py` | Implemented | Disk space dependent |
