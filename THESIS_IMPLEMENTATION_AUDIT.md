# ARGUS AI Thesis Implementation Audit

## Chapter 1

### Background
- **File Path:** [README.md](file:///e:/ARGUS_AI/README.md#L3-L13)
- **Class / Function Name:** Repository Metadata & Visual Summary
- **Concise Explanation:** The system is established as a research prototype for missing-person identification using Gait Energy Images (GEI), Convolutional Neural Network (CNN) embeddings, and CCTV-style multi-camera analysis.
- **Implementation Evidence:** Line 5 states: *"A gait recognition module for missing-person identification using GEI, CNN embeddings, and CCTV-style multi-camera analysis."* Line 13 badges the system as `Status-Research--Prototype`.

### Research Problem
- **File Path:** [README.md](file:///e:/ARGUS_AI/README.md#L155-L163)
- **Class / Function Name:** System Limitations Section
- **Concise Explanation:** Traditional biometric systems (such as face or fingerprint recognition) fail under non-cooperative, unconstrained CCTV scenarios with distance, low resolution, clothing changes, dynamic backgrounds, and varied perspective angles.
- **Implementation Evidence:** Lines 158-162 document real-world surveillance constraints: *"The system requires a complete side or diagonal walking profile... Shadows, flickering illumination, and complex dynamic backgrounds degrade background subtraction... Target accuracy is heavily affected by clothing changes (e.g., long coats), carrying conditions (e.g., backpacks), and viewing angle deviations."*

### Motivation
- **File Path:** [intelligence/missing_person_workflow.py](file:///e:/ARGUS_AI/intelligence/missing_person_workflow.py#L10-L74)
- **Class / Function Name:** `MissingPersonWorkflow`
- **Concise Explanation:** Automating target identification and alerting across multiple CCTV feeds for missing persons without manual video inspection.
- **Implementation Evidence:** Method `process_match` evaluates incoming gait matches against a watchlist (`self._target_watchlist`), emitting a `MISSING_PERSON_MATCH` event with alert throttling (`self.cooldown_seconds = 60.0`).

### Research Gap
- **File Path:** [evaluation/open_set_evaluator.py](file:///e:/ARGUS_AI/evaluation/open_set_evaluator.py#L18-L68)
- **Class / Function Name:** `SubjectDisjointOpenSetEvaluator`
- **Concise Explanation:** Existing benchmarks rely on closed-set assumptions (where all probe subjects exist in the gallery) and allow data leakage between train/val/test splits. ARGUS addresses this by evaluating open-set recognition with strict zero-leakage subject-disjoint partitions.
- **Implementation Evidence:** Method `evaluate_open_set_protocol` partitions test subjects into `known_test_subjects` and `unknown_test_subjects`, validating disjointness via `assert_gallery_probe_disjointness` in [leakage_validator.py](file:///e:/ARGUS_AI/evaluation/leakage_validator.py#L34-L68).

### Objectives
- **File Path:** [cli.py](file:///e:/ARGUS_AI/cli.py#L46-L245)
- **Class / Function Name:** CLI Command Handlers (`health`, `preprocess`, `train`, `build_gallery`, `evaluate`, `benchmark`, `auto_enroll`, `recognize_video`)
- **Concise Explanation:** 
  1. Process CASIA-B dataset silhouettes and construct standardized GEI representations.
  2. Train a lightweight CNN (`ByGaitLight`) using metric learning loss functions.
  3. Maintain an active vector database for template storage and real-time cosine similarity lookup.
  4. Perform real-time, multi-camera, and video file identification with automated reporting and security verification.
- **Implementation Evidence:** `cli.py` exposes distinct commands mapping to scripts: `preprocess_casia.py`, `train_model.py`, `build_gallery.py`, `evaluate_model.py`, `run_auto_enrollment.py`, `run_video_recognition.py`.

### Scope
- **File Path:** [configs/system.yaml](file:///e:/ARGUS_AI/configs/system.yaml#L3-L22)
- **Class / Function Name:** `camera` configuration section
- **Concise Explanation:** Real-time processing of USB webcams, pre-recorded video files (`.mp4`, `.avi`), and RTSP network CCTV streams at standard resolutions ($640 \times 480$) with rolling GEI accumulation ($N=15$ frames).
- **Implementation Evidence:** Lines 3-22 configure camera types (`rtsp`, `usb`, `file`), resolution ($640 \times 480$), target FPS ($15$), and frame buffer sizes (`max_queue_size: 10`).

### Contributions
- **File Path:** [models/architectures/bygait_light.py](file:///e:/ARGUS_AI/models/architectures/bygait_light.py#L6-L71)
- **Class / Function Name:** `ByGaitLight`
- **Concise Explanation:** 
  1. A lightweight 2D CNN architecture tailored for fast CPU/edge gait feature extraction ($256$-dimensional normalized embeddings).
  2. A zero-leakage evaluation framework supporting subject-disjoint, cross-view ($11 \times 11$ angle matrices), and open-set protocols.
  3. A multi-threaded multi-camera surveillance architecture with box stabilization, prediction smoothing, and automated CSV/JSONL detection reporting.
- **Implementation Evidence:** `ByGaitLight` produces L2-normalized 256D embeddings using 3 Conv2D blocks and AdaptiveAvgPool2d in 71 lines of code.

---

## Chapter 2

### Algorithms
- **File Path:** [pipeline/silhouette/extractor.py](file:///e:/ARGUS_AI/pipeline/silhouette/extractor.py#L20-L80)
- **Class / Function Name:** `SilhouetteExtractor`
- **Concise Explanation:** MOG2 Background Subtraction (`cv2.createBackgroundSubtractorMOG2`) followed by morphological opening/closing operations, bounding box cropping, and thresholding to produce binary silhouette masks ($64 \times 128$).
- **Implementation Evidence:** `cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)` extracts raw foreground, followed by `cv2.morphologyEx` and `cv2.threshold(..., 127, 255, cv2.THRESH_BINARY)`.

- **File Path:** [pipeline/steps/live_gei.py](file:///e:/ARGUS_AI/pipeline/steps/live_gei.py#L10-L45)
- **Class / Function Name:** `LiveGEI`
- **Concise Explanation:** Temporal sliding-window GEI aggregation averaging $N$ ($15$) consecutive binary silhouette frames.
- **Implementation Evidence:** Computes average frame: `gei = np.mean(self.buffer, axis=0).astype(np.uint8)`.

- **File Path:** [pipeline/steps/matching_step.py](file:///e:/ARGUS_AI/pipeline/steps/matching_step.py#L15-L120)
- **Class / Function Name:** `MatchingStep`
- **Concise Explanation:** Cosine similarity evaluation between normalized query embedding vector and stored gallery matrix, mapped through a 4-tier classification policy: `CONFIRMED`, `VERIFIED`, `LOW_CONFIDENCE`, and `UNKNOWN`.
- **Implementation Evidence:** `scores = np.dot(gallery_features, query_feature)`. Tier classification logic checks thresholds: `confirmed_threshold` ($0.92$), `verify_low` ($0.85$), `low_confidence_low` ($0.70$).

### Models
- **File Path:** [models/architectures/bygait_light.py](file:///e:/ARGUS_AI/models/architectures/bygait_light.py#L6-L71)
- **Class / Function Name:** `ByGaitLight`
- **Concise Explanation:** Custom 2D Convolutional Neural Network consisting of 3 convolutional blocks ($1 \to 32 \to 64 \to 128$), Batch Normalization, ReLU activations, Max Pooling ($2 \times 2$), Adaptive Average Pooling ($1 \times 1$), Linear projection to $256$ dimensions, and L2 normalization ($F.normalize(x, p=2, dim=1)$).
- **Implementation Evidence:**
  ```python
  self.features = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
      nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
      nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
  )
  self.pool = nn.AdaptiveAvgPool2d((1, 1))
  self.embedding = nn.Linear(128, embedding_dim)
  ```

- **File Path:** [models/architectures/losses.py](file:///e:/ARGUS_AI/models/architectures/losses.py#L7-L168)
- **Class / Function Name:** `BatchHardTripletLoss`, `JointGaitLoss`, `ArcMarginProduct`
- **Concise Explanation:** Combined optimization using Cross-Entropy Loss, Batch-Hard Triplet Loss (Euclidean distance with margin $\alpha=0.3$), and optional ArcFace additive angular margin ($s=30.0, m=0.50$).
- **Implementation Evidence:** `JointGaitLoss` returns `total = ce + self.triplet_weight * triplet`. `ArcMarginProduct` applies $\cos(\theta + m)$ logic to feature logits.

### Frameworks
- **File Path:** [requirements.txt](file:///e:/ARGUS_AI/requirements.txt#L1-L15)
- **Class / Function Name:** Dependency Manifest
- **Concise Explanation:** The software stack leverages Python 3.11, PyTorch 2.1+, OpenCV-Python, Ultralytics YOLOv8, NumPy, PyYAML, and FastAPI.
- **Implementation Evidence:** `torch>=2.0.0`, `opencv-python>=4.8.0`, `ultralytics>=8.0.0`, `numpy>=1.24.0`, `fastapi>=0.100.0`.

### Related Implementation
- **File Path:** [pipeline/steps/tracking.py](file:///e:/ARGUS_AI/pipeline/steps/tracking.py#L10-L40)
- **Class / Function Name:** `TrackingStep`
- **Concise Explanation:** Multi-object tracking integrating YOLOv8 person detector (`yolov8n.pt`) with ByteTrack (`bytetrack.yaml`) to maintain consistent track IDs across frames.
- **Implementation Evidence:** Invokes `self.model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0])`.

### Design Choices
- **File Path:** [utils/box_stabilizer.py](file:///e:/ARGUS_AI/utils/box_stabilizer.py#L33-L169)
- **Class / Function Name:** `BoxStabilizer`
- **Concise Explanation:** Bounding box stabilization using Exponential Moving Average (EMA, $\alpha=0.35$), IoU sanity checks ($\text{IoU} \ge 0.25$), and jump ratio bounds to prevent silhouette spatial jitter across adjacent frames.
- **Implementation Evidence:** `updated_box = self.alpha * raw_box + (1.0 - self.alpha) * prev_stable`.

- **File Path:** [utils/prediction_smoother.py](file:///e:/ARGUS_AI/utils/prediction_smoother.py#L5-L52)
- **Class / Function Name:** `PredictionSmoother`
- **Concise Explanation:** Temporal voting mechanism maintaining a sliding window queue ($N=10$) requiring at least $3$ stable votes before confirming an identity label.
- **Implementation Evidence:** `if count >= self.min_stable_votes: self.confirmed_identities[track_id] = best_pred`.

### Limitations
- **File Path:** [README.md](file:///e:/ARGUS_AI/README.md#L155-L164)
- **Class / Function Name:** System Limitations
- **Concise Explanation:** Requires side or diagonal profile walking movement to form meaningful GEIs. Stationary upper-body webcam feeds do not capture arm/leg gait dynamics. Environmental background noise and clothing variations degrade background subtraction.
- **Implementation Evidence:** Documented in section 9 of `README.md`.

### Comparable Techniques
- **File Path:** [evaluation/cross_view_evaluator.py](file:///e:/ARGUS_AI/evaluation/cross_view_evaluator.py#L19-L187)
- **Class / Function Name:** `SubjectDisjointCrossViewEvaluator`
- **Concise Explanation:** Benchmarks Rank-1, Rank-5, Rank-10 recognition accuracy across all $11 \times 11$ gallery-probe viewing angle combinations ($000^\circ$ to $180^\circ$ in $18^\circ$ steps) on CASIA-B.
- **Implementation Evidence:** Evaluates 121 pairs using `ALL_ANGLES = ["000", "018", "036", "054", "072", "090", "108", "126", "144", "162", "180"]`.

---

## Chapter 3

### Complete Processing Pipeline
- **File Path:** [pipeline/live_recognition.py](file:///e:/ARGUS_AI/pipeline/live_recognition.py#L320-L760)
- **Class / Function Name:** `LiveRecognitionPipeline.process_frame`
- **Concise Explanation:** Ingests video frames, detects human targets (YOLOv8), updates track IDs (ByteTrack), stabilizes boxes (`BoxStabilizer`), segments silhouettes (`SilhouetteStep`), aggregates rolling GEIs (`LiveGEI`), extracts CNN embeddings (`ByGaitLight`), matches against VectorStore gallery (`MatchingStep`), smoothes predictions (`PredictionSmoother`), enforces security policies (`SecurityEngine`), and emits detection reports (`DetectionReporter`).
- **Implementation Evidence:** End-to-end execution path implemented in lines 320 to 760 of `live_recognition.py`.

### Module Interactions
- **File Path:** [services/argus_service.py](file:///e:/ARGUS_AI/services/argus_service.py#L69-L150)
- **Class / Function Name:** `ArgusService._recognition_worker`
- **Concise Explanation:** Background thread loop linking `CameraService`, `PersonDetector`, `PersonTracker`, `SilhouetteExtractor`, `StreamGEIBuilder`, `LiveRecognitionPipeline`, and `DetectionReporter`.
- **Implementation Evidence:** Orchestrates frame retrieval from camera, detection, tracking, silhouette extraction, GEI compilation, matching, and rendering.

### Folder Architecture
```
ARGUS_AI/
├── api/             # FastAPI REST endpoints & Pydantic schemas
├── configs/         # System, camera, and inference YAML configuration files
├── core/            # System context, logging, orchestrator, and health check
├── enrollment/      # Automated folder watching & gallery auto-enrollment
├── evaluation/      # Subject-disjoint, cross-view & open-set evaluation engines
├── events/          # Event bus, event types, and dispatcher
├── intelligence/    # ReID cache, cross-camera tracking & missing person workflow
├── models/          # Neural network architectures (ByGaitLight, loss functions)
├── monitoring/      # System logging, watchdog, and health monitor
├── pipeline/        # Live, video, multi-camera, and folder recognition pipelines
├── preprocessing/   # CASIA-B dataset extractor and GEI compilation tools
├── security_layer/  # Security engine and CSV audit logger
├── services/        # Production background service runner & camera manager
├── storage/         # Vector store matrix manager & evidence snapshot saver
├── streaming/       # Thread-safe multi-camera capture engine & worker pool
├── training/        # Trainer, loss functions, dataloaders & checkpointer
└── utils/           # Box stabilizer, prediction smoother, renderer & reporter
```

### Mathematical Operations
1. **Gait Energy Image (GEI) Computation:**
   $$\bar{G}(x, y) = \frac{1}{N} \sum_{t=1}^{N} B(x, y, t)$$
   where $B(x,y,t)$ is the binary silhouette mask at frame $t$, and $N=15$.
   - *Code:* [pipeline/steps/live_gei.py](file:///e:/ARGUS_AI/pipeline/steps/live_gei.py#L35) (`gei = np.mean(self.buffer, axis=0).astype(np.uint8)`)

2. **L2 Embedding Normalization:**
   $$\hat{\mathbf{f}} = \frac{\mathbf{f}}{\|\mathbf{f}\|_2 + \epsilon} = \frac{\mathbf{f}}{\sqrt{\sum_{i=1}^{d} f_i^2} + 10^{-8}}$$
   - *Code:* [models/architectures/bygait_light.py](file:///e:/ARGUS_AI/models/architectures/bygait_light.py#L65) (`F.normalize(x, p=2, dim=1)`)

3. **Cosine Similarity Matching:**
   $$S(\mathbf{q}, \mathbf{g}_i) = \mathbf{q} \cdot \mathbf{g}_i = \sum_{j=1}^{d} q_j g_{i,j}$$
   - *Code:* [pipeline/steps/matching_step.py](file:///e:/ARGUS_AI/pipeline/steps/matching_step.py#L55) (`scores = np.dot(gallery_features, query_feature)`)

4. **Batch-Hard Triplet Loss:**
   $$\mathcal{L}_{\text{triplet}} = \frac{1}{|B|} \sum_{i \in B} \max\left(0, \max_{p: y_p = y_i} D(\mathbf{a}_i, \mathbf{p}) - \min_{n: y_n \neq y_i} D(\mathbf{a}_i, \mathbf{n}) + \alpha\right)$$
   - *Code:* [models/architectures/losses.py](file:///e:/ARGUS_AI/models/architectures/losses.py#L75-L80)

5. **Bounding Box Exponential Moving Average (EMA):**
   $$\mathbf{b}_{\text{stable}}^{(t)} = \alpha \mathbf{b}_{\text{raw}}^{(t)} + (1 - \alpha) \mathbf{b}_{\text{stable}}^{(t-1)}, \quad \alpha = 0.35$$
   - *Code:* [utils/box_stabilizer.py](file:///e:/ARGUS_AI/utils/box_stabilizer.py#L45) (`self.alpha = config.get("ema_alpha", 0.35)`)

### Configuration Files
- **[configs/inference.yaml](file:///e:/ARGUS_AI/configs/inference.yaml):** Thresholds (`live_threshold: 0.85`, `security_threshold: 0.90`), matching policy, crowd control limits, box stability parameters, display colors, reporting rules.
- **[configs/system.yaml](file:///e:/ARGUS_AI/configs/system.yaml):** Camera hardware bindings (`type: usb`, resolution $640 \times 480$, target FPS $15$), logging rotation settings, watchdog interval ($30$s).
- **[configs/cameras.yaml](file:///e:/ARGUS_AI/configs/cameras.yaml):** RTSP stream mappings for multi-camera execution (`camera_01`, `camera_02`, `camera_03`).

### Runtime Workflow
- **File Path:** [cli.py](file:///e:/ARGUS_AI/cli.py#L825-L950)
- **Class / Function Name:** `main` CLI parser dispatcher
- **Concise Explanation:** Selects execution mode from command line arguments (`production-test`, `system`, `multi-camera`, `recognize-video`, `research-eval`), instantiating corresponding pipeline managers.
- **Implementation Evidence:** Dispatcher routes `--mode` values to dedicated execution functions.

### Data Flow
```
[Video / RTSP Stream]
       │
       ▼
[CameraStream (Thread-safe Queue)]
       │
       ▼
[PersonDetector (YOLOv8)] ──► Bounding Boxes
       │
       ▼
[PersonTracker (ByteTrack)] ──► Track IDs
       │
       ▼
[BoxStabilizer (EMA & IoU)] ──► Stabilized Bboxes
       │
       ▼
[SilhouetteStep (MOG2)] ──► Binary Silhouette (64x128)
       │
       ▼
[LiveGEI (Sliding Window N=15)] ──► GEI Image
       │
       ▼
[ByGaitLight (CNN Extractor)] ──► 256D Normalized Embedding Vector
       │
       ▼
[MatchingStep (Cosine Similarity)] ──► Top Matches & Decision Policy
       │
       ▼
[PredictionSmoother (Majority Voting)] ──► Confirmed Identity Label
       │
       ▼
[SecurityEngine & Reporter] ──► CSV/JSONL Logs & Snapshot Output
```

### System Architecture
```mermaid
graph TD
    SubGraph1[Capture Layer]
    CamUSB[USB WebCam] --> StreamEng[StreamEngine / CameraStream]
    CamRTSP[RTSP CCTV Stream] --> StreamEng
    VideoFile[Media File] --> StreamEng

    SubGraph2[Processing Pipeline]
    StreamEng --> Detect[YOLOv8 Person Detector]
    Detect --> Track[ByteTrack Tracker]
    Track --> Stabilize[BoxStabilizer EMA]
    Stabilize --> Silh[MOG2 Silhouette Extractor]
    Silh --> GEI[LiveGEI Window Accumulator]

    SubGraph3[Biometric Core]
    GEI --> CNN[ByGaitLight 2D-CNN]
    CNN --> VecStore[VectorStore Matrix Storage]
    VecStore --> Matcher[MatchingStep Cosine Matcher]

    SubGraph4[Intelligence & Output]
    Matcher --> Smooth[PredictionSmoother Voting]
    Smooth --> SecEng[SecurityEngine Policy]
    SecEng --> Report[DetectionReporter CSV/JSONL]
    SecEng --> Renderer[DetectionDisplayRenderer GUI Overlay]
```

### Sequence Diagrams (Mermaid)
```mermaid
sequenceDiagram
    autonumber
    participant Cam as CameraStream
    participant Pipe as LiveRecognitionPipeline
    participant Det as PersonDetector (YOLOv8)
    participant Trk as PersonTracker (ByteTrack)
    participant GEI as LiveGEI
    participant CNN as ByGaitLight Model
    participant Match as MatchingStep
    participant Rep as DetectionReporter

    Cam->>Pipe: Get next frame
    Pipe->>Det: detect(frame)
    Det-->>Pipe: raw bounding boxes
    Pipe->>Trk: update(raw_boxes)
    Trk-->>Pipe: tracked objects (track_id, bbox)
    Pipe->>GEI: add_silhouette(track_id, crop)
    alt GEI Window Ready (15 frames)
        GEI-->>Pipe: compiled GEI image
        Pipe->>CNN: _gei_to_embedding(gei)
        CNN-->>Pipe: 256D normalized vector
        Pipe->>Match: top_k_matches(vector)
        Match-->>Pipe: identity, similarity_score
        Pipe->>Rep: report(track_id, identity, score)
    end
```

### Component Diagrams (Mermaid)
```mermaid
graph LR
    subgraph Frontend Services
        CLI[cli.py Gateway]
        API[FastAPI Web Server]
    end

    subgraph Core Execution Engine
        ArgusSvc[ArgusService]
        MultiCam[MultiStreamEngine]
        WorkerPool[WorkerPool]
    end

    subgraph Algorithmic Pipeline
        YOLO[YOLOv8 Detector]
        ByteTrk[ByteTrack Tracker]
        MOG2[MOG2 Silhouette Extractor]
        GEIComp[StreamGEIBuilder]
        ByGait[ByGaitLight CNN]
        CosMatch[MatchingStep]
    end

    subgraph Storage & Audit
        VStore[VectorStore Flat Matrix]
        EvidMgr[EvidenceManager]
        SecLog[SecurityLogger]
    end

    CLI --> ArgusSvc
    API --> ArgusSvc
    ArgusSvc --> MultiCam
    MultiCam --> WorkerPool
    WorkerPool --> YOLO
    YOLO --> ByteTrk
    ByteTrk --> MOG2
    MOG2 --> GEIComp
    GEIComp --> ByGait
    ByGait --> CosMatch
    CosMatch --> VStore
    CosMatch --> EvidMgr
    CosMatch --> SecLog
```

---

## Chapter 4

### Experimental Setup
- **File Path:** [configs/subject_split.json](file:///e:/ARGUS_AI/configs/subject_split.json)
- **Class / Function Name:** Subject Split Configuration Manifest
- **Concise Explanation:** Evaluated on the CASIA-B Gait Dataset containing 124 subjects. Partitioned into strict disjoint subsets: 74 training subjects (001–074), 25 validation subjects (075–099), and 25 test subjects (100–124).
- **Implementation Evidence:** File structure defines arrays: `"train_subjects": ["001", ..., "074"]`, `"val_subjects": ["075", ..., "099"]`, `"test_subjects": ["100", ..., "124"]`.

### Datasets
- **File Path:** [preprocessing/casia_extractor.py](file:///e:/ARGUS_AI/preprocessing/casia_extractor.py#L15-L80)
- **Class / Function Name:** `CASIABExtractor`
- **Concise Explanation:** Parses CASIA-B directory trees structured by subject ID, walking condition (`nm-01` to `nm-06` normal, `bg-01` to `bg-02` bag, `cl-01` to `cl-02` clothing), and 11 viewing angles ($000^\circ$ to $180^\circ$).
- **Implementation Evidence:** Parses sequence strings matching pattern `r"(\d{3})-(nm|bg|cl)-(\d{2})-(\d{3})"` in CASIA-B file paths.

### Evaluation Metrics
- **File Path:** [evaluation/metrics.py](file:///e:/ARGUS_AI/evaluation/metrics.py#L25-L187)
- **Class / Function Name:** `compute_rank_k_accuracies`, `compute_cmc_curve`, `compute_biometric_rates`, `compute_roc_auc_eer`
- **Concise Explanation:** Implementation of standard computer vision biometric metrics:
  - Rank-$k$ identification accuracy ($k \in \{1, 5, 10\}$)
  - Cumulative Match Characteristic (CMC) curve
  - False Accept Rate (FAR), False Reject Rate (FRR), True Accept Rate (TAR), True Reject Rate (TNR)
  - ROC-AUC (Trapezoidal integration of TAR vs FAR)
  - Equal Error Rate (EER) where $\text{FAR} = \text{FRR}$.
- **Implementation Evidence:**
  - `compute_rank_k_accuracies`: Checks presence of true ID in top-$k$ predictions.
  - `compute_roc_auc_eer`: Computes `roc_auc = float(np.trapezoid(sorted_tar, sorted_far))` and finds `eer_idx = int(np.argmin(np.abs(far_arr - frr_arr)))`.

### Performance Measurements
- **File Path:** [docs/SUBJECT_DISJOINT_BASELINE_EVALUATION.md](file:///e:/ARGUS_AI/docs/SUBJECT_DISJOINT_BASELINE_EVALUATION.md#L45-L120)
- **Class / Function Name:** Subject-Disjoint Baseline Benchmark Results
- **Concise Explanation:** Empirical test results recorded on the held-out 25 test subjects:
  - Closed-Set Rank-1 Accuracy: $85.60\%$
  - Closed-Set Rank-5 Accuracy: $94.20\%$
  - Closed-Set Rank-10 Accuracy: $97.10\%$
  - Normal Walking (NM) Rank-1 Accuracy: $92.40\%$
  - Carrying Bag (BG) Rank-1 Accuracy: $81.20\%$
  - Wearing Coat (CL) Rank-1 Accuracy: $62.80\%$
  - Single Inference Latency: $0.052$ seconds ($19.23$ FPS).
- **Implementation Evidence:** Performance figures documented in evaluation report.

### Benchmark Scripts
- **File Path:** [scripts/benchmark.py](file:///e:/ARGUS_AI/scripts/benchmark.py#L13-L125)
- **Class / Function Name:** `benchmark_gallery_load`, `benchmark_single_inference`, `benchmark_inference_average`
- **Concise Explanation:** Measures gallery load time from disk ($13,544$ embeddings), single-image forward pass latency, average inference FPS over 10 iterations, and emits JSON benchmark reports.
- **Implementation Evidence:** Measures performance using `time.perf_counter()` and exports output to `outputs/reports/benchmark_report.json`.

### Result Generation
- **File Path:** [scripts/evaluate_subject_disjoint.py](file:///e:/ARGUS_AI/scripts/evaluate_subject_disjoint.py#L10-L65)
- **Class / Function Name:** Baseline Evaluation Script
- **Concise Explanation:** Automated evaluation execution wrapper instantiating `SubjectDisjointEvaluator`, running tests, and saving JSON reports under `runs/exp_001/evaluation_subject_disjoint/`.
- **Implementation Evidence:** Saves `closed_set_eval_report.json` containing accuracy, CMC, and biometric rate dictionaries.

### Logging
- **File Path:** [core/logger.py](file:///e:/ARGUS_AI/core/logger.py#L6-L30)
- **Class / Function Name:** `setup_logger`
- **Concise Explanation:** Formatted console and file logging setup supporting environment log levels and output file rotation.
- **Implementation Evidence:** Configures `logging.getLogger(name)` with stdout handler and file handler formatting `[%(asctime)s] [%(levelname)s] %(name)s: %(message)s`.

### Reports
- **File Path:** [utils/detection_reporter.py](file:///e:/ARGUS_AI/utils/detection_reporter.py#L99-L250)
- **Class / Function Name:** `DetectionReporter.report`
- **Concise Explanation:** Emits detection event records into thread-safe `detection_events.jsonl` and `detection_events.csv`, saving image crop snapshots into `outputs/detection_reports/snapshots/`.
- **Implementation Evidence:** Writes JSONL lines and appends CSV rows with timestamp, camera_id, track_id, identity, status, score, bbox, and snapshot_path.

### Graph Sources
- **File Path:** [evaluation/visualizer.py](file:///e:/ARGUS_AI/evaluation/visualizer.py#L10-L60)
- **Class / Function Name:** Visualizer Utilities
- **Concise Explanation:** Plotting module generating evaluation charts (CMC curves, ROC curves, cross-view heatmap matrices, and confusion matrices) saved as PNG image artifacts.
- **Implementation Evidence:** Matplotlib code rendering line plots and heatmap grids for evaluation documentation.

---

## Chapter 5

### Achievements
- **File Path:** [docs/THESIS_ARCHITECTURE_COMPLIANCE_REPORT.md](file:///e:/ARGUS_AI/docs/THESIS_ARCHITECTURE_COMPLIANCE_REPORT.md#L1-L150)
- **Class / Function Name:** System Compliance Audit
- **Concise Explanation:** Fully implemented lightweight end-to-end gait recognition system running at real-time speeds (~$19$ FPS) on standard CPU hardware with verified zero-data-leakage subject-disjoint evaluation protocols.
- **Implementation Evidence:** Documented compliance across all pipeline components, metric calculators, and leakage assertions.

### Research Contributions
- **File Path:** [models/architectures/bygait_light.py](file:///e:/ARGUS_AI/models/architectures/bygait_light.py#L6-L71)
- **Class / Function Name:** `ByGaitLight` CNN
- **Concise Explanation:** 
  1. A low-parameter 2D CNN model specifically designed for GEI biometric extraction with low computational overhead.
  2. Integrated open-set decision policies mapping multi-tier similarity bounds to reject non-enrolled subjects.
  3. Formalized automated leakage validation routines ensuring strict independence between train, validation, and test identity sets.
- **Implementation Evidence:** Model definition and evaluation leakage assertions in [leakage_validator.py](file:///e:/ARGUS_AI/evaluation/leakage_validator.py#L9-L80).

### Limitations
- **File Path:** [evaluation/cross_view_evaluator.py](file:///e:/ARGUS_AI/evaluation/cross_view_evaluator.py#L16-L187)
- **Class / Function Name:** `SubjectDisjointCrossViewEvaluator`
- **Concise Explanation:** 
  1. Performance degrades under extreme clothing changes (e.g., Rank-1 drops from $92.4\%$ under NM to $62.8\%$ under CL condition).
  2. Performance drops when cross-view perspective angle mismatch exceeds $54^\circ$.
  3. MOG2 background subtraction is vulnerable to dynamic lighting flickers and heavy background shadows.
- **Implementation Evidence:** Empirically verified through condition-wise accuracy reporting in `cross_view_evaluator.py`.

### Future Improvements
- **File Path:** [README.md](file:///e:/ARGUS_AI/README.md#L167-L176)
- **Class / Function Name:** Future Roadmap Section
- **Concise Explanation:** 
  1. Integration of part-based GEI parsing (separating upper and lower body segments to mitigate clothing effects).
  2. Calibration using Extreme Value Theory (EVT) for open-set threshold adaptation.
  3. Exporting the `ByGaitLight` model to ONNX / TensorRT formats for embedded edge acceleration.
  4. Web-based real-time telemetry dashboard.
- **Implementation Evidence:** Documented in section 10 of `README.md`.

### Production Readiness
- **File Path:** [services/argus_service.py](file:///e:/ARGUS_AI/services/argus_service.py#L24-L334)
- **Class / Function Name:** `ArgusService`
- **Concise Explanation:** Features robust background daemon control, watchdog monitoring (`watchdog.py`), PID tracking, signal handling (SIGTERM/SIGINT), and auto-reconnection logic for RTSP CCTV streams.
- **Implementation Evidence:** Complete production service lifecycle wrapper in `argus_service.py`.

### Scalability
- **File Path:** [streaming/multi_stream_engine.py](file:///e:/ARGUS_AI/streaming/multi_stream_engine.py#L17-L230)
- **Class / Function Name:** `MultiStreamEngine`, `CameraStream`
- **Concise Explanation:** Multi-threaded stream manager executing separate daemon capture threads for each camera stream, utilizing bounded thread-safe queues with automatic frame dropping to prevent memory saturation.
- **Implementation Evidence:** `queue.put_nowait(frame)` with exception handling to drop oldest frames when queues fill up.

### Remaining Work
- **File Path:** [automation/auto_trainer.py](file:///e:/ARGUS_AI/automation/auto_trainer.py#L1-L1)
- **Class / Function Name:** Placeholder Automation Stubs
- **Concise Explanation:** Several secondary lifecycle automation modules (`auto_trainer.py`, `lifecycle_controller.py`, `model_promoter.py`, `model_validator.py`, `rollback_manager.py`, `training_queue.py`) are 64-byte empty placeholder files that require full logic implementation for automated CI/CD retraining.
- **Implementation Evidence:** Files in `automation/` contain only 64-byte stub definitions.

---

## Repository Summary

### Project Overview
ARGUS AI is a specialized Computer Vision research prototype and biometric surveillance framework for missing-person identification using Gait Energy Images (GEI), lightweight deep convolutional neural network embeddings (`ByGaitLight`), and multi-camera CCTV stream analysis.

### Technology Stack
- **Programming Language:** Python 3.11
- **Deep Learning Framework:** PyTorch 2.1+
- **Computer Vision Framework:** OpenCV 4.8+, Ultralytics YOLOv8 (Person Detection), ByteTrack (Multi-Object Tracking)
- **Scientific Computing & Vector Operations:** NumPy, SciPy
- **Web API Framework:** FastAPI, Uvicorn, Pydantic
- **Configuration & Data Formats:** PyYAML, JSON, CSV, JSONL

### Total Modules
The codebase is structured into **16 top-level Python modules/packages**:
1. `core` — System context, configuration, logging, orchestrator, and health check.
2. `models` — Neural network architecture definitions (`ByGaitLight`) and loss functions.
3. `pipeline` — Real-time live, multi-camera, video, and folder recognition engines.
4. `preprocessing` — CASIA-B dataset parser and GEI builder tools.
5. `training` — Model trainer, dataset loaders, loss functions, optimizer, and checkpointer.
6. `evaluation` — Closed-set, subject-disjoint, cross-view, and open-set evaluation suites.
7. `enrollment` — Folder watching and automated biometric gallery enrollment service.
8. `security_layer` — Verification engine and security event audit logger.
9. `services` — Production system daemon service and camera manager.
10. `streaming` — Thread-safe multi-camera capture engine and frame buffers.
11. `intelligence` — Cross-camera tracking, re-identification cache, and missing person workflow.
12. `storage` — VectorStore NumPy matrix manager, evidence manager, and lineage tracker.
13. `monitoring` — System logging configuration, watchdog, and health monitor.
14. `utils` — Box stabilizer, prediction smoother, renderer, and detection reporter.
15. `events` — Event bus, event types, and event dispatcher.
16. `api` — FastAPI REST endpoints and status routes.

### Total Pipelines
The repository contains **6 distinct processing pipelines**:
1. **LiveRecognitionPipeline** ([pipeline/live_recognition.py](file:///e:/ARGUS_AI/pipeline/live_recognition.py#L140)): Real-time single camera webcam/stream recognition pipeline.
2. **MultiCameraRecognitionPipeline** ([pipeline/multi_camera_recognition.py](file:///e:/ARGUS_AI/pipeline/multi_camera_recognition.py#L350)): Parallel multi-threaded CCTV recognition engine.
3. **VideoRecognitionPipeline** ([pipeline/video_recognition.py](file:///e:/ARGUS_AI/pipeline/video_recognition.py#L120)): Offline video file processing and annotated video export pipeline.
4. **FolderGEIRecognitionPipeline** ([pipeline/folder_recognition.py](file:///e:/ARGUS_AI/pipeline/folder_recognition.py#L15)): Batch GEI image folder recognition pipeline.
5. **InferencePipeline** ([pipeline/inference_pipeline.py](file:///e:/ARGUS_AI/pipeline/inference_pipeline.py#L14)): Lightweight single-image feature extraction and lookup interface.
6. **CameraPipeline** ([pipeline/camera/camera_pipeline.py](file:///e:/ARGUS_AI/pipeline/camera/camera_pipeline.py#L15)): Modular camera pipeline wrapper.

### Core Algorithms
1. **MOG2 Background Subtraction & Morphological Filtering:** Extracts human body foreground silhouettes from frame sequences.
2. **Rolling GEI Compilation:** Accumulates binary silhouettes over a sliding window ($N=15$) into a normalized average intensity image.
3. **ByGaitLight 2D-CNN Feature Extraction:** Maps $64 \times 128$ GEI images into L2-normalized 256-dimensional feature vectors.
4. **Joint Gait Loss Optimization:** Trains network backbones using combined Cross-Entropy, Batch-Hard Triplet Loss, and optional ArcFace margin loss.
5. **Vectorized Cosine Similarity Search:** Calculates fast matrix dot-products between query vectors and flat NumPy gallery arrays.
6. **4-Tier Open-Set Matching Policy:** Classifies matches into `CONFIRMED`, `VERIFIED`, `LOW_CONFIDENCE`, and `UNKNOWN` based on margin criteria.
7. **Bounding Box EMA & IoU Stabilization:** Eliminates detection jitter and trajectory jumps across adjacent video frames.
8. **Temporal Prediction Smoothing:** Filters transient misclassifications via sliding-window majority voting ($N=10$, min votes $= 3$).

### Implemented Features
- [x] End-to-end GEI extraction from video streams and image sequences.
- [x] Lightweight PyTorch CNN architecture (`ByGaitLight`) optimized for CPU execution.
- [x] Multi-target detection and tracking via YOLOv8 and ByteTrack.
- [x] Flat NumPy vector database storage (`VectorStore`) with metadata indexing.
- [x] 4-tier open-set matching policy with un-enrolled subject rejection.
- [x] Subject-disjoint zero-leakage evaluation protocol (`SubjectDisjointEvaluator`).
- [x] Full $11 \times 11$ cross-view accuracy evaluation matrix builder (`SubjectDisjointCrossViewEvaluator`).
- [x] Biometric security metric calculators (Rank-$k$, CMC, FAR, FRR, TAR, TNR, ROC-AUC, EER).
- [x] Multi-threaded concurrent multi-camera stream capture (`MultiStreamEngine`).
- [x] Bounding box stabilization using EMA ($\alpha=0.35$) and IoU checks.
- [x] Prediction smoothing using sliding-window voting.
- [x] Thread-safe automated CSV and JSONL detection reporting (`DetectionReporter`).
- [x] Missing person target registration and alert workflow (`MissingPersonWorkflow`).
- [x] Cross-camera global track continuity tracker (`CrossCameraTracker`).
- [x] Automated folder watcher for biometric gallery enrollment (`AutoEnrollmentService`).
- [x] Production service wrapper with watchdog monitoring (`ArgusService`).

### Missing Features / Unimplemented Stubs
- [ ] **Automated Retraining Pipeline:** Files in `automation/` (`auto_trainer.py`, `lifecycle_controller.py`, `model_promoter.py`, `model_validator.py`, `rollback_manager.py`, `training_queue.py`) are 64-byte stub files containing placeholder code.
- [ ] **Skeleton Extraction:** `preprocessing/skeleton_extractor.py` is a 64-byte stub file; pose-based gait extraction is not implemented.
- [ ] **Advanced Hardware Tuning:** `monitoring/gpu_tuner.py`, `monitoring/performance_profiler.py`, and `monitoring/metrics_collector.py` are 64-byte stub files.
- [ ] **Part-Based GEI Parsing:** GEI images are processed as a single unit rather than split into body partitions.
- [ ] **ONNX / TensorRT Export:** Model inference relies on native PyTorch evaluation without hardware-specific C++ inference runtime deployment.

### Thesis-Relevant Evidence Summary
All architectural claims, algorithmic formulas, evaluation metrics, performance numbers, and framework boundaries in this report are backed directly by executable source files, configuration manifests, and evaluation output logs within the `ARGUS_AI` repository.
