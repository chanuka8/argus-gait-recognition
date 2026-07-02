# Pipeline Workflow Specifications

This document outlines the detailed sequence of steps, functions, and data transformations for each of the module's core workflows.

---

## 1. Model Training Workflow

The training workflow implements a standard deep learning training loop using the custom `ByGaitLight` network to learn feature representations from Gait Energy Images (GEIs).

```
[CASIA Dataset Path]
       │
       ▼
[Dataset Loader] ───────────────> (Subject folder checks & paths validation)
       │
       ▼
[GEIDataset Class] ─────────────> (Loads 128x64 PNGs, scales values to [0.1], adds dim)
       │
       ▼
[build_dataloaders()] ──────────> (Splits dataset into 80% train / 20% validation)
       │
       ▼
[CrossEntropy Loss] ────────────> (Evaluates classification output errors)
       │
       ▼
[Adam Optimizer] ───────────────> (Updates CNN weights with weight decay adjustments)
       │
       ▼
[Callbacks Engine] ─────────────> (EarlyStopping controls training, logs stats to file)
       │
       ▼
[Model Checkpointer] ───────────> (Saves weights to best_model.pth & training stats JSON)
```

---

## 2. Gallery Building Workflow

The gallery builder converts a preprocessed dataset folder containing registered individuals into a localized vector database.

1.  **Scanner Phase:** Initiates a scan of the target directory (e.g. `data/casia_processed/gei/`). It validates that each directory corresponds to a unique subject.
2.  **Instantiation Phase:** Loads the pretrained `ByGaitLight` CNN weights and initializes the `FeatureExtractionStep` instance.
3.  **Extraction Loop:** 
    *   Iterates through each subject folder.
    *   Loads all associated GEI images.
    *   Extracts a 128-dimensional embedding vector for each GEI.
    *   Maintains a parallel list of target IDs (labels) corresponding to the vectors.
4.  **Serialization Phase:** Invokes the `VectorStore` to save:
    *   Embeddings as a float32 NumPy array (`gallery_features.npy`).
    *   Target IDs as an array (`gallery_labels.npy`).
    *   Subject configurations list as a JSON file (`gallery_metadata.json`).

---

## 3. Profile Enrollment Workflow

Registers a new person and appends their identity signature to the active gallery database.

```
Incoming Directory Path 
       │
       ▼
[EnrollmentValidator] ──────> (Validates path, checks if images exist, scans count limits)
       │
       ▼
[FeatureExtractionStep] ────> (Generates a 128-d vector embedding for each image)
       │
       ▼
[VectorStore] ──────────────> (Loads active gallery files, appends new records, 
                               re-saves updated features and labels back to disk)
```

---

## 4. Live Recognition Pipeline Workflow

Coordinates real-time ingestion, object tracking, motion modeling, feature extraction, matching, and visual output generation.

```
       +--------------------------------------------------------------+
       |                     INGESTION THREAD                         |
       |  Read stream frame -> Push to BufferQueue -> Drop if Full    |
       +------------------------------+-------------------------------+
                                      |
                                      v
       +--------------------------------------------------------------+
       |                    DETECTION & TRACKING                      |
       |  Pop frame -> Run YOLOv8 -> Track boxes with ByteTrack       |
       +------------------------------+-------------------------------+
                                      |
                                      v
       +--------------------------------------------------------------+
       |                     SILHOUETTE EXTRACTION                    |
       |  Crop track bounding box -> Extract binary mask via MOG2     |
       +------------------------------+-------------------------------+
                                      |
                                      v
       +--------------------------------------------------------------+
       |                     GEI MOTION BUFFER                        |
       |  Add mask to Track ID rolling buffer -> Accumulate 15 frames |
       +------------------------------+-------------------------------+
                                      |
                                      v
       +--------------------------------------------------------------+
       |                     FEATURE MATCHING                         |
       |  Build GEI -> Extract CNN embedding -> Run Cosine Similarity |
       +------------------------------+-------------------------------+
                                      |
                                      v
       +--------------------------------------------------------------+
       |                  DECISION & THREAT LOGGER                    |
       |  Check threshold -> Classify threat -> Append to CSV log     |
       +------------------------------+-------------------------------+
                                      |
                                      v
       +--------------------------------------------------------------+
       |                    VISUAL FEEDBACK OVERLAY                   |
       |  Render labels, overlay box color (Green/Orange/Red), display|
       +--------------------------------------------------------------+
```

---

## 5. REST API Workflow

Handles HTTP request routing and execution contexts for the client endpoints.

1.  **GET `/health` / `/metrics`:**
    *   FastAPI calls the route controller.
    *   Queries `SystemContext` parameters to retrieve runtime indicators and system statuses.
    *   Queries `SystemMonitor` for CPU, RAM, and VRAM resource status, and returns a JSON response.
2.  **POST `/identify`:**
    *   Client submits a JSON request with a target image path.
    *   `api/routes/inference.py` instantiates the `InferencePipeline`.
    *   The pipeline extracts the embedding, queries `VectorStore` to load features, and calls `MatchingStep` to identify the subject.
    *   Returns the matched ID and similarity score.
3.  **POST `/enroll`:**
    *   Client submits a directory path of the candidate profile.
    *   `api/routes/enrollment.py` instantiates the `EnrollmentManager`.
    *   Triggers folder verification, feature extraction, database updating, and returns success logs.

---

## 6. Security Event Logging Workflow

Provides real-time audit trails of all matching outcomes.

```
Similarity Match Result (Target Identity, Score Value, Track ID)
                         │
                         ▼
        [Evaluate Confidence Scorer Threshold]
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
   Score >= 0.80                     Score < 0.80 or ID == UNKNOWN
  Severity: INFO                    Severity: MEDIUM or HIGH
  Decision: ALLOW                   Decision: REVIEW_REQUIRED / SECURITY_ALERT
        │                                 │
        └────────────────┬────────────────┘
                         │
                         ▼
              [Write Security Event Log]
  (Timestamp, Track ID, Matched ID, Score, Severity, Decision)
                         │
                         ▼
    Appended to: outputs/security_logs/security_events.csv
```
