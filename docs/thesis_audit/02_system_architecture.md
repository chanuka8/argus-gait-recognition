# Phase 2 — System Architecture

## 2.1 High-Level Architecture

```mermaid
graph TB
    subgraph Input["Input Sources"]
        USB[USB Webcam]
        RTSP[RTSP/CCTV Stream]
        VID[Video File]
    end

    subgraph Core["Core Pipeline"]
        DET[Person Detection<br/>YOLOv8n]
        TRK[Multi-Target Tracking<br/>ByteTrack]
        SIL[Silhouette Extraction<br/>Otsu + Morphology]
        GEI[GEI Accumulation<br/>Rolling Window Mean]
        EMB[Embedding Generation<br/>ByGaitLight CNN]
        MATCH[Gallery Matching<br/>Cosine Similarity]
        ADAPT[Adaptive Decision<br/>Multi-Tier Thresholds]
        SMOOTH[Prediction Smoothing<br/>Voting-Based]
    end

    subgraph Output["Outputs"]
        DISP[CCTV Overlay Display]
        RPT[Detection Reports<br/>CSV + JSONL]
        SEC[Security Audit Log]
        ALERT[Alert System]
    end

    subgraph Storage["Storage"]
        GAL[Gallery Store<br/>NumPy + JSON]
        EVID[Evidence Manager]
    end

    USB --> DET
    RTSP --> DET
    VID --> DET
    DET --> TRK
    TRK --> SIL
    SIL --> GEI
    GEI --> EMB
    EMB --> MATCH
    MATCH --> ADAPT
    ADAPT --> SMOOTH

    SMOOTH --> DISP
    SMOOTH --> RPT
    SMOOTH --> SEC
    SMOOTH --> ALERT
    GAL --> MATCH
    SMOOTH --> EVID
```

## 2.2 Module Dependency Map

```mermaid
graph LR
    subgraph Pipeline
        LR[live_recognition.py]
        MCR[multi_camera_recognition.py]
        VR[video_recognition.py]
    end

    subgraph Steps
        TS[tracking.py<br/>YOLOv8+ByteTrack]
        SS[silhouette_step.py]
        LG[live_gei.py]
        MS[matching_step.py]
        CMS[centroid_matching_step.py]
        DS[detection.py]
    end

    subgraph Models
        BGL[bygait_light.py<br/>CNN Backbone]
        LOSS[losses.py<br/>ArcFace+Triplet]
    end

    subgraph Training
        TR[trainer.py]
        DS2[dataset.py]
        DL[dataloader.py]
    end

    subgraph Evaluation
        EV[evaluator.py]
        CV[cross_view_evaluator.py]
        OS[open_set_evaluator.py]
        MET[metrics.py]
        LV[leakage_validator.py]
    end

    subgraph Security
        SE[security_engine.py]
        SL[security_logger.py]
    end

    subgraph Storage2[Storage]
        VS[vector_store.py]
        EM[evidence_manager.py]
    end

    LR --> TS
    LR --> SS
    LR --> LG
    LR --> MS
    LR --> CMS
    LR --> BGL
    LR --> VS
    LR --> SE

    MCR --> TS
    MCR --> SS
    MCR --> LG
    MCR --> MS
    MCR --> CMS
    MCR --> BGL
    MCR --> VS
    MCR --> SE

    TR --> BGL
    TR --> LOSS
    TR --> DS2
    TR --> DL

    EV --> BGL
    EV --> MS
    EV --> MET
    EV --> LV

    SE --> SL
```

## 2.3 Multi-Camera Architecture

```mermaid
graph TB
    subgraph Orchestrator["MultiCameraRecognitionPipeline"]
        direction TB
        MC_INIT["Shared Resources<br/>(Read-Only)"]
        MODEL["ByGaitLight Model<br/>(eval mode)"]
        GALLERY["Gallery Features<br/>(NumPy arrays)"]
        MATCHER["MatchingStep<br/>(stateless)"]
        CMATCHER["CentroidMatchingStep<br/>(stateless)"]
    end

    subgraph Workers["Per-Camera Isolated State"]
        W1["CameraWorkerState<br/>camera_01"]
        W2["CameraWorkerState<br/>camera_02"]
    end

    subgraph PerCameraState["Each Worker Contains"]
        TRACKER["Own YOLO + ByteTrack"]
        SIL_STEP["Own SilhouetteStep"]
        BUFFERS["Own GEI Buffers"]
        SMOOTHER["Own PredictionSmoother"]
        BOX_STAB["Own BoxStabilizer"]
        COUNTERS["Own Frame Counters"]
    end

    subgraph Streams["MultiStreamEngine"]
        S1["Stream Thread 1"]
        S2["Stream Thread 2"]
    end

    MC_INIT --> MODEL
    MC_INIT --> GALLERY
    MC_INIT --> MATCHER
    MC_INIT --> CMATCHER

    S1 --> W1
    S2 --> W2

    W1 -.->|reads| MODEL
    W1 -.->|reads| GALLERY
    W2 -.->|reads| MODEL
    W2 -.->|reads| GALLERY

    W1 --> PerCameraState
```

## 2.4 Deployment Architecture

```mermaid
graph LR
    subgraph Local["Local Deployment"]
        CLI[cli.py<br/>Entry Point]
        VENV[Python 3.11+ venv]
        GPU_CPU["CPU / CUDA Device"]
    end

    subgraph Service["Windows Service"]
        PS["install_service.ps1"]
        ARGUS["ARGUS Service"]
    end

    subgraph CI["CI/CD"]
        GHA["GitHub Actions"]
        LINT["Ruff Linter"]
        TEST["Pytest Suite"]
    end

    CLI --> VENV
    VENV --> GPU_CPU
    PS --> ARGUS
    GHA --> LINT
    GHA --> TEST
```

## 2.5 Data Flow Diagram

```mermaid
flowchart LR
    A[Raw Video Frame] --> B[YOLOv8n Detect<br/>persons only, class=0]
    B --> C[ByteTrack Assign<br/>Track IDs]
    C --> D[Crop Person<br/>from Bounding Box]
    D --> E[Grayscale → Blur<br/>→ Otsu Threshold<br/>→ Morphology Clean<br/>→ Contour Extract]
    E --> F[Normalize to<br/>64×128 Canvas]
    F --> G["Rolling Window<br/>Mean (15 frames)"]
    G --> H[GEI Image<br/>64×128 uint8]
    H --> I["Normalize to [0,1]<br/>Unsqueeze to 1×1×128×64"]
    I --> J[ByGaitLight CNN<br/>Forward Pass]
    J --> K[256-dim L2-Normalized<br/>Embedding Vector]
    K --> L[Cosine Similarity<br/>vs Gallery]
    L --> M{Adaptive Decision<br/>Policy}
    M -->|score ≥ 0.92| N[CONFIRMED_MATCH]
    M -->|0.85 ≤ score < 0.92| O[Centroid Verification]
    M -->|0.70 ≤ score < 0.85| P[LOW_CONFIDENCE]
    M -->|score < 0.70| Q[UNKNOWN_PERSON]
    O -->|agrees| R[VERIFIED_MATCH]
    O -->|disagrees| S[REVIEW_REQUIRED]
```

## 2.6 Technology Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.11+ (CI), 3.14.6 (local) |
| Deep Learning | PyTorch | ≥2.0.0 |
| Object Detection | Ultralytics YOLOv8n | ≥8.0.0 |
| Multi-Target Tracking | Supervision ByteTrack | ≥0.18.0 |
| Image Processing | OpenCV | ≥4.8.0 |
| Numerical Computing | NumPy | ≥1.24.0 |
| API Framework | FastAPI + Uvicorn | ≥0.100.0 |
| Configuration | PyYAML | ≥6.0.0 |
| Logging | Python stdlib logging | Built-in |
| Testing | Pytest | ≥8.0.0 |
| Linting | Ruff | ≥0.3.0 |
| CI/CD | GitHub Actions | Ubuntu-latest |
| Deployment | PowerShell service scripts | Windows |
