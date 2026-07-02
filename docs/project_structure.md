# Project Structure Reference

This document maps out the repository folder structure, identifying the roles of the subsystems and detail how they are organized.

---

## Directory Overview

```
ARGUS_AI/
│
├── api/                       # REST API Server Gateway
│   ├── routes/                # Endpoint router declarations (status, enrollment, inference)
│   └── schemas.py             # Pydantic request/response structures
│
├── configs/                   # System configuration files (YAML format)
│   ├── base.yaml              # Global default paths and health checks
│   ├── cameras.yaml           # Parallel stream configurations
│   ├── inference.yaml         # Match matching thresholds and policies
│   └── logging.yaml           # Format patterns for files and stream outputs
│
├── core/                      # Global Bootstrap & System Context Manager
│   ├── boot.py                # BootManager orchestrator
│   ├── config.py              # YAML loader wrapper
│   ├── context.py             # Global context dictionary
│   ├── health_check.py        # Environmental checks checker
│   ├── logger.py              # Central log dispatcher
│   ├── system.py              # Main system lifecycle coordinator
│   └── system_monitor.py      # Core/GPU resource reader
│
├── data/                      # Raw inputs and preprocessed dataset directories
│   ├── casia_processed/       # Output location for extracted GEI frames
│   └── new_input/             # Target directory polled by folder watcher
│
├── docs/                      # Technical specifications and markdown guidelines
│
├── enrollment/                # Biometric Subject Registration Modules
│   ├── enrollment_manager.py  # Coordinates validation and enrollment loops
│   ├── folder_watcher.py      # Background folder polling service
│   └── gallery_updater.py     # Appends features to the database arrays
│
├── evaluation/                # Performance assessment tools
│   ├── evaluator.py           # Validates matches across validation sets
│   ├── metrics.py             # Computes accuracies, FMR, FNMR, and EER indices
│   └── visualizer.py          # Renders matplotlib curves and sweeps
│
├── events/                    # Event-driven system communication
│   ├── event_bus.py           # Global observer pattern event publisher
│   └── event_types.py         # Event enums definitions
│
├── intelligence/              # Decision Calibration Layer
│   └── confidence_scorer.py   # Maps similarities to classification zones
│
├── models/                    # Model checkers and local database files
│   ├── architectures/         # Custom networks architectures (ByGaitLight)
│   ├── gallery/               # Offline validation database (vectors + labels)
│   └── live_gallery/          # Real-time surveillance database
│
├── outputs/                   # System generated outputs
│   ├── eval_reports/          # Evaluation CSV summaries and PNG curves
│   ├── reports/               # Speed benchmark reports
│   └── security_logs/         # Auditing logs (CSV format)
│
├── pipeline/                  # Video frame processing orchestrators
│   ├── inference_pipeline.py  # Single-image verification wrapper
│   ├── live_recognition.py    # Main camera surveillance pipeline
│   ├── multi_camera_recognition.py # Threaded multi-stream recognition orchestrator
│   ├── video_recognition.py   # Video file processing pipeline
│   └── steps/                 # Component steps (YOLO, ByteTrack, Silhouette, LiveGEI)
│
├── preprocessing/             # Image formatting & feature aggregation
│   ├── dataset_builder.py     # Parses CASIA ZIP structures and resizes outputs
│   └── gei_builder.py         # Compiles average GEIs from silhouette sequences
│
├── scripts/                   # Standalone testing and execution scripts
│   ├── benchmark.py           # Speed testing benchmarker
│   ├── evaluate_model.py      # Model performance evaluator
│   └── system_check.py        # Environment diagnostician
│
├── security_layer/            # Risk classification layers
│   ├── security_engine.py     # Classifies scores to threat levels
│   └── security_logger.py     # Writes threat records to audit CSV
│
├── storage/                   # Database read/write managers
│   └── vector_store.py        # Serializer and deserializer for vectors (.npy)
│
├── streaming/                 # Video ingestion
│   └── stream_engine.py       # Threaded webcam frame grabber
│
├── tests/                     # Unit and integration test suites
│   ├── integration/           # Flows verification tests
│   └── unit/                  # Module math and architectures tests
│
├── cli.py                     # Central CLI command dispatcher
├── main.py                    # Orchestrator runner entry point
├── Makefile                   # Automation build file
├── requirements.txt           # Active runtime packages
└── requirements-dev.txt       # Active dev packages
```

---

## Core System Packages vs Placeholder Stubs

The repository maintains a clean boundary between core functionality and future roadmap stubs:
- **Core system modules (~81.1% of files):** Fully implemented and tested. Includes the camera ingestion steps, YOLO and ByteTrack wrappers, background segmentations, ByGaitLight CNN embedding calculations, vectorized lookups, API gateway routes, and security risk classifications.
- **Roadmap stubs:** Represent placeholders for future features. These files contain a single docstring line or empty class definitions and do not impact current system runs (e.g., `models/architectures/gait_encoder.py`, `preprocessing/skeleton_extractor.py`, `monitoring/*`, `registry/*`, `automation/*`).
