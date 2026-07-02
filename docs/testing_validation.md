# Testing and Validation

This document outlines the unit and integration testing suite structure, test coverage areas, and execution workflows.

---

## 1. Test Suite Structure

The test suite resides in the [tests/](file:///e:/ARGUS_AI/tests/) folder, divided into isolated unit and integration test categories:

```
tests/
│
├── conftest.py             # Shared pytest fixtures
│
├── integration/            # Multi-component workflow validation tests
│   ├── test_end_to_end.py  # Inference pipeline end-to-end matching
│   └── test_enrollment_flow.py # Folder monitoring validation and registration
│
└── unit/                   # Math calculations and component APIs tests
    ├── test_evaluation.py   # Chart generation and metrics calculations
    ├── test_model_arch.py   # ByGaitLight tensor shapes forward passes
    ├── test_pipeline.py     # Cache models and streaming speed throttles
    ├── test_preprocessing.py # Rolling GEI buffers and frame aggregations
    ├── test_training.py     # Dataset loader splits
    └── test_vector_store.py # Flat vector saving/loads
```

---

## 2. Test Coverage & Validation Targets

- **Unit Tests:**
  - `test_model_arch.py`: Validates that passing a tensor of shape `[1, 1, 128, 64]` through `ByGaitLight` outputs a L2-normalized embedding of shape `[1, 256]`.
  - `test_preprocessing.py`: Verifies that the rolling GEI buffer appends and aggregates silhouettes, returning the correct average intensities.
  - `test_vector_store.py`: Tests that the local database correctly saves and reads `.npy` arrays and updates metadata mappings without data loss.
- **Integration Tests:**
  - `test_enrollment_flow.py`: Mocks a watcher scan, validates target folder sizes, processes frames, and appends the new subject profile to the vector database.
  - `test_end_to_end.py`: Runs a mock video frame batch, crops targets, generates binary contours, synthesizes rolling GEIs, and compares predictions against registered profiles.

---

## 3. Running the Test Suite

There are multiple ways to execute the tests:

### Unified CLI Wrapper
Runs the standard unittest discovery runner:
```bash
python cli.py --mode tests
```
*Alternative (Makefile):*
```bash
make test
```

### Direct Pytest Execution (Recommended for Dev)
Provides detailed reports, execution times, and supports debugging flags:
```bash
pytest tests -v
```
*Alternative (Makefile):*
```bash
make pytest
```
All 15 tests should pass successfully in under 5 seconds.

### Utility and Integration Compilation Check
Verify all updated components compile cleanly before deployment:
```bash
python -m py_compile utils\display_renderer.py utils\detection_reporter.py pipeline\live_recognition.py pipeline\video_recognition.py pipeline\multi_camera_recognition.py
```

