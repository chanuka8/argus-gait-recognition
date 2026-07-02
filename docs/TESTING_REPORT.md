# System Verification & Testing Report

This testing report document presents the validation results, unit test checks, performance metrics, and security audit verifications for the ARGUS Gait Recognition Module.

---

## 1. Automated System Health Check

We executed the system verification diagnostic script via PowerShell to check library version compatibilities and confirm file paths.

```powershell
python scripts/system_check.py
```

### Verification Output:
```
============================================================
ARGUS SYSTEM CHECK
============================================================
Python version     : 3.11.9
Operating system   : Windows-10-10.0.26200-SP0
CPU cores          : 12
RAM usage          : 88.4%
------------------------------------------------------------
LIBRARIES
------------------------------------------------------------
NumPy              : 2.4.6
OpenCV             : 4.13.0
PyTorch            : 2.12.0+cpu
CUDA available     : False
PyYAML             : 6.0.3
------------------------------------------------------------
PROJECT FILES
------------------------------------------------------------
VERSION                       : OK
main.py                       : OK
core/system.py                : OK
core/orchestrator.py          : OK
core/boot.py                  : OK
configs/base.yaml             : OK
configs/inference.yaml        : OK
requirements.txt              : OK
------------------------------------------------------------
PROJECT DIRECTORIES
------------------------------------------------------------
core                          : OK
configs                       : OK
events                        : OK
models                        : OK
preprocessing                 : OK
pipeline                      : OK
training                      : OK
evaluation                    : OK
storage                       : OK
outputs                       : OK
data                          : OK
============================================================
System check completed successfully.
============================================================
```

---

## 2. Unit Testing Execution

We executed the module's unit tests using the standard Python test discovery runner to verify the integrity of the components:

```powershell
python -m unittest discover -s tests
```

### Test Suite Execution Output:
```
........
----------------------------------------------------------------------
Ran 8 tests in 1.482s

OK
```

### Verified Test Cases:
*   **`TestVectorStore`:**
    *   `test_vector_store_exists`: Confirms initialization of the vector store class.
    *   `test_load_returns_tuple`: Verifies that loading the vector database returns a tuple structure (embeddings, labels, metadata).
*   **`TestEvaluation`:**
    *   `test_visualizer_creation`: Validates that visualizer classes compile.
    *   `test_accuracy_plot`: Verifies that matplotlib outputs plot files correctly.
*   **`TestPipeline`:**
    *   `test_cache_engine`: Validates LRU cache write and read hits.
    *   `test_speed_controller`: Checks FPS throttling logic and initialization.
*   **`TestPreprocessing`:**
    *   `test_gei_buffer`: Verifies rolling queue frame accumulation.
    *   `test_gei_build`: Validates that average GEI calculation returns binary frames.

---

## 3. Model Training & Evaluation Results

We evaluated our custom `ByGaitLight` CNN model using the verification script:

```powershell
python scripts/evaluate_model.py --model-path runs/exp_001/best_model.pth --max-images 500
```

### Evaluation Metrics:
*   **Dataset Source:** CASIA-B processed partitions.
*   **Database Gallery Scale:** **13,750 embeddings** representing **127 enrolled subjects**.
*   **Test Probe Samples:** 500 samples.
*   **Validation Performance Result:** **Rank-1 Matching Accuracy = 52.8%** (0.528).

---

## 4. API Endpoints Verification

We verified the FastAPI routes by running the REST backend and testing it using `curl` requests in PowerShell:

1.  **System Health (`GET /health`):**
    *   *Result:* Returned status code `200 OK` with system parameters:
        ```json
        { "status": "healthy", "mode": "inference" }
        ```
2.  **Telemetry Metrics (`GET /metrics`):**
    *   *Result:* Returned resource utilization logs, confirming CPU loading stats and RAM allocations.
3.  **Target Identification (`POST /identify`):**
    *   *Result:* Verified matching using a target test image path, returning matching results in under 200 ms:
        ```json
        { "identity": "034", "score": 1.0 }
        ```
4.  **Person Enrollment (`POST /enroll`):**
    *   *Result:* Successfully registered a new subject folder, updating the NumPy database with the extracted vectors.

---

## 5. Live Recognition Testing

We tested the real-time processing feed using the webcam feed launcher script:

```powershell
python scripts/test_live_recognition.py
```

### Observed Behavior:
*   **Ingestion:** Real-time camera streams are captured and routed through the frame buffer queue.
*   **Detection & Tracking:** Bounding boxes are drawn around subjects, and persistent Track IDs are assigned.
*   **Gait Processing:** As subjects walk, their rolling GEIs accumulate. Once the buffer is full, matching queries the database and updates target bounding boxes with identity annotations.

---

## 6. Security Threat Layer Verification

We verified the threat assessment rules using the security layer test script:

```powershell
python scripts/test_security_layer.py
```

### Output:
```
{'severity': 'INFO', 'decision': 'ALLOW'}
{'severity': 'MEDIUM', 'decision': 'REVIEW_REQUIRED'}
{'severity': 'HIGH', 'decision': 'SECURITY_ALERT'}
```

### Log File Validation:
An inspection of `outputs/security_logs/security_events.csv` confirmed that security event records are appended correctly:
```csv
timestamp,track_id,identity,score,severity,decision
2026-06-11T10:01:18.528190,1,027,0.92,INFO,ALLOW
2026-06-11T10:01:18.530190,2,027,0.65,MEDIUM,REVIEW_REQUIRED
2026-06-11T10:01:18.532190,3,UNKNOWN,0.31,HIGH,SECURITY_ALERT
```

---

## 7. System Latency Benchmarking

We evaluated system latency and throughput on the host CPU using the benchmark tool:

```powershell
python scripts/benchmark.py
```

### Benchmark Metrics:
*   **Database Load Time (`gallery_load_time`):** **0.0079 seconds** (7.9 ms)
*   **Pipeline Initialization Latency (`pipeline_init_time`):** **0.0526 seconds** (52.6 ms)
*   **Target Inference Match Time (`prediction_time`):** **0.1555 seconds** (155.5 ms)
*   **Total Processing Cycle (`total_benchmark_time`):** **0.2195 seconds** (219.5 ms)
