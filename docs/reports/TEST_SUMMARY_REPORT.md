# ARGUS AI — Test Summary Audit Report

**Report Generated:** 2026-07-30T10:37:00+05:30  
**Repository Version / Commit Hash:** `d9aefed6c95def63f01cd3fc4ad2f718cdd1ea13`  
**Working Tree Status:** Dirty (Uncommitted documentation/report suite additions)  
**Audit Policy:** Zero False Positive Evidence-Based Reporting Policy  
**Target Path:** `docs/reports/TEST_SUMMARY_REPORT.md`

---

## Executive Summary & Official Test Metrics

> **Execution Command:** `.\venv\Scripts\python.exe -m pytest -q --tb=short`  
> **Execution Duration:** 23.45 seconds  
> **Active Test Pass Rate:** **100.0%** ($341 / (341 + 0)$)  
> *Note:* Skipped tests are **NOT** counted as passed in accordance with strict audit policy.

---

## 1. Overall Test Results Breakdown

| Outcome Category | Count | Percentage | Verification Status |
| :--- | :---: | :---: | :--- |
| **Total Tests Executed** | **342** | 100.0% | **Verified** |
| **Passed Tests** | **341** | 99.71% of total (100.0% active) | **Verified** |
| **Skipped Tests** | **1** | 0.29% | **Verified** |
| **Failed Tests** | **0** | **0.00%** | **Verified** |
| **Warnings Recorded** | **17** | Deprecation notices | **Verified** |

---

## 2. Test Category Inventory

| Test Suite Category | Directory Location | Passed Count | Status |
| :--- | :--- | :---: | :---: |
| **Unit Tests** | `tests/unit/` | **260** | **PASS** |
| **Integration Tests** | `tests/integration/` | **65** | **PASS** |
| **Security Tests** | `tests/security/` | **16** | **PASS** |
| **Deployment Readiness Tests** | `tests/unit/test_deployment_readiness.py` | Verified | **PASS** |
| **Inference Backend Tests** | `tests/unit/test_inference_backends.py` | Verified | **PASS** |
| **ONNX Export Tests** | `tests/unit/test_onnx_export_validation.py` | Verified | **PASS** |
| **Configuration Tests** | `tests/unit/test_config_validator.py` | Verified | **PASS** |
| **Doctor Tests** | `tests/unit/test_doctor.py` | Verified | **PASS** |

---

## 3. Skipped Test Inventory

| Test Name | File Location | Skip Reason / Prerequisite |
| :--- | :--- | :--- |
| `test_rtsp_reconnection_resilience` | `tests/integration/test_pipeline_performance.py` | Physical network RTSP camera stream unavailable in test environment |

---

## 4. Warning Categories Summary

| Warning Type | Triggering Source | Details |
| :--- | :--- | :--- |
| **FutureWarning** | `ByteTrack` tracker | Deprecated in supervision v0.28.0; removal scheduled in v0.30.0 |
| **DeprecationWarning** | Legacy TorchScript ONNX export | `torch.onnx.export` legacy exporter (PyTorch 2.9 transition) |

---

## 5. Code Coverage Statement

> [!NOTE]
> **Code Coverage Measurement:** Code coverage percentage was **NOT** measured during this pytest run (coverage tool was not invoked). Code coverage percentage is officially classified as `UNABLE_TO_VERIFY`.

---
**Status:** `VERIFIED - 100% ACTIVE TEST PASS RATE`
