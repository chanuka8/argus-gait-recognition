# ARGUS AI — Final Thesis Technical Report

**Author:** Chanuka Sandun
**Repository:** `chanuka8/argus-gait-recognition`
**Audit Date:** 2026-07-23
**Branch:** `main` (commit `4db1632`)

---

## Executive Summary

This report documents a comprehensive evidence-based technical audit of the ARGUS AI Gait Recognition system. The analysis covers the complete repository: architecture, pipeline, model, dataset, evaluation, security, testing, and research contributions.

> [!IMPORTANT]
> **Critical Finding:** The current model checkpoint (`best_model.pth`) was trained on ALL 124 CASIA-B subjects, including the 50 test subjects. This constitutes indirect data leakage for embedding-based evaluation. All performance metrics must be reported as **preliminary baselines** until a clean subject-disjoint checkpoint is trained on subjects 001-074 only. The evaluation infrastructure (leakage validators, split configs, evaluation scripts) is fully operational and ready to produce valid results.

---

## 1. System Overview

ARGUS AI is a **gait recognition surveillance system** designed for missing-person identification. It processes video feeds from cameras, extracts walking patterns as Gait Energy Images (GEI), embeds them using a lightweight CNN, and matches them against an enrolled gallery using cosine similarity.

### Core Pipeline (13 Stages)

```
Video → YOLOv8 Detection → ByteTrack Tracking → Box Stabilization →
Person Crop → Silhouette Extraction → GEI Accumulation →
CNN Embedding → Gallery Matching → Adaptive Decision →
Prediction Smoothing → Security Logging → Detection Reporting
```

**Full details:** [03_gait_pipeline.md](file:///e:/ARGUS_AI/docs/thesis_audit/03_gait_pipeline.md)

---

## 2. Model Architecture

### ByGaitLight CNN

| Property | Value |
|---|---|
| Input | 1×128×64 grayscale GEI |
| Architecture | 3 Conv blocks (32→64→128) + AdaptiveAvgPool + Linear(128→256) |
| Output | 256-dim L2-normalized embedding |
| Parameters | ~126,144 (backbone only) |
| Training Loss | ArcFace CrossEntropy (s=64, m=0.35) |
| Epochs | 50 |
| Optimizer | Adam (lr=1e-4, CosineAnnealing) |
| Best Val Accuracy | 80.14% (on all 124 classes — not subject-disjoint) |

**Full details:** [04_model_architecture.md](file:///e:/ARGUS_AI/docs/thesis_audit/04_model_architecture.md)

---

## 3. Evaluation Results (Preliminary — Subject Leakage Present)

### Closed-Set Identification (50 test subjects)

| Metric | Value |
|---|---|
| Rank-1 | 86.89% |
| Rank-5 | 93.96% |
| Rank-10 | 95.75% |
| NM Rank-1 | 96.82% |
| BG Rank-1 | 91.23% |
| CL Rank-1 | 72.64% |

### Open-Set Identification (25 known + 25 unknown)

| Metric | Value |
|---|---|
| ROC-AUC | 0.915 |
| EER | 16.88% |
| TAR @ θ=0.9913 | 93.73% |
| FAR @ θ=0.9913 | 36.75% |

### Cross-View Recognition

| Metric | Value |
|---|---|
| Cross-View Avg (excl. same view) | 71.17% |
| Same-View Avg | 86.53% |
| Best Pair (054°→036°) | 90.3% |
| Worst Pair (180°→090°) | 54.7% |

### Inference Performance

| Metric | Value |
|---|---|
| Embedding latency | 0.78 ms (CPU) |
| Embedding FPS | 1,277 (CPU) |
| Full probe latency | 11.20 ms |

> [!CAUTION]
> All accuracy metrics above are preliminary due to the model being trained on all 124 subjects. Re-run evaluation after training on subjects 001-074 only.

**Full details:** [10_performance_metrics.md](file:///e:/ARGUS_AI/docs/thesis_audit/10_performance_metrics.md), [15_cross_view_and_openset.md](file:///e:/ARGUS_AI/docs/thesis_audit/15_cross_view_and_openset.md)

---

## 4. Dataset and Leakage Analysis

| Property | Value |
|---|---|
| Dataset | CASIA-B (124 subjects, 11 views, 3 conditions) |
| Subject Split | Train 001-062, Val 063-074, Test 075-124 |
| Leakage Validators | 3 automated assertions (subject, path, calibration) |
| Current Checkpoint | Trained on ALL 124 subjects (**LEAKAGE**) |
| Clean Checkpoint | **Does not exist** — must be retrained |
| Threshold Calibration | Val-only (063-074), θ=0.9913 at min-EER |

**Full details:** [05_dataset_and_preprocessing.md](file:///e:/ARGUS_AI/docs/thesis_audit/05_dataset_and_preprocessing.md)

---

## 5. Matching and Decision Logic

### Adaptive Hybrid Decision Policy

| Score Range | Decision | Verification Method |
|---|---|---|
| ≥ 0.92 | CONFIRMED_MATCH | Flat cosine similarity |
| 0.85 – 0.92 (agrees) | VERIFIED_MATCH | Centroid + margin + top-k consensus |
| 0.85 – 0.92 (disagrees) | REVIEW_REQUIRED | Centroid verification fails |
| 0.70 – 0.85 | LOW_CONFIDENCE | Score-only |
| < 0.70 | UNKNOWN_PERSON | Below all thresholds |

Plus temporal prediction smoothing (voting window=10, min votes=3).

**Full details:** [06_gallery_and_matching.md](file:///e:/ARGUS_AI/docs/thesis_audit/06_gallery_and_matching.md)

---

## 6. Security Assessment

### Implemented Controls

| Control | Implementation | Status |
|---|---|---|
| Security severity classification | `security_engine.py` | ✅ Implemented |
| Thread-safe audit CSV logging | `security_logger.py` | ✅ Implemented |
| Evidence management + retention | `evidence_manager.py` | ✅ Implemented |
| Missing person watchlist + alerts | `missing_person_workflow.py` | ✅ Implemented |

### Not Implemented

| Control | Status |
|---|---|
| Authentication (API/CLI) | ❌ |
| Authorization (RBAC/ABAC) | ❌ |
| Template encryption | ❌ |
| Tamper-proof logging | ❌ |
| Biometric template protection | ❌ |
| Adversarial robustness | ❌ |

**Full details:** [08_security_and_privacy.md](file:///e:/ARGUS_AI/docs/thesis_audit/08_security_and_privacy.md)

---

## 7. Research Contributions

| # | Contribution | Novelty |
|---|---|---|
| C1 | End-to-end gait recognition with security-aware architecture | Integration |
| C2 | Adaptive hybrid matching decision policy | Moderate |
| C3 | Privacy-preserving silhouette-to-GEI pipeline | Engineering |
| C4 | Multi-camera gait recognition with isolated per-camera state | Engineering |
| C5 | Subject-disjoint evaluation framework with leakage detection | Moderate |
| C6 | Structured security audit logging | Engineering |
| C7 | Missing person surveillance workflow | Integration |

**Full details:** [12_research_contributions.md](file:///e:/ARGUS_AI/docs/thesis_audit/12_research_contributions.md)

---

## 8. Critical Actions Required

| # | Action | Priority | Impact |
|---|---|---|---|
| 1 | **Retrain model on subjects 001-074 only** | CRITICAL | Produces clean, defensible evaluation results |
| 2 | **Re-run all evaluations with clean checkpoint** | CRITICAL | Generates valid thesis metrics |
| 3 | **Add random seed to training** | HIGH | Reproducibility |
| 4 | **Install full dependencies in venv** | HIGH | Runtime capability |
| 5 | **Run full pytest suite** | MEDIUM | Verify test status |
| 6 | **Document hardware specifications** | LOW | Reproducibility |

---

## 9. Component Status Summary

| Category | Implemented | Placeholder | Not Present |
|---|---|---|---|
| Pipeline stages | 13/13 | 0 | 0 |
| Model architecture | 1/1 | 1 (gait_encoder) | 0 |
| Evaluation scripts | 7/7 | 0 | 0 |
| Security components | 4/4 | 0 | Auth, RBAC, encryption |
| Intelligence components | 4/7 | 3 | 0 |
| Automation | 0/6 | 6/6 | 0 |
| Monitoring | 3/7 | 4/7 | 0 |
| Tests | 13+ files | 0 | API tests, integration tests |

**Total placeholder files:** 14 (all 64-byte stubs)

---

## 10. Thesis Validity Verdict

| Aspect | Assessment |
|---|---|
| **System design** | ✅ Well-architected, modular, production-oriented |
| **Implementation quality** | ✅ Good code quality, thread-safe, configurable |
| **Model architecture** | ✅ Appropriate for undergraduate thesis scope |
| **Evaluation methodology** | ⚠️ Framework is excellent; current results have leakage |
| **Security analysis** | ⚠️ Audit logging implemented; most controls are planned/future |
| **Test coverage** | ⚠️ Good structure; could not verify execution |
| **Reproducibility** | ⚠️ Good tooling; needs seed control and clean checkpoint |
| **Academic contribution** | ✅ Multiple moderate/engineering contributions identified |
| **Limitations disclosure** | ✅ 18 limitations documented; honest assessment |

### Overall Recommendation

The ARGUS AI system is a **well-engineered undergraduate thesis project** with strong system design and implementation quality. The **single critical action** is retraining the model on subjects 001-074 only and re-running all evaluations to produce clean, defensible thesis results. The evaluation infrastructure is ready for this — no code changes are needed, only a clean training run.

---

## 11. Report Index

| Report File | Content |
|---|---|
| [01_repository_audit.md](file:///e:/ARGUS_AI/docs/thesis_audit/01_repository_audit.md) | Repository structure, environment, dependencies |
| [02_system_architecture.md](file:///e:/ARGUS_AI/docs/thesis_audit/02_system_architecture.md) | Architecture diagrams and technology stack |
| [03_gait_pipeline.md](file:///e:/ARGUS_AI/docs/thesis_audit/03_gait_pipeline.md) | 13-stage pipeline analysis |
| [04_model_architecture.md](file:///e:/ARGUS_AI/docs/thesis_audit/04_model_architecture.md) | CNN architecture and training details |
| [05_dataset_and_preprocessing.md](file:///e:/ARGUS_AI/docs/thesis_audit/05_dataset_and_preprocessing.md) | Dataset, split, and leakage analysis |
| [06_gallery_and_matching.md](file:///e:/ARGUS_AI/docs/thesis_audit/06_gallery_and_matching.md) | Gallery, matching, and decision logic |
| [07_multi_camera_architecture.md](file:///e:/ARGUS_AI/docs/thesis_audit/07_multi_camera_architecture.md) | Multi-camera and surveillance systems |
| [08_security_and_privacy.md](file:///e:/ARGUS_AI/docs/thesis_audit/08_security_and_privacy.md) | Security controls and STRIDE threat model |
| [09_testing_and_code_quality.md](file:///e:/ARGUS_AI/docs/thesis_audit/09_testing_and_code_quality.md) | Tests, CI/CD, code quality |
| [10_performance_metrics.md](file:///e:/ARGUS_AI/docs/thesis_audit/10_performance_metrics.md) | All evaluation results |
| [11_objective_rq_mapping.md](file:///e:/ARGUS_AI/docs/thesis_audit/11_objective_rq_mapping.md) | Objective and research question mapping |
| [12_research_contributions.md](file:///e:/ARGUS_AI/docs/thesis_audit/12_research_contributions.md) | 7 research contributions |
| [13_limitations_and_gaps.md](file:///e:/ARGUS_AI/docs/thesis_audit/13_limitations_and_gaps.md) | 18 limitations, 12 research gaps |
| [14_reproducibility.md](file:///e:/ARGUS_AI/docs/thesis_audit/14_reproducibility.md) | Evidence chains and traceability |
| [15_cross_view_and_openset.md](file:///e:/ARGUS_AI/docs/thesis_audit/15_cross_view_and_openset.md) | Cross-view and open-set analysis |
| [16_algorithm_reference.md](file:///e:/ARGUS_AI/docs/thesis_audit/16_algorithm_reference.md) | Mathematical formulas |
| [17_thesis_writing_guide.md](file:///e:/ARGUS_AI/docs/thesis_audit/17_thesis_writing_guide.md) | Chapter mapping and writing guide |
| FINAL_THESIS_TECHNICAL_REPORT.md | This master summary |
