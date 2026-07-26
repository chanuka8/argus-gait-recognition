# Phase 12 — Limitations and Research Gaps

## 13.1 Confirmed Limitations

| # | Limitation | Category | Severity | Evidence |
|---|---|---|---|---|
| L1 | **Subject leakage in current checkpoint** | Methodology | CRITICAL | `metrics.json`: 124 classes, 13544 samples |
| L2 | **Non-reproducible train/val split** | Methodology | MEDIUM | `dataloader.py` uses `random_split` without seed |
| L3 | **Silhouette quality degrades in complex backgrounds** | Algorithm | HIGH | Otsu thresholding is environment-dependent |
| L4 | **Coat wearing significantly reduces accuracy (72.64%)** | Algorithm | HIGH | `closed_set_eval_report.json` CL condition |
| L5 | **Cross-view accuracy drops sharply for extreme angles** | Algorithm | HIGH | Best 90.3% vs worst 54.7% |
| L6 | **No template encryption** | Security | HIGH | Gallery stored as plaintext NumPy files |
| L7 | **No authentication or access control** | Security | HIGH | API has no auth middleware |
| L8 | **No adversarial robustness** | Security | MEDIUM | No adversarial defence implemented |
| L9 | **Manual threshold selection** | Methodology | MEDIUM | Thresholds hand-tuned, not learned |
| L10 | **No early stopping in training** | Methodology | LOW | Training runs all epochs regardless |
| L11 | **CPU-only training and inference** | Performance | LOW | No GPU utilization code |
| L12 | **Requires ≥10 walking frames for GEI** | Algorithm | MEDIUM | LiveGEI `min_frames` default is 10 |
| L13 | **No data augmentation during training** | Methodology | MEDIUM | No transforms applied during training |
| L14 | **No dropout regularization** | Architecture | LOW | ByGaitLight has no dropout layers |
| L15 | **14 placeholder modules** | Completeness | MEDIUM | Multiple 64-byte stubs remain |
| L16 | **Cross-camera tracking not integration-tested** | Testing | MEDIUM | Unit tests only, no end-to-end validation |
| L17 | **Webcam limitations for full gait capture** | Deployment | HIGH | Upper-body webcam cannot capture full gait cycle |
| L18 | **No liveness detection** | Security | MEDIUM | System processes pre-recorded video without verification |

## 13.2 Research Gaps

| # | Gap | Description | Potential Solution | Priority |
|---|---|---|---|---|
| G1 | **No biometric template protection** | Raw embeddings stored without any protection scheme | Implement cancelable biometrics, secure sketch, or homomorphic encryption | HIGH |
| G2 | **No multi-modal fusion** | System uses gait only; no fusion with face, appearance, or body shape | Implement score-level or feature-level fusion | MEDIUM |
| G3 | **No attention mechanism** | ByGaitLight uses basic CNN without attention | Add spatial/channel attention (CBAM, SE blocks) | MEDIUM |
| G4 | **No part-based analysis** | GEI treated as holistic image | Split GEI into upper/lower body regions | MEDIUM |
| G5 | **No temporal modelling** | GEI averages away temporal dynamics | Use LSTM, Transformer, or 3D CNN on silhouette sequences | HIGH |
| G6 | **No unsupervised domain adaptation** | Model performance may degrade across domains | Apply domain adaptation techniques for deployment environments | MEDIUM |
| G7 | **No Extreme Value Theory (EVT) for open-set** | Open-set threshold is static | Use Weibull fitting for calibrated open-set rejection | HIGH |
| G8 | **No model interpretability** | No attention maps, Grad-CAM, or saliency visualization | Add interpretability tools for understanding which body parts drive recognition | LOW |
| G9 | **No federated learning** | Privacy-preserving distributed training not explored | Explore federated learning for multi-site deployment | LOW |
| G10 | **No edge deployment optimization** | Model not exported to ONNX, TensorRT, or quantized | Convert to optimized formats for edge devices | MEDIUM |
| G11 | **No cross-dataset evaluation** | Evaluated on CASIA-B only | Test on OU-MVLP, USF, or GREW datasets | MEDIUM |
| G12 | **No clothing-invariant features** | CL condition shows significant accuracy drop | Research clothing-invariant gait representations | HIGH |

## 13.3 Threats to Validity

### Internal Validity

| Threat | Description | Mitigation |
|---|---|---|
| **Subject leakage** | Model trained on test subjects | Evaluation infrastructure ready; retraining required |
| **Threshold calibration** | Threshold calibrated on val set only (792 probes) | Val set is small (12 subjects); larger calibration set recommended |
| **Non-reproducible splits** | Random dataloader split without seed | Does not affect subject-disjoint eval (which uses explicit config) |
| **Single training run** | Only one training run recorded | No variance or confidence intervals available |
| **No cross-validation** | Single fixed split used | K-fold cross-validation would provide more robust estimates |

### External Validity

| Threat | Description | Mitigation |
|---|---|---|
| **Single dataset** | CASIA-B only (controlled indoor environment) | Results may not generalize to real-world CCTV |
| **Known subjects** | All subjects are cooperative walkers | Performance on uncooperative subjects unknown |
| **Indoor setting** | CASIA-B is indoor with controlled lighting | Outdoor environments with shadows, weather may degrade performance |
| **Fixed camera angles** | 11 discrete angles | Continuous angle variation not tested |
| **Limited conditions** | 3 conditions (NM, BG, CL) | Real-world has more variation (shoes, speed, terrain) |

### Construct Validity

| Threat | Description | Mitigation |
|---|---|---|
| **Cosine similarity as metric** | Assumes linear separability in embedding space | Standard metric for embedding-based biometrics |
| **Rank-k as primary metric** | Does not capture open-set rejection quality | Supplemented with ROC-AUC and EER |
| **Manual threshold semantics** | Decision labels (CONFIRMED, REVIEW, etc.) are subjective | Thresholds derived from EER calibration |

## 13.4 What Can Be Validly Claimed in the Thesis

### Safely Claimable

1. ✅ A working end-to-end gait recognition system was designed and implemented
2. ✅ The system supports single and multi-camera modes
3. ✅ GEI-based representations provide privacy-by-design
4. ✅ An adaptive hybrid matching policy reduces false positives
5. ✅ A subject-disjoint evaluation framework with leakage detection was built
6. ✅ Security audit logging provides accountability
7. ✅ The architecture supports real-time CPU-based inference (~1277 FPS embedding)
8. ✅ Preliminary evaluation demonstrates the pipeline's functional correctness

### Cannot Be Claimed (Without Additional Evidence)

1. ❌ The system achieves [specific %] under subject-disjoint evaluation (current results have leakage)
2. ❌ The system is Zero-Trust compliant
3. ❌ Biometric templates are protected
4. ❌ The system has been validated in real-world CCTV deployment
5. ❌ The security architecture prevents insider threats
6. ❌ Results generalize beyond CASIA-B
7. ❌ The model outperforms state-of-the-art methods
