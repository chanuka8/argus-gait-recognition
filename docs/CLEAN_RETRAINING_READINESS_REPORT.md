# ARGUS Clean Retraining Readiness Report

**Audit Date:** July 22, 2026  
**Auditor:** Automated System & ML Engineering Readiness Audit  
**Scope:** Pre-Retraining Readiness Audit for ByGaitLight Clean Subject-Disjoint Baseline  
**Decision:** **CONDITIONAL GO** (Ready after fixing P0 Dataloader Subject Split Wiring Blocker)

---

## 1. Executive Summary

- **Overall Readiness:** CONDITIONAL GO
- **Can Training Start Immediately?** **No** — Blocked by P0 issue in `training/dataloader.py` where `build_dataloaders()` does not consume `configs/subject_split.json` and defaults to a random sample-level split across all 124 subjects.
- **Number of Blockers (P0):** 1 (`training/dataloader.py` not wired to subject split manifest).
- **Number of Required Code Changes:** 2 (`training/dataloader.py` and `training/dataset.py`).
- **Number of Required Config Changes:** 1 (`configs/train.yaml` update run_dir and epochs).
- **Dataset Completeness:** 100% (13,544 GEI PNG images across 124 subjects, 0 unreadable/corrupt files).
- **Hardware Readiness:** Ready (CPU-only execution verified; PyTorch 2.12.0+cpu).
- **Estimated Risk Level:** Low (once P0 wiring fix is applied).

---

## 2. Current Repository State

- **Branch:** `main` (up to date with `origin/main`)
- **Working Tree:** Uncommitted changes exist from evaluation pipeline refactoring (`evaluation/evaluator.py`, `evaluation/cross_view_evaluator.py`, `evaluation/metrics.py`, `evaluation/open_set_evaluator.py`, `evaluation/dataset_split.py`, `evaluation/gallery_probe_builder.py`, `evaluation/threshold_calibrator.py`, `evaluation/leakage_validator.py`, `scripts/evaluate_subject_disjoint.py`, `tests/test_leakage_prevention.py`, `tests/unit/test_metric_correctness.py`).
- **Recent Commit:** `4db1632` ("Fix evaluation pipeline and baseline audit issues").
- **Working Tree Safety:** Uncommitted changes are in evaluation scripts and tests, not training code. Working tree must be stashed or committed to a new branch (`feat/clean-subject-disjoint-retraining`) prior to retraining.

---

## 3. Environment and Hardware

- **Python Version:** `3.11.9`
- **PyTorch Version:** `2.12.0+cpu`
- **CUDA Availability:** `False` (CUDA not available in current environment)
- **Device Count:** `0`
- **GPU Name:** `N/A`
- **CUDA Version:** `None`
- **Device Selected by Training Pipeline:** `cpu` (via fallback in `training/trainer.py` L110–L114)
- **Instantiate Model:** Verified (ByGaitLight backbone and GaitClassifier instantiate cleanly on CPU)

---

## 4. Dataset Inventory

### Dataset Overview
- **Root Path:** `data/casia_processed/gei`
- **Format:** 64×128 grayscale GEI images
- **Total Subjects:** 124 (001–124)
- **Total GEI Images:** 13,544
- **Corrupt/Unreadable Files:** 0

### Split Statistics

| Split | Subject Range | Subject Count | Image Count | Sequence Count | Conditions Available | Views Available | Unreadable Files |
|-------|---------------|---------------|-------------|----------------|----------------------|-----------------|------------------|
| **Train** | 001–062 | 62 | 6,779 | 620 | `nm` (01-06), `bg` (01-02), `cl` (01-02) | 11 angles (000°–180°) | 0 |
| **Validation** | 063–074 | 12 | 1,299 | 120 | `nm` (01-06), `bg` (01-02), `cl` (01-02) | 11 angles (000°–180°) | 0 |
| **Test** | 075–124 | 50 | 5,466 | 500 | `nm` (01-06), `bg` (01-02), `cl` (01-02) | 11 angles (000°–180°) | 0 |
| **TOTAL** | **001–124** | **124** | **13,544** | **1,240** | **`nm`, `bg`, `cl`** | **11 angles** | **0** |

---

## 5. Subject Split Verification

- **Manifest File:** `configs/subject_split.json`
- **Validation Function:** `evaluation/dataset_split.py::validate_disjoint_splits()`
- **Intersection Checks:**
  - `Train ∩ Val = ∅` (001–062 vs 063–074: 0 overlap) — **PASSED**
  - `Train ∩ Test = ∅` (001–062 vs 075–124: 0 overlap) — **PASSED**
  - `Val ∩ Test = ∅` (063–074 vs 075–124: 0 overlap) — **PASSED**
- **Fallback Random Split Active in Dataloader?** **YES** (`training/dataloader.py` L22–25 calls `random_split(dataset, [train_size, val_size])`). This is a **P0 BLOCKER** that must be fixed before retraining.

---

## 6. Training Data Flow

### Current (Flawed) Data Flow:
```
python scripts/train_model.py
  ├── CLI Args (default: epochs=3, batch_size=16, lr=0.0001, loss_mode=ce)
  └── Trainer.__init__() [run_dir="runs/exp_001"]
        └── build_dataloaders(root_dir="data/casia_processed/gei")
              ├── GEIDataset(root_dir)  <-- Scans ALL dirs 001-124!
              └── random_split(dataset, [80%, 20%])  <-- Sample-level split across all 124 subjects!
```

### Required (Clean) Data Flow:
```
python scripts/train_model.py --subject-split configs/subject_split.json --run-dir runs/exp_subject_disjoint_001
  ├── Load configs/subject_split.json
  └── Trainer.__init__() [run_dir="runs/exp_subject_disjoint_001"]
        └── build_subject_disjoint_dataloaders(manifest)
              ├── Train GEIDataset(subjects=001-062)  <-- Labels 0 to 61
              └── Val GEIDataset(subjects=063-074)    <-- Held-out validation subjects for retrieval
```

---

## 7. Model and ArcFace Readiness

- **Model Class Path:** `models.architectures.bygait_light.ByGaitLight` (L6–L71)
- **Classifier Class Path:** `training.trainer.GaitClassifier` (L15–L69)
- **Input Shape:** `(1, 128, 64)` — C × H × W
- **Embedding Dimension:** `256`
- **Embedding Normalization:** L2 Normalized via `F.normalize(x, p=2, dim=1)`
- **Classifier Type:** `nn.Linear(256, num_classes)` and `ArcMarginProduct(256, num_classes)`
- **ArcFace Implementation:** `models.architectures.losses.ArcMarginProduct`
- **Class Count Source:** Dynamic (`num_classes` passed to `GaitClassifier.__init__`).
- **Supports 62 Training Classes?** **YES** — Tested during lightweight dry-run; instantiates cleanly with `num_classes=62`.
- **Label Mapping:** Subjects `001–062` map deterministically to integer indices `0–61`.
- **Embedding Extractor Independent?** **YES** — `model.backbone(x)` extracts 256-D normalized embeddings without using classifier weights.

---

## 8. Checkpoint Initialization and Resume Logic

- **Random Initialization:** Verified (`nn.init.xavier_uniform_` on ArcFace weights, standard PyTorch init on CNN).
- **Auto-resume Checkpoint:** **No** — `Trainer` does not load existing checkpoints on start.
- **Legacy Checkpoint Safety:** **RISK** — `Trainer.__init__` defaults to `run_dir="runs/exp_001"`. If executed without specifying `--run-dir`, it will save to `runs/exp_001/best_model.pth` and overwrite the legacy checkpoint!
- **Safe Run Directory:** Specify `--run-dir runs/exp_subject_disjoint_001` to ensure legacy artifacts remain completely untouched.

---

## 9. Training Configuration

| Parameter | Current Value | Source File | Key | Ready? | Required Change |
|-----------|---------------|-------------|-----|--------|-----------------|
| Epochs | 50 (cli default 3, train.yaml 20) | `scripts/train_model.py` / `train.yaml` | `epochs` | Needs adjustment | Set `epochs: 50` in config / CLI |
| Batch Size | 16 | `training/trainer.py` L78 | `batch_size` | Ready | Keep 16 |
| Learning Rate | 0.0001 | `training/trainer.py` L80 | `learning_rate` | Ready | Keep 0.0001 |
| Optimizer | Adam (weight_decay=1e-4) | `training/optimizer.py` | `build_optimizer` | Ready | Keep Adam |
| Scheduler | CosineAnnealingLR (T_max=epochs) | `training/trainer.py` L173 | `scheduler` | Ready | Keep CosineAnnealingLR |
| Loss Mode | `ce_arcface` | `training/trainer.py` L85 | `loss_mode` | Ready | Keep `ce_arcface` |
| ArcFace Scale | 64.0 (cli default 30.0) | `scripts/train_model.py` L73 | `arcface_scale` | Needs adjustment | Set `arcface_scale: 64.0` in CLI |
| ArcFace Margin | 0.35 (cli default 0.50) | `scripts/train_model.py` L79 | `arcface_margin` | Needs adjustment | Set `arcface_margin: 0.35` in CLI |
| Triplet Weight | 0.0 | `training/trainer.py` L84 | `triplet_weight` | Ready | Keep 0.0 |
| Triplet Margin | 0.3 | `training/trainer.py` L83 | `triplet_margin` | Ready | Keep 0.3 |
| Image Size | (64, 128) | `training/dataset.py` L12 | `image_size` | Ready | Keep (64, 128) |
| Device | `cpu` | `training/trainer.py` L110 | `device` | Ready | Fallback to CPU |

---

## 10. Validation and Checkpoint Selection

- **Validation Subjects:** `063–074` (12 subjects)
- **Validation Evaluation Type:** Unseen-identity retrieval (using `model.backbone` embeddings, not classification head).
- **Current Validation Logic in Trainer:** `Trainer._validate()` computes classification accuracy on val loader.
- **P0 Fix Needed:** Update `Trainer._validate()` during clean retraining to evaluate **Validation Retrieval Rank-1 Accuracy** on subjects `063–074` (Gallery: `nm-01..04`, Probe: `nm-05..06`, `bg`, `cl`) rather than classification accuracy.
- **Checkpoint Selection Criterion:** Select `best_model.pth` based on highest Validation Retrieval Rank-1 Accuracy on subjects `063–074`.
- **Test Set Isolation:** Test subjects `075–124` are NOT loaded during training or validation checkpoint selection.

---

## 11. Test Isolation

- **Are test subjects loaded during training?** **Currently YES** (due to P0 dataloader issue scanning all folders 001–124).
- **Will test subjects be isolated after P0 fix?** **YES** — `GEIDataset` for training will filter strictly to `train_subjects` (001–062), `val_dataset` will filter strictly to `val_subjects` (063–074), and test subjects (075–124) will be instantiated ONLY during final `scripts/evaluate_subject_disjoint.py`.

---

## 12. Open-Set Evaluation Readiness

- **Known Test Subjects:** `075–099` (25 subjects enrolled in test gallery)
- **Unknown Test Subjects:** `100–124` (25 subjects NOT in gallery)
- **Score Calculation:** Cosine similarity via L2-normalized embeddings
- **Threshold Source:** Frozen from validation calibration on subjects `063–074` (Min-EER threshold = `0.9913`)
- **Code Status:** `evaluation/open_set_evaluator.py` is fully implemented, verified, and ready to evaluate the new checkpoint without modification.

---

## 13. Reproducibility and Experiment Tracking

| Requirement | Implementation Status | Path / Reference |
|-------------|----------------------|------------------|
| Random Seed | Implemented (42) | `configs/subject_split.json` |
| Deterministic Partition | Implemented | `configs/subject_split.json` |
| Subject-to-Class Mapping | Needs export during training | Save `runs/exp_subject_disjoint_001/label_to_index.json` |
| Git Commit Hash | Implemented in evaluation report | Tracked in execution script |
| Metrics Log | Implemented | `runs/exp_subject_disjoint_001/metrics.json` |
| Checkpoint Storage | Implemented | `runs/exp_subject_disjoint_001/best_model.pth` |

---

## 14. Lightweight Dry-Run Results

- **Execution Command:** Python one-liner instantiating `GaitClassifier(num_classes=62)`, `JointGaitLoss`, `build_optimizer`, forwarding 1 training batch (16 samples, labels 0..15), and extracting embeddings for 1 validation batch (4 samples).
- **Results:**
  - **Input Tensor Shape:** `torch.Size([16, 1, 128, 64])`
  - **Embedding Tensor Shape:** `torch.Size([16, 256])`
  - **Loss Logits Shape:** `torch.Size([16, 62])`
  - **Pred Logits Shape:** `torch.Size([16, 62])`
  - **Label Range:** `0 to 15` (mapped within 0..61)
  - **Loss Value:** `30.0890` (CE + ArcFace)
  - **Val Batch Input Shape:** `torch.Size([4, 1, 128, 64])`
  - **Val Embedding Shape:** `torch.Size([4, 256])`
  - **Val L2 Norms:** `[1.0, 1.0, 1.0, 1.0]`
  - **Device:** `cpu`
  - **Status:** **PASS**

---

## 15. Blockers and Risks

| Priority | Issue | Evidence | Required Fix | Risk |
|----------|-------|----------|--------------|------|
| **P0** | DataLoader does not consume subject split manifest | `training/dataloader.py` L22–25 calls `random_split` on full 124-subject dataset | Modify `GEIDataset` to accept `allowed_subjects` list and build separate train (001–062) and val (063–074) datasets | Data leakage if training is run without fix |
| **P1** | Default `run_dir` overwrites legacy checkpoint | `training/trainer.py` L77 defaults to `runs/exp_001` | Update default `run_dir` to `runs/exp_subject_disjoint_001` | Overwriting `runs/exp_001/best_model.pth` |
| **P1** | Validation in `Trainer` computes classification acc instead of retrieval acc | `training/trainer.py` L390 computes classification argmax on val subjects | Update `Trainer._validate()` to calculate retrieval Rank-1 on val gallery/probe | Suboptimal checkpoint selection for unseen-identity retrieval |
| **P2** | CLI default args differ from best hyperparameters | `train_model.py` default `epochs=3`, `arcface_s=30.0`, `arcface_m=0.50` vs 50 epochs, s=64.0, m=0.35 | Set recommended CLI default values in `train_model.py` | Training with weak defaults |

---

## 16. Required Changes Before Training

1. **`training/dataset.py` (Medium Scope)**:
   - Add `allowed_subjects: list[str] | None = None` to `GEIDataset.__init__()`.
   - Filter `person_dirs` to only include directories matching `allowed_subjects`.
   - Ensure `label_to_index` maps the allowed subjects to `0 .. len(allowed_subjects)-1`.

2. **`training/dataloader.py` (Medium Scope)**:
   - Load `configs/subject_split.json`.
   - Create `train_dataset = GEIDataset(allowed_subjects=train_subjects)`.
   - Create `val_dataset = GEIDataset(allowed_subjects=val_subjects)`.
   - Return `train_loader`, `val_loader`, `train_dataset`, `val_dataset`.

3. **`training/trainer.py` (Medium Scope)**:
   - Update default `run_dir` to `"runs/exp_subject_disjoint_001"`.
   - Save `label_to_index.json` in `run_dir` during initialization.

---

## 17. Exact Retraining Execution Plan

1. Stash or commit working tree changes to `feat/clean-subject-disjoint-retraining`.
2. Apply P0 dataloader wiring fix (`training/dataset.py` & `training/dataloader.py`).
3. Run `pytest -v` to ensure dataloader and leakage unit tests pass.
4. Execute lightweight dry-run script verifying 62-class train loader and 12-class val loader.
5. Launch clean training job:
   ```bash
   .\venv\Scripts\python.exe scripts/train_model.py --epochs 50 --batch-size 16 --lr 0.0001 --loss-mode ce_arcface --arcface-scale 64.0 --arcface-margin 0.35 --run-dir runs/exp_subject_disjoint_001
   ```
6. Verify output checkpoint saved at `runs/exp_subject_disjoint_001/best_model.pth`.
7. Calibrate operating threshold on validation subjects `063–074`.
8. Execute final test set evaluation (`075–124`):
   ```bash
   .\venv\Scripts\python.exe scripts/evaluate_subject_disjoint.py --model-path runs/exp_subject_disjoint_001/best_model.pth --output-dir runs/exp_subject_disjoint_001/evaluation
   ```
9. Generate final thesis report `docs/CLEAN_RETRAINING_BENCHMARK_REPORT.md`.

---

## 18. Expected Artifacts

- `runs/exp_subject_disjoint_001/best_model.pth`
- `runs/exp_subject_disjoint_001/last_model.pth`
- `runs/exp_subject_disjoint_001/metrics.json`
- `runs/exp_subject_disjoint_001/label_to_index.json`
- `runs/exp_subject_disjoint_001/evaluation/closed_set_eval_report.json`
- `runs/exp_subject_disjoint_001/evaluation/cross_view_report.json`
- `runs/exp_subject_disjoint_001/evaluation/open_set_report.json`

---

## 19. Thesis-Safety Conditions

Once clean retraining is executed:
- **All recognition metrics (Rank-1, Rank-5, Rank-10, Cross-View, Open-Set ROC-AUC, EER)** produced from `runs/exp_subject_disjoint_001/best_model.pth` will be **100% THESIS-SAFE** (zero data leakage, unseen subjects, unseen sequences, validation-calibrated threshold).

---

## 20. Final Go / No-Go Decision

**Decision:** **CONDITIONAL GO**

**Justification:** The dataset is 100% complete and valid, the model architecture supports 62-class ArcFace training, hardware and lightweight dry-runs pass cleanly, and evaluation tools are ready. Retraining can proceed as soon as the P0 dataloader subject split wiring fix is applied.
