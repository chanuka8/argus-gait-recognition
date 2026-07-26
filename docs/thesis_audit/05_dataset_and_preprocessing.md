# Phase 4 — Dataset and Preprocessing Audit

## 5.1 Dataset Identity

| Property | Value |
|---|---|
| **Dataset** | CASIA-B Gait Dataset |
| **Source** | Institute of Automation, Chinese Academy of Sciences |
| **Total Subjects** | 124 |
| **Conditions** | NM (Normal Walking), BG (Carrying Bag), CL (Wearing Coat) |
| **Sequences per Subject** | 10 (6 NM + 2 BG + 2 CL) |
| **View Angles** | 11 (0°, 18°, 36°, 54°, 72°, 90°, 108°, 126°, 144°, 162°, 180°) |
| **Total GEI Samples** | ~13,544 (varying per subject due to missing sequences) |
| **Local Path** | `data/casia_processed/gei/` |
| **Raw Archive** | `data/casia_b_raw.zip` (763 MB) |

## 5.2 Subject-Disjoint Split

**Source:** `configs/subject_split.json`

| Split | Subject Range | Count | Total GEI Samples |
|---|---|---|---|
| **Training** | 001–062 | 62 | ~6,780 |
| **Validation** | 063–074 | 12 | ~1,309 |
| **Test** | 075–124 | 50 | ~5,456 |
| **Total** | 001–124 | 124 | ~13,544 |

### Disjointness Verification

**Evidence:** `evaluation/leakage_validator.py::assert_subject_disjointness()` verifies:
- No overlap between train and val sets: ✅ Verified
- No overlap between train and test sets: ✅ Verified
- No overlap between val and test sets: ✅ Verified

**Protocol:** `"CASIA-B Subject-Disjoint Standard Partition (001-074 Train/Val, 075-124 Test)"`

> [!CAUTION]
> **Critical Finding: The split configuration is correctly defined, but the actual model checkpoint (`best_model.pth`) was NOT trained using this split.** The `metrics.json` file records `num_classes: 124` and `samples: 13544`, proving the model was trained on ALL 124 subjects. A clean subject-disjoint checkpoint trained on only subjects 001-062 (train) + 063-074 (val) does **not exist** in the repository.

## 5.3 Dataset Folder Structure

```
data/casia_processed/gei/
├── 001/
│   ├── 001_nm-01_000.png
│   ├── 001_nm-01_018.png
│   ├── 001_nm-01_036.png
│   ├── ...
│   ├── 001_cl-02_180.png
│   └── (110 files total)
├── 002/
│   └── (110 files)
├── ...
└── 124/
    └── (110 files)
```

**Filename Convention:** `{subject_id}_{condition}-{seq_num}_{angle}.png`

Example: `075_bg-02_090.png` = Subject 075, Bag condition, Sequence 2, 90° angle

## 5.4 Gallery/Probe Protocol

**Source:** `evaluation/gallery_probe_builder.py`

| Set | Sequences | Purpose |
|---|---|---|
| **Gallery** | nm-01, nm-02, nm-03, nm-04 | Enrolled templates (4 normal walks) |
| **Probe NM** | nm-05, nm-06 | Test normal walking probes |
| **Probe BG** | bg-01, bg-02 | Test bag-carrying probes |
| **Probe CL** | cl-01, cl-02 | Test coat-wearing probes |

- Sequences are strictly non-overlapping between gallery and probe
- `assert_gallery_probe_disjointness()` validates no path overlap
- Gallery view filtering is optional (default: all views in gallery)

## 5.5 Sample Counts per Subject (selected examples)

| Subject | Samples | Notes |
|---|---|---|
| 001 | 110 | Full |
| 005 | 91 | Missing sequences |
| 026 | 109 | Missing 1 |
| 037 | 100 | Missing 10 |
| 048 | 100 | Missing 10 |
| 068 | 89 | Missing 21 (val set) |
| 079 | 105 | Missing 5 (test set) |
| 088 | 94 | Missing 16 (test set) |
| 096 | 108 | Missing 2 (test set) |
| 109 | 99 | Missing 11 (test set) |

## 5.6 Preprocessing Pipeline

| Step | Method | File |
|---|---|---|
| 1. Raw video extraction | CASIA-B ZIP extraction | `preprocessing/casia_extractor.py` |
| 2. Silhouette extraction | Background subtraction / Otsu | `preprocessing/silhouette_extractor.py` |
| 3. GEI generation | Frame averaging per sequence | `preprocessing/gei_builder.py` |
| 4. Dataset building | Organize by subject/sequence | `preprocessing/dataset_builder.py` |
| 5. Augmentation | Horizontal flip, rotation, noise | `preprocessing/augmentation.py` |
| 6. Skeleton extraction | Placeholder only | `preprocessing/skeleton_extractor.py` (64 bytes) |

## 5.7 GEI Dataset Loader

**File:** `training/dataset.py::GEIDataset`

| Property | Value |
|---|---|
| **Root directory** | `data/casia_processed/gei` |
| **Image size** | `(64, 128)` (width × height) |
| **Channels** | 1 (grayscale) |
| **Normalization** | Divide by 255.0 to [0, 1] |
| **Label encoding** | Sequential integer index per subject directory |
| **File format** | PNG |
| **Scan method** | Sorted directory iteration → sorted glob("*.png") |

## 5.8 Dataloader Configuration

**File:** `training/dataloader.py`

| Property | Value |
|---|---|
| **Train/Val split ratio** | 80% / 20% (random) |
| **Split method** | `torch.utils.data.random_split` |
| **Batch size** | 16 |
| **Shuffle** | Train: Yes, Val: No |
| **Num workers** | 0 |
| **Random seed** | **Not set** (non-reproducible split) |

> [!WARNING]
> The dataloader uses `random_split` without a seed, meaning the train/val split within the dataset is **not reproducible** across runs. This is separate from the subject-disjoint split (which IS deterministic via `subject_split.json`).

## 5.9 Data Leakage Risk Assessment

| Risk | Severity | Status | Evidence |
|---|---|---|---|
| **Subject leakage in current checkpoint** | **CRITICAL** | **CONFIRMED** | `metrics.json` shows 124 classes, 13544 samples |
| **Gallery/probe sequence overlap** | None | **Mitigated** | `gallery_probe_builder.py` enforces strict sequence separation |
| **Gallery/probe path overlap** | None | **Mitigated** | `leakage_validator.py::assert_gallery_probe_disjointness()` |
| **Train/test subject overlap in eval** | None | **Mitigated** | `leakage_validator.py::assert_subject_disjointness()` |
| **Threshold calibration leakage** | None | **Mitigated** | `threshold_calibration.json` uses val subjects 063-074 only |
| **Non-reproducible random split** | LOW | Present | `random_split` without seed in `dataloader.py` |
| **Duplicate samples** | LOW | Unknown | No explicit deduplication check |

## 5.10 Impact on Thesis Validity

The current checkpoint was trained on all 124 subjects. However, the **evaluation pipeline** uses the model as an **embedding extractor** (backbone only, no classifier head). The evaluation is performed on test subjects 075-124 using gallery sequences (nm-01 to nm-04) and probe sequences (nm-05/06, bg-01/02, cl-01/02) from those **test subjects only**.

**Key question:** Does training the backbone on test subjects inflate embedding-based evaluation metrics?

**Analysis:**
1. The backbone learned features from test subjects' walking patterns during training
2. Even without the classifier head, the embedding space may have been optimized to separate test subjects
3. This constitutes **indirect leakage** — the backbone has seen test subjects' GEIs during training
4. A properly trained model should use subjects 001-074 only for training, then evaluate on 075-124

**Thesis Recommendation:**
- Report the current results as **"preliminary baseline with known indirect leakage"**
- Clearly state that the model was trained on all subjects
- Note that a clean subject-disjoint retraining is required for final thesis results
- The evaluation pipeline infrastructure (leakage validators, split configs) demonstrates the capability to produce clean results
