# ARGUS AI — Project Structure Audit

**Date:** 2026-09-21
**Scope:** Repository layout only (no code/behavior changes). This is a findings + target-structure report; nothing described here has been applied to the tree.
**Goal:** Assess whether the current directory layout reads as a professional, production-grade surveillance system, and propose a target structure to migrate toward.

---

## 1. Executive Summary

ARGUS AI has mature, well-separated *domain* modules (`intelligence/`, `pipeline/`, `security_layer/`, `streaming/`, `services/`) and a genuinely good test layout (`tests/unit`, `tests/integration`, `tests/system`, `tests/performance`). The core engineering is not the problem.

What undermines the "professional surveillance platform" impression is **top-level hygiene and grouping**: 38 flat top-level entries with no layering, two competing tooling directories, a duplicated/legacy secrets folder, generated artifacts checked in next to source, and documentation that no longer matches the tree it describes. These are exactly the things a security auditor, new hire, or enterprise procurement reviewer notices in the first five minutes.

Severity breakdown: **3 Critical**, **5 High**, **6 Medium**, **4 Low** findings (below), plus a proposed target structure and a phased (non-destructive) migration path.

---

## 2. Current State Snapshot

- 38 top-level entries in the repo root (excluding VCS/cache/venv dirs), no grouping between application code, ML/data platform, ops tooling, and docs.
- `intelligence/` — 48 flat `.py` files, no internal sub-packages, mixing fusion, tracking, crowd analytics, drift/audit, alerting, and continual learning.
- `evaluation/` — ~30 flat experiment/eval scripts at top level *plus* `benchmarks/`, `scripts/`, and `results/` subfolders whose scope overlaps them.
- `models/` — 9 subdirectories (`active`, `candidates`, `rollback`, `engines`, `export`, `gallery`, `appearance_gallery`, `live_gallery`, `architectures`, `inference`, `weights`) with no separation between *model code*, *model artifact lifecycle*, and *runtime galleries*.
- `scripts/` and `tools/` — near-duplicate directories (see Finding C-1).
- `config/` (singular, 1 file) vs `configs/` (plural, the real config directory).
- `docs/reports/` holds machine-generated JSON run reports; `docs/` also holds hand-written architecture docs directly in its root, with only `security/` broken into a subfolder.
- The README's own "Project Structure" diagram omits `scripts/`, `config/`, `runs/`, `outputs/`, `.agents/`, `.qodo/`, `dataconnect/`, and the root `package.json` — i.e., the documented structure has drifted from the actual one.

---

## 3. Findings

### Critical

**C-1. `scripts/` and `tools/` are duplicate, drifting toolchains.**
Both directories contain files with identical names: `activate_venv.ps1`, `manage_venv.ps1`, `dev.js`, `doctor.py`, `start_system.bat`, `start_system.sh`, `export_bygait_onnx.py`, `export_silhouette_unet_onnx.py`, `install_git_hooks.py`, `sync_folder_readmes.py`. The two copies are not identical (e.g., `tools/manage_venv.ps1` is 11.6 KB vs `scripts/manage_venv.ps1` at 610 bytes) — meaning some have already diverged, and it is unclear which one is authoritative or which one `Makefile`/CI actually invokes. `tools/` is additionally the well-organized one (`benchmark/`, `data/`, `maintenance/`, `migration/`, `security/`, `validation/` subpackages); `scripts/` looks like a legacy flat copy left behind after a reorg.
*Why it matters:* in a surveillance/biometric system, "which script is real" ambiguity around security tooling (`tools/security/encrypt_model_artifact.py`, `migrate_gallery_encryption.py`) is a genuine operational risk, not just clutter.

**C-2. Stray `.bak` files are committed to version control.**
`scripts/doctor.py.bak`, `scripts/export_bygait_onnx.py.bak`, `scripts/export_silhouette_unet_onnx.py.bak`, `scripts/install_git_hooks.py.bak`, `scripts/sync_folder_readmes.py.bak`, `tools/benchmark/large_media_ingestion_benchmark.py.bak` are tracked, executable, backup copies of live scripts. This is the kind of thing that fails a professional code-quality/security review outright — it signals no `.gitignore` discipline and leaves ambiguous, unreviewed code paths in the tree.

**C-3. Two competing config/secrets directories: `config/` vs `configs/`.**
`config/` (singular) contains exactly one file: `firebase-service-account.json` — a credential. `configs/` (plural) is the real, populated configuration directory (14 YAML/JSON files) and *also* contains a 1 MB `credentials.enc` and `gallery_probe_manifest.json`/`subject_split.json` (which look like evaluation artifacts, not config). Having a second, near-identically-named directory whose sole purpose is to hold a service-account credential is a naming collision that invites someone to `git add config/` (or a build script to glob `config*/`) and leak a secret. This should be a single, unambiguous `configs/secrets/` (gitignored) location.

### High

**H-1. `models/` conflates three different lifecycles.**
Model *code* (`architectures/`, `inference/`), model *artifact lifecycle* (`weights/`, `active/`, `candidates/`, `rollback/`, `engines/`, `export/`, `model_registry.py`), and *runtime identity galleries* (`gallery/`, `appearance_gallery/`, `live_gallery/`) all sit side by side as siblings. Three separately named "gallery" directories in particular (`gallery/`, `appearance_gallery/`, `live_gallery/`) is confusing even to someone who knows the codebase — it's not obvious which is the source-of-truth enrollment gallery vs. a runtime cache vs. a legacy name.

**H-2. `intelligence/` is a 48-file flat package with no sub-boundaries.**
It mixes at least six distinct concerns as flat files: fusion (`dual_modal_fusion.py`, `fusion_weights.py`, `learned_fusion.py`, `fusion_diagnostics.py`), cross-camera tracking (`cross_camera_tracker.py`, `concurrent_track_manager.py`, `track_recovery_manager.py`, `track_reliability_scorer.py`), crowd analytics (`crowd_density_estimator.py`, `crowd_intelligence_system.py`, `crowd_occlusion_analyzer.py`, `crowd_robustness_manager.py`), continual learning (`continual_learning_audit_trail.py`, `continual_learning_evaluator.py`, `date_aware_learning_scheduler.py`, `nn_fine_tuner.py`), decisioning/policy (`decision_engine.py`, `policy_engine.py`, `human_review_decision.py`), and drift/statistical validation (`drift_detector.py`, `statistical_accuracy_validator.py`, `accuracy_validation_gate.py`). At 48 files this is the single largest and least navigable package in the repo.

**H-3. `evaluation/` has three overlapping "scripts" locations.**
Top-level `evaluation/*.py` (30 files: `evaluate_*.py`, `run_exp*.py`, `analyze_*.py`), plus `evaluation/benchmarks/`, plus `evaluation/scripts/` — all hold runnable evaluation entry points, and it's not discoverable from the names alone which subfolder a new script belongs in.

**H-4. Generated run artifacts are committed inside `docs/`.**
`docs/reports/*.json` (BACKEND_REPORT.json, BENCHMARK_REPORT.json, EVALUATION_REPORT.json, etc.) are machine-generated output, not documentation. `docs/` should hold only human-authored, versioned explanation; generated reports belong under `outputs/reports/` (which already exists and is gitignored) so `docs/` stays a trustworthy, curated reference.

**H-5. Documentation has drifted from the real tree.**
The README's "Project Structure" section (the project's own canonical structure diagram) does not mention `scripts/`, `config/`, `runs/`, `outputs/`, `.agents/`, `.qodo/`, `dataconnect/`, `automation/requirements/`, or the root `package.json`. A structure diagram that is already wrong is worse than none, since it actively misleads new contributors and auditors.

### Medium

**M-1. Root directory has no layering.** 38 top-level entries (`api`, `automation`, `cli.py`, `config`, `configs`, `core`, `data`, `dataconnect`, `deployment`, `docs`, `enrollment`, `evaluation`, `events`, `frontend`, `intelligence`, `main.py`, `models`, `monitoring`, `outputs`, `pipeline`, `preprocessing`, `runs`, `scripts`, `security_layer`, `services`, `storage`, `streaming`, `tests`, `tools`, `training`, `utils`, …) sit as flat siblings with no visual/physical grouping between "runtime application," "ML/data platform," "ops & tooling," and "governance/docs." At this scale (a system with camera ingestion, ReID, continual learning, a security layer, and a full React frontend) a professional layout groups these, even if only via naming convention or a top-level README map, so the boundary between "what ships to production" and "what's for research/ops" is visible at a glance.

**M-2. `data/` mixes research datasets with live operational state.** `data/casia_b_raw.zip`, `data/casia_processed/`, `data/casia_cache/` (research dataset artifacts) sit next to `data/operator_store.json`, `data/recon_intelligence.db`, `data/firebase_offline_store.json`, `data/learning_jobs.json`, `data/cases/` (live production/operational state, some of it holding case data tied to real biometric investigations). These have entirely different lifecycle, backup, retention, and access-control needs and read oddly as siblings in a system that markets itself on data governance.

**M-3. `outputs/` is an undifferentiated dumping ground.** It holds ephemeral benchmark JSON, PNG plots, ad-hoc debug images (`debug_crop.jpg`), `logs/`, `media/`, `monitoring/`, `temporary/`, `test_iso_*` directories, and `watchlist/` all as flat siblings. Ephemeral CI/test output, persistent monitoring exports, and what looks like real watchlist data shouldn't share one undifferentiated bucket.

**M-4. `api/` versioning is inconsistent.** `api/routes/` (unversioned: `enrollment.py`, `health.py`, `inference.py`, `status.py`) sits alongside `api/v1/` (`auth_router.py`, `router.py`). Either everything is v1 and should live under `api/v1/`, or there's an intentional versioning scheme that isn't documented anywhere.

**M-5. Firebase docs and security docs use inconsistent grouping.** `docs/security/*.md` (14 files) is a proper subfolder, but `docs/firebase_architecture.md`, `docs/firebase_data_model.md`, `docs/firebase_integration_audit.md`, `docs/firebase_security.md` sit flat in `docs/` root rather than in a `docs/firebase/` subfolder of their own, despite being an equally coherent topic cluster.

**M-6. Ambiguous AI-tool directories at repo root.** `.agents/`, `.qodo/` are root-level directories from third-party AI coding tools. They're harmless, but at root they read as unexplained clutter to an outside reviewer; tool-specific config conventionally lives hidden/scoped or is called out in a CONTRIBUTING doc so it doesn't look like an unknown/undocumented integration.

### Low

**L-1.** `assets/github/` is a slightly unclear name for "README banner images" — `assets/branding/` or `docs/assets/` would read more clearly.

**L-2.** `security_layer/` is the only module using a `_layer` suffix; every sibling module is a bare noun (`services`, `pipeline`, `intelligence`). Harmless, but breaks the naming convention.

**L-3.** Root carries both a Python entry point set (`cli.py`, `main.py`) and a Node `package.json` whose scripts largely delegate into `frontend/` and `tools/dev.js`. This is a reasonable monorepo pattern, but nothing at root explains *why* both ecosystems live there — worth a one-paragraph note in README or a `CONTRIBUTING.md`.

**L-4.** `evaluation/results/` (checked-in PNG/JSON result artifacts) vs `outputs/` vs `runs/` — a third location for what is conceptually the same kind of thing (evaluation/experiment output).

---

## 4. Proposed Target Structure

This is a **target to migrate toward**, not a mandate to do it all at once (see §5 phasing). It keeps every existing module name where possible and only changes *placement/grouping*, to keep the diff reviewable.

```
ARGUS_AI/
├── app/                          # Everything that ships in a production deployment
│   ├── api/                      #   (unchanged) — but flatten to api/v1/... consistently
│   ├── core/                     #   (unchanged)
│   ├── services/                 #   (unchanged)
│   ├── pipeline/                 #   (unchanged)
│   ├── intelligence/             #   sub-packaged (see below)
│   ├── streaming/                #   (unchanged)
│   ├── enrollment/                #   (unchanged)
│   ├── events/                   #   (unchanged)
│   ├── storage/                  #   (unchanged)
│   ├── security_layer/           #   (unchanged, or renamed `security/`)
│   └── monitoring/                #   (unchanged)
│
├── ml_platform/                  # Research / model-lifecycle side, not part of the running app
│   ├── models/                   #   model code + registry only (architectures/, inference/, model_registry.py)
│   ├── model_store/               #   artifact lifecycle: weights/, active/, candidates/, rollback/, engines/, export/
│   ├── galleries/                 #   gallery/, appearance_gallery/, live_gallery/ — clearly named by role
│   ├── training/                  #   (unchanged)
│   ├── preprocessing/              #   (unchanged)
│   └── evaluation/
│       ├── experiments/            #   the ad-hoc evaluate_*/run_exp*/analyze_* scripts
│       ├── benchmarks/             #   (unchanged)
│       └── results/                #   (unchanged, or move fully into outputs/evaluation/)
│
├── ops/                           # Deployment, hardware, automation, ad-hoc tooling
│   ├── deployment/                 #   (unchanged)
│   ├── automation/                  #   (unchanged)
│   ├── tools/                       #   single source of truth — scripts/ retired into this
│   └── configs/                     #   configs/ (unchanged) + configs/secrets/ (gitignored, replaces config/)
│
├── data/                           # Split by lifecycle
│   ├── datasets/                    #   casia_b_raw.zip, casia_processed/, casia_cache/, dataset_manifests/
│   └── runtime/                     #   operator_store.json, recon_intelligence.db, cases/, upload_sessions/, ...
│
├── outputs/                        # Ephemeral, fully gitignored, regenerable
│   ├── benchmarks/, evaluation/, logs/, media/, monitoring/, reports/, temporary/
│
├── frontend/                        # (unchanged)
├── tests/                            # (unchanged — already a good pattern)
├── docs/
│   ├── architecture/                 # system-level docs
│   ├── firebase/                      # the 4 firebase_*.md files grouped
│   ├── security/                       # (unchanged)
│   └── README_INDEX.md
│
├── cli.py, main.py, Makefile, VERSION, README.md, LICENSE, requirements.txt, package.json
```

**`intelligence/` internal sub-packaging** (addresses H-2), keeping it inside `app/`:

```
intelligence/
├── fusion/         # dual_modal_fusion, fusion_weights, learned_fusion, fusion_diagnostics, score_calibrator, score_normalizer
├── tracking/       # cross_camera_tracker, concurrent_track_manager, track_recovery_manager, track_reliability_scorer, identity_persistence, track_identity_aggregator
├── crowd/          # crowd_density_estimator, crowd_intelligence_system, crowd_occlusion_analyzer, crowd_robustness_manager
├── learning/       # continual_learning_*, date_aware_learning_scheduler, nn_fine_tuner, background_learning_worker, training_dataset_builder, drift_detector
├── decision/       # decision_engine, policy_engine, human_review_decision, recognition_deferral_engine, confidence_scorer, confusion_detector
├── evidence/       # operational_evidence_manager, operational_embedding_collector, event_timeline_reconstructor, multi_camera_evidence_fusion, explainable_recognition_report
└── validation/     # accuracy_validation_gate, statistical_accuracy_validator, longitudinal_accuracy_evaluator, candidate_validator
```

---

## 5. Recommended Migration Approach (not executed)

Structure changes on a codebase this size carry real regression risk (import paths, CI, packaged scripts, `Makefile` targets, Windows `.ps1`/`.bat` launchers). If/when you want to act on this report, the lowest-risk order is:

1. **Zero-risk hygiene first (Critical findings):** delete tracked `.bak` files; pick one of `scripts/`/`tools/` as canonical, diff the divergent pairs (esp. `manage_venv.ps1`), delete the loser; collapse `config/` into `configs/secrets/` and update the one reference to `firebase-service-account.json`'s path.
2. **Move generated content out of `docs/`:** relocate `docs/reports/*.json` to `outputs/reports/`, update whatever writer script currently targets `docs/reports/`.
3. **Re-sync the README structure diagram** with reality (cheap, high trust payoff, do this regardless of anything else).
4. **Sub-package `intelligence/`** — mechanical, import-path-only change; do it as its own PR with a full test run before/after.
5. **Clarify `api/` versioning** — either move `api/routes/*` under `api/v1/`, or document why they're separate.
6. **Larger regroupings last** (the `app/`, `ml_platform/`, `ops/` top-level split) — highest churn, so only worth doing if you actually want the repo to visually communicate "production system" vs "research platform" to outside reviewers (e.g., for an audit, investment due-diligence, or open-sourcing). If the repo will stay a single-maintainer research prototype, this step is optional; steps 1–5 already fix the findings that would embarrass the project in a professional review.

---

## 6. What's Already Right (keep as-is)

- `tests/` layout (`unit/integration/system/performance`, mirrored subfolders by domain) is a genuinely good pattern — worth using as the template when sub-packaging `intelligence/` and `evaluation/`.
- Domain separation at the module level (`security_layer`, `streaming`, `services`, `pipeline`, `enrollment`) is sound; the problem is almost entirely *placement and internal flatness*, not the domain boundaries themselves.
- `.gitignore` already correctly excludes `data/`, `runs/`, `outputs/`, `*.enc`, `*credentials*` — the C-2/C-3 findings are about files that slipped past these rules or duplicate paths the rules don't cover, not a missing `.gitignore` strategy.
