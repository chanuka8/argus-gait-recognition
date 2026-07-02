# ARGUS Current Risk Points

**Date:** 2026-06-13  
**Scope:** Gait Recognition Module

---

## Risk Severity Legend

| Level | Meaning | Demo Impact |
|---|---|---|
| 🔴 HIGH | Could cause wrong results or crash | Could fail during demo |
| 🟡 MEDIUM | Inconsistency or logic gap | Unlikely to appear in demo but is a design flaw |
| 🟢 LOW | Cosmetic or minor inconvenience | Harmless during demo |

---

## Risk #1: InferencePipeline Uses Wrong Gallery
**Severity:** 🔴 HIGH  
**File:** `pipeline/inference_pipeline.py` (line 26)  
**Issue:** `VectorStore()` defaults to `models/gallery/` (CASIA-B training gallery) instead of `models/live_gallery/`. This means `FolderRecognitionPipeline` matches against CASIA-B subjects, not enrolled live identities.

**Impact:** If you demo `--mode recognize-folder`, the results will show CASIA-B numeric IDs (001, 002...) instead of real enrolled names.

**Affected modes:** `recognize-folder`

---

## Risk #2: InferencePipeline Does Not Filter by Status
**Severity:** 🟡 MEDIUM  
**File:** `pipeline/inference_pipeline.py` (line 57-61)  
**Issue:** `self.matcher.match()` is called WITHOUT the `metadata` argument. Even if the gallery were changed to `live_gallery`, DISABLED and ARCHIVED identities would still participate in matching.

**Impact:** Folder recognition could return a DISABLED identity as a match.

**Affected modes:** `recognize-folder`

---

## Risk #3: `_needs_enrollment()` Always Returns True
**Severity:** 🟡 MEDIUM  
**File:** `enrollment/auto_enrollment_service.py` (line 267)  
**Issue:** 
```python
if marker.get("fingerprint") != fingerprint:
    return True
return True    # ← BUG: should be return False
```
After the `_already_enrolled()` check passes (returns False → enrollment needed), this function checks the marker. Even if the marker says "enrolled" and the fingerprint matches, it returns True.

**Impact:** Mitigated by `_already_enrolled()` which checks the actual gallery. The identity will still be "skipped" because the gallery already has the person. But if gallery is manually cleared, re-enrollment occurs every cycle.

---

## Risk #4: Threshold Inconsistency Across Modes
**Severity:** 🟡 MEDIUM  
**Files:** Multiple  
**Issue:**

| Path | Effective Threshold |
|---|---|
| Live recognition (camera) | 0.85 |
| Video recognition (pipeline class) | 0.85 |
| Video recognition (script default) | 0.75 |
| Video recognition (CLI default) | 0.75 |
| Folder recognition (CLI default) | 0.70 |
| Folder recognition (MatchingStep default) | 0.75 |

**Impact:** Using `--mode recognize-video` from CLI without `--threshold` uses 0.75, which is significantly more permissive than the live pipeline's 0.85. Could produce more false positives in video mode.

---

## Risk #5: Appearance Gallery Not Used for Recognition
**Severity:** 🟢 LOW  
**Files:** All recognition pipelines  
**Issue:** The appearance gallery (`models/appearance_gallery/`) contains 147 embeddings for 5 identities. However, NO recognition pipeline queries it. It's enrollment-only infrastructure.

**Impact:** Photo-enrolled identities (Devhan, Isuru, person01) that only have appearance data (no walking videos) are invisible to all recognition modes. This is by design for the gait module but should be documented.

---

## Risk #6: No Gallery Hot-Reload During Live Recognition
**Severity:** 🟢 LOW  
**File:** `pipeline/live_recognition.py` (line 40-49)  
**Issue:** Gallery is loaded once in `__init__()`. Auto-enrollment watcher runs in a separate process and can add new identities. These new identities are not visible until the live recognition is restarted.

**Impact:** If you enroll a new person during a live demo, they won't be recognized until you restart the system mode. Workaround: restart `--mode system`.

---

## Risk #7: `run_auto_enrollment.py` Summary Print Bug
**Severity:** 🟢 LOW  
**File:** `scripts/run_auto_enrollment.py` (line 98)  
**Issue:** Prints `result.get('embeddings_added')` but the result dict now uses `gait_embeddings_added` and `appearance_embeddings_added`.

**Impact:** The one-shot enrollment summary will print `None` for embeddings count. Functional behavior is unaffected.

---

## Risk #8: PredictionSmoother Track ID Reuse
**Severity:** 🟢 LOW  
**File:** `utils/prediction_smoother.py`  
**Issue:** If ByteTrack reuses a track_id for a different person (person A leaves, person B appears), the smoother's deque still contains person A's votes. Person B needs 6+ consecutive above-threshold matches to gain majority.

**Impact:** Brief delay before person B is correctly identified. This is actually desirable — prevents momentary flickers.

---

## Risk #9: Silhouette Quality Gap (Lab vs Real-World)
**Severity:** 🟡 MEDIUM  
**Files:** `pipeline/steps/silhouette_step.py`, `pipeline/steps/feature_extraction.py`  
**Issue:** CASIA-B training data has clean, controlled silhouettes. Real-world webcam silhouettes use Otsu thresholding which can be noisy with complex backgrounds. The `_normalize_to_silhouette()` in FeatureExtractionStep adds adaptive thresholding fallback, but `SilhouetteStep` (used in live/video) uses only basic Otsu.

**Impact:** Live recognition may underperform compared to evaluation metrics. The GEI quality directly affects embedding quality. Complex backgrounds, multiple overlapping people, or partial occlusion degrade silhouettes.

**Mitigation already in place:** Morphological open/close cleanup in both paths.

---

## Risk #10: `_get_label_color()` Returns Same Color for Known and Unknown
**Severity:** 🟢 LOW  
**File:** `pipeline/live_recognition.py` (lines 225-236)  
**Issue:**
```python
if identity == "UNKNOWN":
    return (0, 255, 0)      # green
return (0, 255, 0)           # also green
```
Both UNKNOWN and recognized identities display in green. Only COLLECTING is yellow.

**Impact:** During live demo, you cannot visually distinguish between a recognized person and an unknown person by color. Video recognition correctly uses red for recognized and green for unknown.

---

## False-Positive Risk Summary

| Risk | Likelihood | Impact |
|---|---|---|
| Low threshold (0.70-0.75) in folder/video CLI | Medium | Could match wrong person |
| CASIA-B gallery used in folder recognition | High | Returns CASIA-B IDs instead of real names |
| Smoother vote carry-over on track reuse | Low | Brief wrong identity display |
| Noisy real-world silhouettes | Medium | Reduced accuracy vs evaluation |
| Appearance gallery not queried | N/A | Photo-only identities are always UNKNOWN |

---

## Risk Priority for Fix Before Demo

1. 🔴 **Risk #10** — Fix `_get_label_color()` so recognized = red, unknown = green (easy 1-line fix, high visual impact for demo)
2. 🟡 **Risk #4** — Align CLI video threshold to 0.85 (or document why it differs)
3. 🟢 **Risk #7** — Fix enrollment summary print key names
4. 🔴 **Risk #1** — Only fix if you plan to demo `--mode recognize-folder` (otherwise irrelevant)
5. 🟡 **Risk #3** — Fix the dead `return True` to `return False`
