# ARGUS Final One-Mode Runtime Policy

## Overview

The ARGUS system runs with a single command:

```bash
python cli.py --mode system
```

No manual demo/security mode selection is needed. The runtime automatically
chooses the safest practical decision path using an adaptive hybrid matching
policy.

## Decision Flow

```
                    ┌──────────────────────┐
                    │  Extract GEI + Embed │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Flat MatchingStep   │
                    │  (dot-product match) │
                    └──────────┬───────────┘
                               │
                  flat_identity, flat_score
                               │
              ┌────────────────┼────────────────┐
              │                │                │
     flat == UNKNOWN    flat_score >= 0.92    0.85 <= flat_score < 0.92
              │                │                │
     UNKNOWN_PERSON    CONFIRMED_MATCH   ┌──────▼────────┐
     (green box)       (red box)         │ Centroid +    │
                                         │ Margin +     │
                                         │ Top-K Verify │
                                         └──────┬───────┘
                                                │
                                    centroid agrees?
                                      /          \
                                    YES           NO
                                     │             │
                              VERIFIED_MATCH  REVIEW_REQUIRED
                              (red box)       (orange box)
              │
     0.70 <= flat_score < 0.85
              │
       LOW_CONFIDENCE
       (orange box)
              │
     flat_score < 0.70
              │
       UNKNOWN_PERSON
       (green box)
```

## Decision Levels

| Decision         | Flat Score Range       | Verification      | Box Color | Severity |
|------------------|------------------------|--------------------|-----------|----------|
| CONFIRMED_MATCH  | >= 0.92                | None needed        | Red       | INFO     |
| VERIFIED_MATCH   | [0.85, 0.92)           | Centroid agrees    | Red       | INFO     |
| REVIEW_REQUIRED  | [0.85, 0.92)           | Centroid disagrees | Orange    | MEDIUM   |
| LOW_CONFIDENCE   | [0.70, 0.85)           | Not run            | Orange    | MEDIUM   |
| UNKNOWN_PERSON   | < 0.70 or flat=UNKNOWN | N/A                | Green     | HIGH     |
| COLLECTING       | N/A (not enough frames)| N/A                | Yellow    | INFO     |

## Box Colors (BGR)

| Decision          | BGR Color     | Visual  |
|-------------------|---------------|---------|
| CONFIRMED_MATCH   | (0, 0, 255)   | Red     |
| VERIFIED_MATCH    | (0, 0, 255)   | Red     |
| REVIEW_REQUIRED   | (0, 165, 255) | Orange  |
| LOW_CONFIDENCE    | (0, 165, 255) | Orange  |
| UNKNOWN_PERSON    | (0, 255, 0)   | Green   |
| COLLECTING        | (0, 255, 255) | Yellow  |

## Verification Logic (Mid-Range Zone)

When `0.85 <= flat_score < 0.92`, the system runs full centroid + margin + top-k
consensus verification:

1. **Centroid matching**: Build one centroid per identity from gallery embeddings.
   L2-normalize centroids. Compute cosine similarity.
2. **Margin rule**: If `best_score - second_best_score < margin` (default 0.05),
   reject as UNKNOWN.
3. **Top-K consensus**: Use top_k=5 flat gallery neighbors. Confirm identity only
   if majority of top-k labels match the centroid best identity.

If the centroid+margin+topk identity matches the flat identity → **VERIFIED_MATCH**.
If they disagree → **REVIEW_REQUIRED**.

## Configuration

All thresholds are configurable in `configs/inference.yaml`:

```yaml
matching_policy:
  confirmed_threshold: 0.92
  verify_low: 0.85
  verify_high: 0.92
  low_confidence_low: 0.70
  low_confidence_high: 0.85
  unknown_ceiling: 0.70
  centroid_threshold: 0.85
  margin: 0.05
  top_k: 5
  min_stable_votes: 3
  history_size: 10
```

## Prediction Smoother Integration

The PredictionSmoother provides temporal stability:

- `history_size=10`: Tracks the last 10 predictions per track.
- `min_stable_votes=3`: Requires at least 3 consistent votes before confirming
  a new identity.
- Previously confirmed identity is retained if it has at least 1 supporting vote
  in the buffer, preventing flickering.

## Files Modified

| File | Change |
|------|--------|
| `configs/inference.yaml` | Added `matching_policy` section |
| `pipeline/live_recognition.py` | Added `_adaptive_decision()`, centroid matcher, decision-based colors |
| `pipeline/video_recognition.py` | Same adaptive decision logic as live recognition |

## Files Unchanged

| File | Reason |
|------|--------|
| `pipeline/steps/centroid_matching_step.py` | Already supports all required modes |
| `pipeline/steps/matching_step.py` | Flat matching unchanged |
| `utils/prediction_smoother.py` | Logic unchanged, parameters driven by config |
| `utils/alert_manager.py` | Already accepts decision parameter |
| `security_layer/security_engine.py` | Unchanged, severity is now set by decision policy |
| `cli.py` | `--mode system` already exists and works |
| Model architecture | No changes |
| Training code | No changes |
