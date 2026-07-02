# Open-Set Evaluation

This document details the open-set evaluation framework, the centroid-margin top-K matching policy, and performance metrics for handling unknown/unregistered individuals.

---

## 1. Open-Set vs. Closed-Set Recognition

In a **closed-set** recognition scenario, the system assumes that every tracked individual is enrolled in the gallery database. In a real-world surveillance deployment (e.g., airport terminals or building access points), this assumption is invalid.

An **open-set** biometric system must:
1. Match known individuals to their correct profiles.
2. Correctly reject unregistered individuals, classifying them as `UNKNOWN`.

To evaluate this, the dataset is split:
- **Known Class Partition:** Enrolled in the gallery database.
- **Unknown Class Partition:** Kept completely separate from the gallery to simulate unregistered intruders.

---

## 2. Centroid-Margin Top-K Matching Policy

To prevent false matches when query patterns are ambiguous or close to decision boundaries, ARGUS implements a hybrid **Centroid-Margin Top-K** policy in [pipeline/steps/centroid_matching_step.py](file:///e:/ARGUS_AI/pipeline/steps/centroid_matching_step.py):

1. **Top-K Retrieval:** Retrieve the top $K$ (default: 5) closest individual matching vectors from the database.
2. **Centroid Distance:** Compute the centroid embedding for each unique identity present in the top $K$.
3. **Margin Constraint:** If the similarity score of the best match is not greater than the second-best match's similarity by a minimum margin (default: $0.05$), the match is flagged as ambiguous and routed for review.
4. **Rejection Threshold:** If the highest similarity falls below the rejection ceiling (e.g. $0.70$), the subject is classified as `UNKNOWN`.

---

## 3. Open-Set Metrics

The evaluator calculates:
- **True Positive (TP):** Known subject correctly identified.
- **False Positive (FP):** Unknown subject incorrectly matched to an identity.
- **True Negative (TN):** Unknown subject correctly rejected as `UNKNOWN`.
- **False Negative (FN):** Known subject incorrectly rejected as `UNKNOWN` or matched to a wrong profile.
- **False Accept Rate (FAR):**
  $$\text{FAR} = \frac{\text{FP}}{\text{Total Unknowns}}$$
- **False Reject Rate (FRR):**
  $$\text{FRR} = \frac{\text{FN}}{\text{Total Knowns}}$$

---

## 4. Open-Set Evaluation Commands

To run the open-set validation script:
```bash
python scripts/evaluate_open_set.py
```
This script evaluates the system's ability to reject unknown subjects, writing outputs to `outputs/eval_reports/open_set_report.json`.

To sweep open-set thresholds:
```bash
python scripts/evaluate_open_set_threshold_sweep.py
```
This determines the optimal threshold that minimizes the False Accept Rate while maintaining high True Positive rates, writing results to `outputs/eval_reports/open_set_threshold_sweep.csv`.
