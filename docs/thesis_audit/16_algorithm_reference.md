# Phase 15 — Algorithm and Formula Reference

## 16.1 Gait Energy Image (GEI) — Han & Bhanu, 2006

### Definition

The Gait Energy Image is computed as the normalized average of binary silhouette images over a complete gait cycle:

```
GEI(x, y) = (1/N) × Σᵢ₌₁ᴺ Bᵢ(x, y)
```

Where:
- `Bᵢ(x, y)` is the binary silhouette at frame `i` (0 or 255)
- `N` is the number of frames in the cycle
- `(x, y)` are pixel coordinates

### Implementation

**File:** `pipeline/steps/live_gei.py::LiveGEI`

```python
def build(self):
    stack = np.stack(self.frames, axis=0).astype(np.float32)
    gei = np.mean(stack, axis=0).astype(np.uint8)
    return gei
```

### Parameters

| Parameter | Value | Notes |
|---|---|---|
| `max_frames` | 15 | Rolling window size |
| `min_frames` | 10 | Minimum frames before GEI is valid |
| `image_size` | (64, 128) | Width × Height |

## 16.2 Cosine Similarity

### Definition

```
cos(q, g) = (q · g) / (‖q‖₂ × ‖g‖₂)
```

Since ARGUS L2-normalizes all embeddings:

```
cos(q, g) = q · g    (when ‖q‖₂ = ‖g‖₂ = 1)
```

### Implementation

**File:** `pipeline/steps/matching_step.py::MatchingStep.top_k_matches()`

```python
norms = np.linalg.norm(gallery_features, axis=1, keepdims=True)
gallery_features = gallery_features / (norms + 1e-8)
query_feature = query_feature / (np.linalg.norm(query_feature) + 1e-8)
scores = np.dot(gallery_features, query_feature)
```

## 16.3 ArcFace Loss — Deng et al., 2019

### Definition

```
L_ArcFace = -log( exp(s × cos(θᵧᵢ + m)) / (exp(s × cos(θᵧᵢ + m)) + Σⱼ≠ᵧᵢ exp(s × cos(θⱼ))) )
```

Where:
- `s = 64` (scaling factor)
- `m = 0.35` (angular margin, radians)
- `θⱼ = arccos(W̃ⱼᵀ × x̃)` (angle between normalized weight and feature)

### Implementation

**File:** `models/architectures/losses.py::ArcMarginProduct`

```python
cosine = F.linear(F.normalize(input), F.normalize(self.weight))
sine = torch.sqrt(1.0 - cosine.pow(2))
phi = cosine * self.cos_m - sine * self.sin_m  # cos(θ + m)
output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
output = output * self.s
```

## 16.4 Batch-Hard Triplet Loss — Hermans et al., 2017

### Definition

```
L_Triplet = (1/P) × Σₐ max(0, max_{p: yₚ=yₐ} d(aₐ, aₚ) - min_{n: yₙ≠yₐ} d(aₐ, aₙ) + margin)
```

Where:
- `d(a, b) = ‖a - b‖₂` (Euclidean distance)
- `margin = 0.3`
- Hardest positive: max distance with same label
- Hardest negative: min distance with different label

### Implementation

**File:** `models/architectures/losses.py::BatchHardTripletLoss`

```python
dist_matrix = torch.cdist(embeddings, embeddings, p=2)
# Hardest positive: max distance where labels match
# Hardest negative: min distance where labels differ
loss = F.relu(d_ap - d_an + self.margin).mean()
```

### Note

In the current training configuration, `triplet_weight = 0.0`, so triplet loss was **disabled**. The active loss was ArcFace CrossEntropy only.

## 16.5 Otsu's Thresholding — Otsu, 1979

### Definition

Otsu's method finds the threshold `t*` that minimizes the weighted intra-class variance:

```
t* = argmin_t { ω₀(t) × σ₀²(t) + ω₁(t) × σ₁²(t) }
```

Equivalently, it maximizes the inter-class variance:

```
t* = argmax_t { ω₀(t) × ω₁(t) × (μ₀(t) - μ₁(t))² }
```

Where:
- `ω₀, ω₁` are the class probabilities (foreground/background)
- `σ₀², σ₁²` are the class variances
- `μ₀, μ₁` are the class means

### Implementation

**File:** `pipeline/steps/silhouette_step.py`

```python
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

## 16.6 Exponential Moving Average (Box Stabilization)

### Definition

```
box_stable(t) = α × box_raw(t) + (1 - α) × box_stable(t-1)
```

Where:
- `α = 0.35` (EMA smoothing factor)

### Implementation

**File:** `utils/box_stabilizer.py`

## 16.7 Prediction Smoothing (Majority Voting)

### Algorithm

```
votes = {}
for (prediction, score) in history[-10:]:
    if prediction ≠ "UNKNOWN" and score ≥ threshold:
        votes[prediction] += 1

if max(votes.values()) ≥ min_stable_votes (3):
    return argmax(votes)
elif current_confirmed has ≥ 1 vote:
    return current_confirmed
else:
    return "UNKNOWN"
```

### Implementation

**File:** `utils/prediction_smoother.py::PredictionSmoother`

## 16.8 Centroid Matching with Margin

### Algorithm

```
1. For each identity i, compute centroid:
   c_i = L2_normalize(mean(gallery[labels == i]))

2. Compute centroid scores:
   s_i = query · c_i  (cosine similarity)

3. Sort by score descending

4. Margin rule:
   if s_best - s_second_best < margin (0.05):
       return UNKNOWN  # ambiguous

5. Threshold rule:
   if s_best < threshold (0.85):
       return UNKNOWN

6. Top-k consensus:
   flat_scores = query · all_gallery_features
   top_k_labels = labels[argsort(flat_scores)[-5:]]
   if count(best_identity in top_k) > k/2:
       return best_identity
   else:
       return UNKNOWN
```

### Implementation

**File:** `pipeline/steps/centroid_matching_step.py::CentroidMatchingStep`

## 16.9 Biometric Rate Definitions

| Metric | Formula | Meaning |
|---|---|---|
| FAR | FP / (FP + TN) | Rate of impostors falsely accepted |
| FRR | FN / (FN + TP) | Rate of genuine users falsely rejected |
| TAR | TP / (TP + FN) = 1 - FRR | Rate of genuine users correctly accepted |
| TNR | TN / (TN + FP) = 1 - FAR | Rate of impostors correctly rejected |
| EER | FAR = FRR | Operating point where both error rates equal |
| ROC-AUC | ∫ TAR d(FAR) | Area under ROC curve (0.5 = random, 1.0 = perfect) |
| Rank-k | P(true_id in top-k) | Probability of correct ID appearing in top-k |

### Implementation

**File:** `evaluation/metrics.py`

## 16.10 Rank-k Accuracy

### Definition

```
Rank_k = (1/Q) × Σ_{i=1}^{Q} 𝟙[true_id_i ∈ top_k_predictions_i]
```

Where:
- `Q` is the number of probe queries
- `top_k_predictions_i` are the k nearest gallery identities for probe i

### Implementation

**File:** `evaluation/metrics.py::compute_rank_k_accuracies()`
