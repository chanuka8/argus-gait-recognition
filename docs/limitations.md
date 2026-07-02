# System Limitations

This document outlines the known physical, algorithmic, and architectural limitations of the current ARGUS Gait Recognition system.

---

## 1. Background Modeling Dependencies

The silhouette extraction pipeline relies on **MOG2 background subtraction** combined with **Otsu binary thresholding**:
- **Dynamic Backgrounds:** Moving branches, scrolling digital signs, escalators, or heavy illumination shadows are incorrectly classified as moving foreground crops. This introduces structural noise to the silhouette masks.
- **Lighting Shifts:** Gradual or sudden ambient light level changes can skew the binary thresholding, eroding target outlines and generating incomplete GEIs.
- **Static Requirement:** The system operates under the assumption that camera sensors remain static. Moving camera setups (e.g., pan-tilt-zoom cameras or drone feeds) will fail without complex, frame-level image stabilization preprocessing.

---

## 2. Tracking Occlusions & Intersections

The ByteTrack association engine maps targets across frames based on bounding boxes intersections and motion vectors:
- **Physical Barriers:** If an individual walks behind a pillar or barrier, the tracking chain is broken. When the subject reappears, the system assigns a new, blank Track ID, resetting the rolling GEI accumulator.
- **Dense Crowds:** Multiple targets walking in close proximity or crossing paths can cause bounding box overlapping. This leads to track switching, merging silhouettes of different individuals, and corrupting the extracted GEIs.

---

## 3. Flat Vector Store Lookup Scaling

Currently, gallery features are stored in flat NumPy arrays (`.npy`), and queries perform a linear scan ($O(N)$) calculating cosine similarity across all rows:
- **Small-to-Medium Galleries:** Sub-millisecond lookup speeds for up to $15,000$ embeddings representing $150$ subjects.
- **Enterprise Scaling:** As the database scales to millions of registered profiles, linear scanning will introduce noticeable lookup delays, bottlenecking the real-time processing loop.

---

## 4. Clothing and Carrying Variations

Gait silhouettes capture outline projections, which are affected by changes in clothing or carrying items:
- **Carrying Items:** Carrying large backpacks, briefcases, or luggage changes the silhouette outline, reducing matching accuracy against "normal walking" reference samples.
- **Clothing Shifts:** Transitioning from summer clothing to heavy winter coats (`cl-01` to `cl-02` sequences in CASIA-B) obscures leg shapes and body proportions, leading to a performance decrease.
- **Footwear Variations:** Walking in high heels, boots, or heavy shoes changes the walking cadence and stance stride, impacting temporal gait signatures.
