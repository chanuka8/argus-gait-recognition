# Technical Methodology Specification

This document details the mathematical models, network architecture, and processing algorithms used in the ARGUS Gait Recognition Module.

---

## 1. Dataset Preprocessing

To train and evaluate the system, we used the CASIA-B dataset, which provides multi-view walking sequences for 124 subjects under different conditions (e.g., normal walking, walking with a coat, walking with a bag).

The preprocessing workflow converts raw video frames or silhouettes into normalized arrays:
1.  **Ingestion:** Scans subject folders to parse sequence tags (e.g., `nm` for normal walking, `bg` for carrying a bag, `cl` for wearing a coat).
2.  **Silhouette Cropping:** Locates silhouette regions in frame masks and extracts bounding boxes around the subjects.
3.  **Resizing:** Silhouettes are resized to a fixed resolution of 128 pixels (height) by 64 pixels (width).
4.  **Centering:** Aligns silhouettes horizontally by centering the horizontal center-of-mass, ensuring structural consistency.

---

## 2. Gait Energy Image (GEI) Representation

Gait recognition must model both spatial postures and temporal walking frequencies. Instead of using complex 3D modeling (which is computationally expensive for edge processors), we use **Gait Energy Images (GEIs)**.

A GEI is calculated by averaging aligned silhouette frames over a complete gait cycle:

$$G(x,y) = \frac{1}{N} \sum_{t=1}^{N} B_t(x,y)$$

Where:
*   $B_t(x,y)$ represents the binary silhouette image value at coordinates $(x,y)$ for frame $t$.
*   $N$ is the total number of frames in one gait cycle.

The resulting GEI is a grayscale image where pixel intensity represents the duration of body part occupancy at that coordinate during a gait cycle. This highlights structural body shapes (high intensity at the torso and head) and motion paths (low intensity/blurs at the limbs).

---

## 3. ByGaitLight Neural Architecture

To run on standard processors without requiring high-end GPUs, we designed a lightweight convolutional neural network called **ByGaitLight**.

### Network Structure:
*   **Input Layer:** Processes single-channel grayscale GEIs ($1 \times 128 \times 64$).
*   **Layer 1 (Conv + ReLU + Pool):**
    *   Convolution: 8 filters of size $3 \times 3$, stride 1, padding 1.
    *   Max Pooling: $2 \times 2$ pool size, downscaling the image to $64 \times 32$.
*   **Layer 2 (Conv + ReLU + Pool):**
    *   Convolution: 16 filters of size $3 \times 3$, stride 1, padding 1.
    *   Max Pooling: $2 \times 2$ pool size, downscaling the image to $32 \times 16$.
*   **Layer 3 (Conv + ReLU + Pool):**
    *   Convolution: 32 filters of size $3 \times 3$, stride 1, padding 1.
    *   Max Pooling: $2 \times 2$ pool size, downscaling the image to $16 \times 8$.
*   **Layer 4 (Conv + ReLU + Pool):**
    *   Convolution: 64 filters of size $3 \times 3$, stride 1, padding 1.
    *   Max Pooling: $2 \times 2$ pool size, downscaling the image to $8 \times 4$.
*   **Layer 5 (Fully Connected Embedder):**
    *   Flattens features ($64 \times 8 \times 4 = 2048$ dimensions) and maps them to a **128-dimensional embedding vector**.
*   **Layer 6 (Classification Layer - Training Only):**
    *   Maps the 128-dimensional vector to the number of classes (subjects) using a Linear projection. This layer is discarded during inference.

---

## 4. Model Training Protocol

*   **Loss Function:** The network is trained using Multi-Class Cross Entropy Loss to maximize classification accuracy.
*   **Optimizer:** Uses the Adam optimizer with a learning rate of $0.001$ and a weight decay of $1\text{e-}4$ to prevent overfitting.
*   **Regularization:** Integrates callback controls including:
    *   **Early Stopping:** Stops training if the validation loss does not improve for 5 consecutive epochs.
    *   **Epoch Checkpointing:** Automatically saves model weights if validation accuracy increases.

---

## 5. Gallery Vector Matching

Database search queries use normalized vector dot products:
1.  **Feature Extraction:** The system maps the candidate GEI to a 128-dimensional embedding vector $v_q$ using `ByGaitLight`.
2.  **Normalization:** Normalizes the embedding vector using L2 normalization:
    $$\hat{v}_q = \frac{v_q}{\|v_q\|_2}$$
3.  **Matrix Search:** Computes the cosine similarity against all gallery embeddings using a single matrix dot product:
    $$S = \hat{V}_g \cdot \hat{v}_q$$
    Where $\hat{V}_g$ is the pre-normalized matrix of gallery embeddings.
4.  **Identification:** The index with the highest cosine score determines the identity match.

---

## 6. Live Ingestion & Tracking Pipeline

Running recognition in real-time requires continuous target tracking:
*   **Person Detection:** YOLOv8 extracts target bounding boxes from camera frames.
*   **Tracking:** ByteTrack associates detections across frames to maintain stable subject identities (Track IDs).
*   **Rolling GEI Buffers:** Instead of waiting for a manual capture, each active Track ID maintains a rolling queue of silhouettes. Once the queue reaches 15 frames, the system builds a GEI and queries the database, refreshing the identification label overlay.

---

## 7. Performance Limitations

1.  **Clothing & Carrying Variations:** Wearing thick coats (`cl` sequences) or carrying bags (`bg` sequences) changes silhouette patterns. Since `ByGaitLight` extracts features from silhouette shapes, these variations can reduce matching accuracy.
2.  **Viewing Angle Dependency:** If the model is trained on side-angle views (e.g., 90 degrees), accuracy will drop on front-facing profiles (e.g., 0 degrees) because the silhouettes look different.
3.  **Low Inundation Thresholds:** The basic MOG2 subtraction algorithm struggles to segment targets if the subject's clothing color is highly similar to the background.
