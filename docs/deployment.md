# Production Deployment

This document provides guidelines for deploying the ARGUS Gait Recognition system in production settings, detailing API routing endpoints, server topologies, and Docker configurations.

---

## 1. Production API Reference

The system exposes a FastAPI REST API (implemented in [api/server.py](file:///e:/ARGUS_AI/api/server.py)) that allows external client integrations to query health data, enroll subjects, and perform inferences.

### GET `/health`
Returns the status of the system.
- **Response:**
  ```json
  {
    "status": "healthy",
    "timestamp": "2026-06-15T05:52:36.124800",
    "gpu_available": true
  }
  ```

### GET `/metrics`
Queries current hardware diagnostics (CPU, RAM, VRAM usage).
- **Response:**
  ```json
  {
    "cpu_usage_percent": 12.5,
    "memory_used_gb": 4.12,
    "memory_total_gb": 16.0,
    "gpu_memory_used_mb": 512
  }
  ```

### POST `/identify`
Extracts features from a pre-saved GEI image and returns the best matched profile.
- **Request Body:**
  ```json
  {
    "image_path": "data/casia_processed/gei/034/034_nm-01_126.png",
    "threshold": 0.75
  }
  ```
- **Response:**
  ```json
  {
    "identity": "034",
    "score": 0.9412,
    "decision": "CONFIRMED_MATCH"
  }
  ```

### POST `/enroll`
Registers a new identity using a folder path containing images.
- **Request Body:**
  ```json
  {
    "folder_path": "data/new_input/034",
    "person_id": "034"
  }
  ```
- **Response:**
  ```json
  {
    "status": "success",
    "person_id": "034",
    "embeddings_enrolled": 6
  }
  ```

---

## 2. Running the API Server

For production environments, run the FastAPI server using **Uvicorn** or **Gunicorn**:
```bash
python cli.py --mode api
```
*Alternative (Manual Launch):*
```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 4
```
Access the automated Swagger API documentation interface by navigating to `http://127.0.0.1:8000/docs` in your web browser.

---

## 3. Containerization Guidelines (Docker)

To deploy the service in isolated containers (e.g., Kubernetes or AWS ECS) with GPU support, use the following `Dockerfile` structure:

```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose ports
EXPOSE 8000

# Start API gateway
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```
Ensure you install the **NVIDIA Container Toolkit** on the host machine to allow Docker containers to access GPUs for deep model inference.
