# REST API Reference Manual

This document details the HTTP endpoints, data contracts, request schemas, and JSON responses for the ARGUS REST API.

---

## 1. REST Backend Server Start

The API server runs using Uvicorn on localhost (`127.0.0.1:8000`).

To start the server:
```powershell
python cli.py --mode api
```

The Swagger UI documentation is available at `http://127.0.0.1:8000/docs` while the server is active.

---

## 2. API Schema Reference

The request and response payloads use Pydantic schemas defined in `api/schemas.py`.

```
                        API Router (api/server.py)
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
   GET /health /metrics       POST /identify             POST /enroll
   (System context,           (Evaluate individual       (Add profile and
    hardware stats)            embeddings matches)        embeddings to DB)
```

---

## 3. Endpoints Specification

### 3a. System Diagnostics Check (`GET /health`)
Queries global configurations and context state parameters.

*   **Endpoint Route:** `/health`
*   **Request Method:** `GET`
*   **Response Payload (200 OK):**
    ```json
    {
      "status": "healthy",
      "mode": "inference"
    }
    ```

### 3b. Telemetry Metrics (`GET /metrics`)
Queries resource usage statistics.

*   **Endpoint Route:** `/metrics`
*   **Request Method:** `GET`
*   **Response Payload (200 OK):**
    ```json
    {
      "cpu_usage_percent": 12.4,
      "ram_usage_percent": 88.4,
      "vram_usage_percent": 0.0,
      "active_tracks_count": 0,
      "system_status": "initialized"
    }
    ```

### 3c. Target Identification (`POST /identify`)
Extracts features from a provided GEI and queries the database for matches.

*   **Endpoint Route:** `/identify`
*   **Request Method:** `POST`
*   **Request Content-Type:** `application/json`
*   **Request Body JSON:**
    ```json
    {
      "image_path": "data/casia_processed/gei/034/034_nm-01_126.png"
    }
    ```
*   **Response Payload (200 OK):**
    ```json
    {
      "identity": "034",
      "score": 1.0
    }
    ```
*   **Error Response (500 Internal Error - Gallery not compiled):**
    ```json
    {
      "detail": "Gallery not built."
    }
    ```

### 3d. Profile Enrollment (`POST /enroll`)
Scans, extracts features, and adds subject embeddings from a folder to the gallery.

*   **Endpoint Route:** `/enroll`
*   **Request Method:** `POST`
*   **Request Content-Type:** `application/json`
*   **Request Body JSON:**
    ```json
    {
      "folder_path": "data/casia_processed/gei/034"
    }
    ```
*   **Response Payload (200 OK):**
    ```json
    {
      "success": true,
      "person_id": "034",
      "message": "Enrollment completed",
      "embeddings_added": 6
    }
    ```
*   **Error Response (500 Internal Error - Folder does not exist):**
    ```json
    {
      "detail": "Folder path data/casia_processed/gei/999 does not exist."
    }
    ```
