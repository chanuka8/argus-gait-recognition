# Storage

The `storage` package manages data persistence, evidence retention policy enforcement, data operation lineage tracking, vector embedding indexing, and dataset cache loading for ARGUS AI.

## Responsibilities

- Saving evidence snapshots, GEIs, and metadata into organized directories.
- Enforcing retention policy max-age cleanup on evidence files.
- Recording data processing operations into structured lineage logs (`outputs/reports/explainable/lineage.json`).
- Managing in-memory and disk vector indexes for fast embedding lookup.
- Boundaries: Does not handle real-time RTSP stream acquisition or GUI video rendering.

## Key Modules

<!-- BEGIN SYNC: KEY_MODULES -->
| Module | Purpose |
| --- | --- |
| [cache_manager.py](cache_manager.py) | General-purpose thread-safe key-value cache manager |
| [data_manager.py](data_manager.py) | Central storage manager coordinating datasets, evidence, and vector stores |
| [dataset_loader.py](dataset_loader.py) | Loads GEI image datasets and feature matrix caches from disk |
| [embedding_database.py](embedding_database.py) | Module/resource file embedding_database.py |
| [evidence_manager.py](evidence_manager.py) | Saves evidence snapshots, GEIs, and JSON metadata with automated retention purging |
| [firebase_embedding_store.py](firebase_embedding_store.py) | Module/resource file firebase_embedding_store.py |
| [lineage_tracker.py](lineage_tracker.py) | Tracks data processing operations and writes audit lineage records |
| [vector_store.py](vector_store.py) | Vector store indexing 256-dim feature embeddings for fast cosine nearest-neighbor retrieval |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

Detection Output → `storage/evidence_manager.py` → `outputs/media/detections/` (snapshots, GEIs, metadata) & `storage/lineage_tracker.py` → `outputs/reports/explainable/lineage.json`.

## Configuration

- [configs/system.yaml](../configs/system.yaml): `recognition.gallery_dir`

## Firebase Admin SDK Setup (Optional Live Mode)

When unconfigured, ARGUS runs in **offline mode** with zero inference degradation and queues transactions in `data/firebase_offline_store.json`. To connect to live Firebase services:

1. **Download Service Account Key**:
   Download the Firebase Admin SDK private key JSON from the Firebase Console for project `argus-17702`.
2. **Place Key File**:
   Save the credential file as:
   ```text
   E:\ARGUS_AI\config\firebase-service-account.json
   ```
   *(This path is strictly ignored by Git and must never be committed).*
3. **Configure Environment Variable**:
   ```powershell
   $env:FIREBASE_SERVICE_ACCOUNT_PATH="E:\ARGUS_AI\config\firebase-service-account.json"
   ```
   *(Or set `GOOGLE_APPLICATION_CREDENTIALS`)*
4. **Verify File Existence**:
   ```powershell
   Test-Path $env:FIREBASE_SERVICE_ACCOUNT_PATH
   # Expected: True
   ```
5. **Safely Validate Structure**:
   ```powershell
   $j = Get-Content $env:FIREBASE_SERVICE_ACCOUNT_PATH -Raw | ConvertFrom-Json
   [PSCustomObject]@{
       Type           = $j.type
       ProjectId      = $j.project_id
       HasPrivateKey  = [bool]$j.private_key
       HasClientEmail = [bool]$j.client_email
   }
   # Expected:
   # Type           = service_account
   # ProjectId      = argus-17702
   # HasPrivateKey  = True
   # HasClientEmail = True
   ```
   > [!CAUTION]
   > Never print, log, or commit the `private_key` value.

## Public Interfaces

- `EvidenceManager`: Evidence persistence manager in [storage/evidence_manager.py](evidence_manager.py).
- `LineageTracker`: Operations tracker in [storage/lineage_tracker.py](lineage_tracker.py).
- `VectorStore`: Embedding vector index in [storage/vector_store.py](vector_store.py).

## Tests

- [tests/unit/test_output_layout.py](../tests/unit/test_output_layout.py)
- [tests/integration/test_dual_modal_pipeline.py](../tests/integration/test_dual_modal_pipeline.py)

## Related Documentation

- [Root README](../README.md)
- [Security Layer Documentation](../security_layer/README.md)
