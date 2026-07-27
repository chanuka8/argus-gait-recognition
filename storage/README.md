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
|---|---|
| [cache_manager.py](cache_manager.py) | General-purpose thread-safe key-value cache manager |
| [data_manager.py](data_manager.py) | Central storage manager coordinating datasets, evidence, and vector stores |
| [dataset_loader.py](dataset_loader.py) | Loads GEI image datasets and feature matrix caches from disk |
| [evidence_manager.py](evidence_manager.py) | Saves evidence snapshots, GEIs, and JSON metadata with automated retention purging |
| [lineage_tracker.py](lineage_tracker.py) | Tracks data processing operations and writes audit lineage records |
| [vector_store.py](vector_store.py) | Vector store indexing 256-dim feature embeddings for fast cosine nearest-neighbor retrieval |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

Detection Output → `storage/evidence_manager.py` → `outputs/media/detections/` (snapshots, GEIs, metadata) & `storage/lineage_tracker.py` → `outputs/reports/explainable/lineage.json`.

## Configuration

- [configs/system.yaml](../configs/system.yaml): `recognition.gallery_dir`

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
