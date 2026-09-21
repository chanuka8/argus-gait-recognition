# Intelligence

The `intelligence` package contains advanced biometric decision logic, open-set recognition engines, multi-camera evidence fusion algorithms, track reliability scoring, crowd intelligence, camera topology learning, and real-time watchlist workflows for ARGUS AI.

## Responsibilities

- Evaluating open-set match confidence and managing unknown subject rejection.
- Fusing dual-modal gait and appearance feature embeddings dynamically.
- Scoring tracklet reliability, crowd density, and occlusion severity.
- Learning cross-camera transition travel times and handling missing person watchlist alerts.
- Boundaries: Does not handle low-level video stream decoding or raw bounding box drawing.

## Key Modules

<!-- BEGIN SYNC: KEY_MODULES -->
| Module | Purpose |
| --- | --- |
| [alert_manager.py](alert_manager.py) | Threat alert throttling and priority alert dispatching |
| [appearance_embedding.py](appearance_embedding.py) | Feature extractor and embedding generator for person appearance ReID |
| [continuous_improvement_engine.py](continuous_improvement_engine.py) | Module/resource file continuous_improvement_engine.py |
| `crowd/` | Module/resource file crowd/ |
| `decision/` | Module/resource file decision/ |
| `evidence/` | Module/resource file evidence/ |
| `fusion/` | Module/resource file fusion/ |
| `learning/` | Module/resource file learning/ |
| `tracking/` | Module/resource file tracking/ |
| `validation/` | Module/resource file validation/ |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

Pipeline Feature Embeddings → `intelligence/open_set_recognizer.py` & `intelligence/dual_modal_fusion.py` → `intelligence/decision_engine.py` → `intelligence/missing_person_workflow.py` → Alert / Security Output.

## Configuration

- [configs/inference.yaml](../configs/inference.yaml): thresholds, ReID, watchlist, crowd intelligence, topology parameters

## Public Interfaces

- `OpenSetRecognizer`: Open-set matching engine in [intelligence/open_set_recognizer.py](open_set_recognizer.py).
- `DualModalFusion`: Adaptive gait and appearance fusion in [intelligence/dual_modal_fusion.py](dual_modal_fusion.py).
- `CrowdIntelligenceSystem`: Crowd orchestrator in [intelligence/crowd_intelligence_system.py](crowd_intelligence_system.py).
- `MissingPersonWorkflow` (`WatchlistManager`): Watchlist engine in [intelligence/missing_person_workflow.py](missing_person_workflow.py).

## Tests

- [tests/unit/test_dual_modal_fusion.py](../tests/unit/test_dual_modal_fusion.py)
- [tests/test_phase6_intelligence.py](../tests/test_phase6_intelligence.py)
- [tests/test_watchlist_integration.py](../tests/test_watchlist_integration.py)

## Related Documentation

- [Root README](../README.md)
- [Pipeline Documentation](../pipeline/README.md)
