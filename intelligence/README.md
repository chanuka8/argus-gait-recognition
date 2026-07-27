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
|---|---|
| [alert_manager.py](file:///e:/ARGUS_AI/intelligence/alert_manager.py) | Threat alert throttling and priority alert dispatching |
| [appearance_embedding.py](file:///e:/ARGUS_AI/intelligence/appearance_embedding.py) | Feature extractor and embedding generator for person appearance ReID |
| [camera_topology_learner.py](file:///e:/ARGUS_AI/intelligence/camera_topology_learner.py) | Learns directed topology graphs and travel-time bounds across cameras |
| [camera_transition_model.py](file:///e:/ARGUS_AI/intelligence/camera_transition_model.py) | Evaluates cross-camera travel times against learned probability bounds |
| [confidence_scorer.py](file:///e:/ARGUS_AI/intelligence/confidence_scorer.py) | Multi-factor confidence scoring combining similarity, quality, and track length |
| [cross_camera_tracker.py](file:///e:/ARGUS_AI/intelligence/cross_camera_tracker.py) | Tracks target identities across multi-camera topology networks |
| [crowd_density_estimator.py](file:///e:/ARGUS_AI/intelligence/crowd_density_estimator.py) | Estimates crowd density and spatial clutter surrounding detected subjects |
| [crowd_intelligence_system.py](file:///e:/ARGUS_AI/intelligence/crowd_intelligence_system.py) | Unified orchestrator for crowd-robust recognition and occlusion deferral |
| [crowd_occlusion_analyzer.py](file:///e:/ARGUS_AI/intelligence/crowd_occlusion_analyzer.py) | Analyzes bounding box overlap and inter-person occlusion ratio |
| [crowd_robustness_manager.py](file:///e:/ARGUS_AI/intelligence/crowd_robustness_manager.py) | Controls recognition threshold adaptation under heavy crowd congestion |
| [decision_engine.py](file:///e:/ARGUS_AI/intelligence/decision_engine.py) | Tiered decision policy for identity classification and confidence assignment |
| [dual_modal_fusion.py](file:///e:/ARGUS_AI/intelligence/dual_modal_fusion.py) | Adaptive fusion engine combining gait GEI and appearance ReID embeddings |
| [fusion_weights.py](file:///e:/ARGUS_AI/intelligence/fusion_weights.py) | Quality-aware dynamic weight calculator for multi-modal similarity scores |
| [human_review_decision.py](file:///e:/ARGUS_AI/intelligence/human_review_decision.py) | Manages human operator review queues for borderline verification decisions |
| [identity_persistence.py](file:///e:/ARGUS_AI/intelligence/identity_persistence.py) | Applies temporal score decay and maintains track identity continuity |
| [missing_person_workflow.py](file:///e:/ARGUS_AI/intelligence/missing_person_workflow.py) | Operational watchlist manager and missing person search workflow |
| [multi_camera_evidence_fusion.py](file:///e:/ARGUS_AI/intelligence/multi_camera_evidence_fusion.py) | Aggregates multi-camera observations into unified identity probability scores |
| [open_set_recognizer.py](file:///e:/ARGUS_AI/intelligence/open_set_recognizer.py) | Open-set recognition engine enforcing open-world rejection bounds |
| [policy_engine.py](file:///e:/ARGUS_AI/intelligence/policy_engine.py) | Enforces operational security rules and alert escalation policies |
| [quality_assessment.py](file:///e:/ARGUS_AI/intelligence/quality_assessment.py) | Assesses silhouette clarity, GEI resolution, and bounding box quality |
| [recognition_deferral_engine.py](file:///e:/ARGUS_AI/intelligence/recognition_deferral_engine.py) | Defers recognition decisions under high occlusion until evidence accumulates |
| [reid_cache.py](file:///e:/ARGUS_AI/intelligence/reid_cache.py) | Thread-safe LRU feature vector cache for fast ReID candidate lookup |
| [score_normalizer.py](file:///e:/ARGUS_AI/intelligence/score_normalizer.py) | Normalizes raw similarity distributions into calibrated probability scores |
| [track_recovery_manager.py](file:///e:/ARGUS_AI/intelligence/track_recovery_manager.py) | Recovers lost person tracks after extended occlusions or scene exits |
| [track_reliability_scorer.py](file:///e:/ARGUS_AI/intelligence/track_reliability_scorer.py) | Scores temporal track consistency, motion smoothness, and bounding box stability |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

Pipeline Feature Embeddings → `intelligence/open_set_recognizer.py` & `intelligence/dual_modal_fusion.py` → `intelligence/decision_engine.py` → `intelligence/missing_person_workflow.py` → Alert / Security Output.

## Configuration

- [configs/inference.yaml](file:///e:/ARGUS_AI/configs/inference.yaml): thresholds, ReID, watchlist, crowd intelligence, topology parameters

## Public Interfaces

- `OpenSetRecognizer`: Open-set matching engine in [intelligence/open_set_recognizer.py](file:///e:/ARGUS_AI/intelligence/open_set_recognizer.py).
- `DualModalFusion`: Adaptive gait and appearance fusion in [intelligence/dual_modal_fusion.py](file:///e:/ARGUS_AI/intelligence/dual_modal_fusion.py).
- `CrowdIntelligenceSystem`: Crowd orchestrator in [intelligence/crowd_intelligence_system.py](file:///e:/ARGUS_AI/intelligence/crowd_intelligence_system.py).
- `MissingPersonWorkflow` (`WatchlistManager`): Watchlist engine in [intelligence/missing_person_workflow.py](file:///e:/ARGUS_AI/intelligence/missing_person_workflow.py).

## Tests

- [tests/unit/test_dual_modal_fusion.py](file:///e:/ARGUS_AI/tests/unit/test_dual_modal_fusion.py)
- [tests/test_phase6_intelligence.py](file:///e:/ARGUS_AI/tests/test_phase6_intelligence.py)
- [tests/test_watchlist_integration.py](file:///e:/ARGUS_AI/tests/test_watchlist_integration.py)

## Related Documentation

- [Root README](file:///e:/ARGUS_AI/README.md)
- [Pipeline Documentation](file:///e:/ARGUS_AI/pipeline/README.md)
