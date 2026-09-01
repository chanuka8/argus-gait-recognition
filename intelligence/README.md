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
| [accuracy_validation_gate.py](accuracy_validation_gate.py) | Module/resource file accuracy_validation_gate.py |
| [alert_manager.py](alert_manager.py) | Threat alert throttling and priority alert dispatching |
| [appearance_embedding.py](appearance_embedding.py) | Feature extractor and embedding generator for person appearance ReID |
| [background_learning_worker.py](background_learning_worker.py) | Module/resource file background_learning_worker.py |
| [camera_topology_learner.py](camera_topology_learner.py) | Learns directed topology graphs and travel-time bounds across cameras |
| [camera_transition_model.py](camera_transition_model.py) | Evaluates cross-camera travel times against learned probability bounds |
| [candidate_validator.py](candidate_validator.py) | Module/resource file candidate_validator.py |
| [concurrent_track_manager.py](concurrent_track_manager.py) | Module/resource file concurrent_track_manager.py |
| [confidence_scorer.py](confidence_scorer.py) | Multi-factor confidence scoring combining similarity, quality, and track length |
| [confusion_detector.py](confusion_detector.py) | Module/resource file confusion_detector.py |
| [continual_learning_audit_trail.py](continual_learning_audit_trail.py) | Module/resource file continual_learning_audit_trail.py |
| [continual_learning_evaluator.py](continual_learning_evaluator.py) | Module/resource file continual_learning_evaluator.py |
| [continuous_improvement_engine.py](continuous_improvement_engine.py) | Module/resource file continuous_improvement_engine.py |
| [cross_camera_tracker.py](cross_camera_tracker.py) | Tracks target identities across multi-camera topology networks |
| [crowd_density_estimator.py](crowd_density_estimator.py) | Estimates crowd density and spatial clutter surrounding detected subjects |
| [crowd_intelligence_system.py](crowd_intelligence_system.py) | Unified orchestrator for crowd-robust recognition and occlusion deferral |
| [crowd_occlusion_analyzer.py](crowd_occlusion_analyzer.py) | Analyzes bounding box overlap and inter-person occlusion ratio |
| [crowd_robustness_manager.py](crowd_robustness_manager.py) | Controls recognition threshold adaptation under heavy crowd congestion |
| [date_aware_learning_scheduler.py](date_aware_learning_scheduler.py) | Module/resource file date_aware_learning_scheduler.py |
| [decision_engine.py](decision_engine.py) | Tiered decision policy for identity classification and confidence assignment |
| [drift_detector.py](drift_detector.py) | Module/resource file drift_detector.py |
| [dual_modal_fusion.py](dual_modal_fusion.py) | Adaptive fusion engine combining gait GEI and appearance ReID embeddings |
| [event_timeline_reconstructor.py](event_timeline_reconstructor.py) | Module/resource file event_timeline_reconstructor.py |
| [explainable_recognition_report.py](explainable_recognition_report.py) | Module/resource file explainable_recognition_report.py |
| [fusion_diagnostics.py](fusion_diagnostics.py) | Module/resource file fusion_diagnostics.py |
| [fusion_weights.py](fusion_weights.py) | Quality-aware dynamic weight calculator for multi-modal similarity scores |
| [human_review_decision.py](human_review_decision.py) | Manages human operator review queues for borderline verification decisions |
| [identity_persistence.py](identity_persistence.py) | Applies temporal score decay and maintains track identity continuity |
| [learned_fusion.py](learned_fusion.py) | Module/resource file learned_fusion.py |
| [longitudinal_accuracy_evaluator.py](longitudinal_accuracy_evaluator.py) | Module/resource file longitudinal_accuracy_evaluator.py |
| [missing_person_workflow.py](missing_person_workflow.py) | Operational watchlist manager and missing person search workflow |
| [multi_camera_evidence_fusion.py](multi_camera_evidence_fusion.py) | Aggregates multi-camera observations into unified identity probability scores |
| [nn_fine_tuner.py](nn_fine_tuner.py) | Module/resource file nn_fine_tuner.py |
| [open_set_recognizer.py](open_set_recognizer.py) | Open-set recognition engine enforcing open-world rejection bounds |
| [operational_embedding_collector.py](operational_embedding_collector.py) | Module/resource file operational_embedding_collector.py |
| [operational_evidence_manager.py](operational_evidence_manager.py) | Module/resource file operational_evidence_manager.py |
| [policy_engine.py](policy_engine.py) | Enforces operational security rules and alert escalation policies |
| [quality_assessment.py](quality_assessment.py) | Assesses silhouette clarity, GEI resolution, and bounding box quality |
| [recognition_deferral_engine.py](recognition_deferral_engine.py) | Defers recognition decisions under high occlusion until evidence accumulates |
| [reid_cache.py](reid_cache.py) | Thread-safe LRU feature vector cache for fast ReID candidate lookup |
| [score_calibrator.py](score_calibrator.py) | Module/resource file score_calibrator.py |
| [score_normalizer.py](score_normalizer.py) | Normalizes raw similarity distributions into calibrated probability scores |
| [statistical_accuracy_validator.py](statistical_accuracy_validator.py) | Module/resource file statistical_accuracy_validator.py |
| [track_identity_aggregator.py](track_identity_aggregator.py) | Module/resource file track_identity_aggregator.py |
| [track_recovery_manager.py](track_recovery_manager.py) | Recovers lost person tracks after extended occlusions or scene exits |
| [track_reliability_scorer.py](track_reliability_scorer.py) | Scores temporal track consistency, motion smoothness, and bounding box stability |
| [training_dataset_builder.py](training_dataset_builder.py) | Module/resource file training_dataset_builder.py |
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
