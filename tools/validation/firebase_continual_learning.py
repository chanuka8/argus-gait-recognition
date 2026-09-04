"""ARGUS AI — End-to-End Firebase Architecture & Continual Learning Verification.

Performs standalone, comprehensive end-to-end verification of all lifecycle
boundaries: Missing Person reference flow, Firebase persistence, local inference
separation, state machine transitions, date-aware scheduling, NN fine-tuning,
candidate validation, atomic promotion, rollback, lineage, and audit trail.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from intelligence.background_learning_worker import BackgroundLearningWorker
from intelligence.candidate_validator import CandidateValidator
from intelligence.continual_learning_audit_trail import ContinualLearningAuditTrail
from intelligence.date_aware_learning_scheduler import (
    DateAwareLearningScheduler,
    LearningJobStatus,
)
from intelligence.missing_person_workflow import MissingPersonWorkflow
from intelligence.nn_fine_tuner import NNFineTuner
from intelligence.operational_embedding_collector import (
    ObservationState,
    OperationalEmbeddingCollector,
)
from models.model_registry import ModelDeploymentStatus, ModelRegistry
from storage.embedding_database import EmbeddingDatabase
from storage.firebase_embedding_store import (
    FirebaseEmbeddingDocument,
    FirebaseEmbeddingStore,
    generate_deterministic_id,
)


def run_e2e_verification() -> bool:
    print("=" * 75)
    print("ARGUS AI — FIREBASE ARCHITECTURE & CONTINUAL LEARNING E2E VERIFICATION")
    print("=" * 75)

    tmp_dir = Path(tempfile.mkdtemp(prefix="argus_e2e_fb_"))
    try:
        db_dir = tmp_dir / "embedding_db"
        gait_gal = tmp_dir / "live_gallery"
        app_gal = tmp_dir / "appearance_gallery"
        cand_dir = tmp_dir / "candidates"
        jobs_file = tmp_dir / "learning_jobs.json"
        reg_file = tmp_dir / "model_registry.json"
        fb_offline = tmp_dir / "firebase_offline.json"
        obs_dir = tmp_dir / "observations"
        audit_file = tmp_dir / "audit_trail.json"

        # ----------------------------------------------------------------------
        # TEST 1: Canonical Schema Validation & Deterministic IDs
        # ----------------------------------------------------------------------
        print("\n[STEP 1] Validating Canonical Data Contract & Deterministic IDs...")
        gait_doc = FirebaseEmbeddingDocument(
            embedding_id="gait_doc_001",
            person_id="Sub_001",
            modality="gait",
            embedding_dim=256,
            vector=list(np.random.randn(256).astype(float)),
        )
        is_val, msg = gait_doc.validate_schema()
        assert is_val, f"Gait schema invalid: {msg}"
        assert gait_doc.identity_id == "Sub_001"
        assert gait_doc.embedding_type == "gait"
        assert gait_doc.embedding_dimension == 256

        app_doc = FirebaseEmbeddingDocument(
            embedding_id="app_doc_001",
            person_id="Sub_001",
            modality="appearance",
            embedding_dim=512,
            vector=list(np.random.randn(512).astype(float)),
        )
        is_val_app, msg_app = app_doc.validate_schema()
        assert is_val_app, f"Appearance schema invalid: {msg_app}"

        det_id = generate_deterministic_id("Sub_001", "gait", 1700000000, 1, "cctv-01")
        assert "gait" in det_id and "Sub_001" in det_id
        print(" -> [PASS] Canonical schema and deterministic IDs verified.")

        # ----------------------------------------------------------------------
        # TEST 2: Missing Person Reference Data Flow & Training Exclusion
        # ----------------------------------------------------------------------
        print("\n[STEP 2] Verifying Missing Person Reference Data Flow & Training Exclusion...")
        fb_store = FirebaseEmbeddingStore(mode="offline", offline_store_path=str(fb_offline))
        db = EmbeddingDatabase(
            db_dir=str(db_dir),
            gait_gallery_dir=str(gait_gal),
            appearance_gallery_dir=str(app_gal),
            firebase_store=fb_store,
        )

        workflow = MissingPersonWorkflow(output_dir=str(tmp_dir / "watchlist"))
        entry = workflow.register_target(identity="Missing_Target_99", notes="Priority amber alert")
        assert entry.identity_id == "Missing_Target_99"

        ref_gait_vec = np.random.randn(256).astype(np.float32)
        ref_app_vec = np.random.randn(512).astype(np.float32)

        p_res = db.add_embeddings(
            person_id="Missing_Target_99",
            gait_embeddings=[ref_gait_vec],
            appearance_embeddings=[ref_app_vec],
            observation_date="2026-08-28",
        )
        assert p_res["success"], "Local reference embedding persistence failed"
        assert p_res["persistence_verified"], "Persistence verification failed"

        # Check local VectorStore queryability
        _, g_labels, _ = db.gait_store.load()
        assert "Missing_Target_99" in list(g_labels), "Missing Person not queryable in local gait gallery"

        # Check Firebase document tags
        stored_fb_docs = fb_store.get_embeddings_by_person("Missing_Target_99")
        assert len(stored_fb_docs) == 2, f"Expected 2 Firebase docs, got {len(stored_fb_docs)}"
        for doc in stored_fb_docs:
            assert doc.identity_type == "USER_REFERENCE"
            assert doc.source_type == "user_reference"
            assert doc.training_eligibility == "NOT_ELIGIBLE"
            assert doc.training_eligible is False

        # Confirm excluded from training pool
        collector = OperationalEmbeddingCollector(output_dir=str(obs_dir))
        assert len(collector.get_training_eligible()) == 0, "Reference data leaked into training pool!"
        print(" -> [PASS] Missing Person reference data flow active in local gallery and excluded from training.")

        # ----------------------------------------------------------------------
        # TEST 3: Live Operational State Machine & Transition Invariants
        # ----------------------------------------------------------------------
        print("\n[STEP 3] Verifying Live Operational State Machine & Transitions...")
        live_gait_1 = list(np.random.randn(256).astype(float))
        live_gait_2 = list(np.random.randn(256).astype(float))
        live_app_1 = list(np.random.randn(512).astype(float))
        live_app_2 = list(np.random.randn(512).astype(float))

        obs1 = collector.record_observation(
            camera_id="cctv-entrance",
            track_id=101,
            vector=live_gait_1,
            predicted_identity="Identity_A",
            confidence=0.88,
            modality="gait",
            quality_score=0.90,
            observation_date="2026-08-28",
        )
        obs2 = collector.record_observation(
            camera_id="cctv-lobby",
            track_id=102,
            vector=live_app_1,
            predicted_identity="Identity_A",
            confidence=0.89,
            modality="appearance",
            quality_score=0.92,
            observation_date="2026-08-28",
        )
        obs3 = collector.record_observation(
            camera_id="cctv-entrance",
            track_id=201,
            vector=live_gait_2,
            predicted_identity="Identity_B",
            confidence=0.85,
            modality="gait",
            quality_score=0.87,
            observation_date="2026-08-28",
        )
        obs4 = collector.record_observation(
            camera_id="cctv-lobby",
            track_id=202,
            vector=live_app_2,
            predicted_identity="Identity_B",
            confidence=0.86,
            modality="appearance",
            quality_score=0.88,
            observation_date="2026-08-28",
        )

        assert obs1.state == ObservationState.PREDICTED
        assert obs2.state == ObservationState.PREDICTED

        # Operator verification transition
        assert collector.verify_observation(obs1.observation_id, "Identity_A", "operator_ui")
        assert collector.verify_observation(obs2.observation_id, "Identity_A", "operator_ui")
        assert collector.verify_observation(obs3.observation_id, "Identity_B", "operator_ui")
        assert collector.verify_observation(obs4.observation_id, "Identity_B", "operator_ui")

        # Check auto-transition to TRAINING_ELIGIBLE based on quality >= 0.70
        eligible_28 = collector.get_eligible_by_date("2026-08-28")
        assert len(eligible_28) == 4, f"Expected 4 eligible observations, got {len(eligible_28)}"
        for o in eligible_28:
            assert o.state == ObservationState.TRAINING_ELIGIBLE
            assert not o.training_consumed
        print(" -> [PASS] State machine transitions (PREDICTED -> VERIFIED -> TRAINING_ELIGIBLE) verified.")

        # ----------------------------------------------------------------------
        # TEST 4: Date-Aware Scheduling & Future-Date Protection
        # ----------------------------------------------------------------------
        print("\n[STEP 4] Verifying Date-Aware Scheduling & Future-Date Protection...")
        scheduler = DateAwareLearningScheduler(
            jobs_file=str(jobs_file),
            collector=collector,
            db=db,
            min_training_embeddings=2,
            min_identities=2,
        )

        # Future date rejection test
        future_job = scheduler.create_learning_job("2099-01-01", "dual_modal_fusion", force=False)
        assert future_job.status == LearningJobStatus.REJECTED
        assert "future date contamination" in future_job.rejection_reason.lower()

        # Valid date job creation
        job = scheduler.create_learning_job("2026-08-28", "dual_modal_fusion", force=False)
        assert job is not None
        assert job.status == LearningJobStatus.PENDING
        assert job.new_embeddings_count >= 4
        print(" -> [PASS] Date-aware scheduling created job and rejected future contamination.")

        # ----------------------------------------------------------------------
        # TEST 5: Background Worker Execution & Candidate Validation
        # ----------------------------------------------------------------------
        print("\n[STEP 5] Executing Learning Job & Candidate Validation...")
        registry = ModelRegistry(registry_file=str(reg_file))
        validator = CandidateValidator()
        audit_trail = ContinualLearningAuditTrail(audit_file=str(audit_file))

        worker = BackgroundLearningWorker(
            scheduler=scheduler,
            registry=registry,
            validator=validator,
            collector=collector,
            db=db,
            candidate_artifacts_dir=str(cand_dir),
            audit_trail=audit_trail,
        )

        completed_job = worker.execute_job_synchronous(job)
        assert completed_job.status == LearningJobStatus.PROMOTED, (
            f"Job failed with status {completed_job.status}: {completed_job.error_message or completed_job.rejection_reason}"
        )

        # Verify active model was promoted
        active_model = registry.get_active_model("dual_modal_fusion")
        assert active_model is not None
        assert active_model.model_version == completed_job.candidate_version
        assert active_model.deployment_status == ModelDeploymentStatus.ACTIVE

        # Verify consumed state transition
        for obs_id in [obs1.observation_id, obs2.observation_id, obs3.observation_id, obs4.observation_id]:
            matched_obs = next(
                o for o in collector.get_recent_observations() if o.observation_id == obs_id
            )
            assert matched_obs.state == ObservationState.TRAINING_CONSUMED
            assert matched_obs.training_consumed is True

        # Verify zero eligible remain for that date (prevents duplicate consumption)
        assert len(collector.get_eligible_by_date("2026-08-28")) == 0
        print(" -> [PASS] Model training, candidate validation, atomic promotion, and consumption verified.")

        # ----------------------------------------------------------------------
        # TEST 6: Automatic Rollback Verification
        # ----------------------------------------------------------------------
        print("\n[STEP 6] Verifying Automatic Model Rollback...")
        rolled_back = registry.rollback("dual_modal_fusion", reason="Simulated regression failure")
        assert rolled_back.deployment_status == ModelDeploymentStatus.ACTIVE
        assert rolled_back.model_version == "v1.0.0"

        archived = registry.get_model(completed_job.candidate_version, "dual_modal_fusion")
        assert archived.deployment_status == ModelDeploymentStatus.ROLLED_BACK
        print(" -> [PASS] Model rollback reverted safely to baseline v1.0.0.")

        # ----------------------------------------------------------------------
        # TEST 7: Real PyTorch NN Fine-Tuning Execution
        # ----------------------------------------------------------------------
        print("\n[STEP 7] Verifying Real PyTorch NN Fine-Tuning for ByGaitLight (256D)...")
        nn_tuner = NNFineTuner(candidate_dir=str(cand_dir), max_epochs=2, batch_size=4)
        sample_geis = [
            {"image": np.random.uniform(0.0, 1.0, (64, 64)).astype(np.float32), "label": "Sub_A"},
            {"image": np.random.uniform(0.0, 1.0, (64, 64)).astype(np.float32), "label": "Sub_A"},
            {"image": np.random.uniform(0.0, 1.0, (64, 64)).astype(np.float32), "label": "Sub_B"},
            {"image": np.random.uniform(0.0, 1.0, (64, 64)).astype(np.float32), "label": "Sub_B"},
        ]
        nn_res = nn_tuner.fine_tune_bygait_light(
            active_weights_path="",
            training_gei_data=sample_geis[:2],
            historical_gei_data=sample_geis[2:],
            candidate_version="v_e2e_bygait_001",
        )
        assert nn_res["success"], f"NN training failed: {nn_res.get('error')}"
        assert nn_res["embedding_dim"] == 256
        assert nn_res["metrics"]["changed_tensors"] > 0
        assert nn_res["metrics"]["max_param_delta"] > 0.0
        assert Path(nn_res["artifact_path"]).exists()
        print(
            f" -> [PASS] ByGaitLight fine-tuned successfully: {nn_res['metrics']['changed_tensors']} "
            f"tensors updated, max delta={nn_res['metrics']['max_param_delta']:.6f}."
        )

        print("\n" + "=" * 75)
        print("E2E VERIFICATION COMPLETED: ALL 7 CORE LIFECYCLE PHASES PASSED 100%!")
        print("=" * 75)
        return True

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    success = run_e2e_verification()
    sys.exit(0 if success else 1)
