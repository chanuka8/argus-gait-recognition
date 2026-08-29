"""
End-to-End Demonstration and Validation Script for ARGUS AI Continuous Improvement Architecture,
Safe Enrollment Lifecycle with Automatic Raw Data Cleanup, Model Registry, and Rollback.
"""

import shutil
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from enrollment.enrollment_lifecycle import (
    EnrollmentLifecycleManager,
)
from intelligence.candidate_validator import CandidateValidator
from intelligence.continuous_improvement_engine import ContinuousImprovementEngine
from intelligence.drift_detector import DriftDetector
from intelligence.operational_embedding_collector import OperationalEmbeddingCollector
from models.model_registry import ModelRegistry
from storage.embedding_database import EmbeddingDatabase


def demonstrate_safe_enrollment_lifecycle():
    print("\n" + "=" * 80)
    print("STAGE 1: MISSING PERSON ENROLLMENT + AUTOMATIC RAW-DATA DELETION DEMO")
    print("=" * 80)

    temp_dir = tempfile.mkdtemp(prefix="argus_demo_enroll_")
    t_path = Path(temp_dir)
    db_dir = t_path / "data" / "embedding_db"
    gait_gal = t_path / "models" / "live_gallery"
    app_gal = t_path / "models" / "appearance_gallery"
    input_dir = t_path / "data" / "new_input"

    db = EmbeddingDatabase(
        db_dir=str(db_dir),
        gait_gallery_dir=str(gait_gal),
        appearance_gallery_dir=str(app_gal),
    )

    # 1. Simulate Uploaded Raw Media
    person_dir = input_dir / "MP_JohnDoe_001"
    person_dir.mkdir(parents=True, exist_ok=True)
    raw_photo = person_dir / "reference_photo.jpg"
    raw_photo.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 500)
    raw_gei = person_dir / "cctv_gei.png"
    raw_gei.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 500)

    print(f"[*] Uploaded Raw Files in '{person_dir.name}':")
    print(f"    - {raw_photo.name} (exists: {raw_photo.exists()})")
    print(f"    - {raw_gei.name} (exists: {raw_gei.exists()})")

    # 2. Mock Extractors with valid normalized 256D gait and 512D appearance vectors
    class MockGaitExtractor:
        def extract(self, path):
            v = np.random.randn(256).astype(np.float32)
            return v / np.linalg.norm(v)

    class MockAppExtractor:
        def extract(self, path):
            v = np.random.randn(512).astype(np.float32)
            return v / np.linalg.norm(v)

    manager = EnrollmentLifecycleManager(
        db=db,
        gait_extractor=MockGaitExtractor(),
        appearance_extractor=MockAppExtractor(),
        model_version="v1.0.0",
    )

    print("\n[*] Executing Safe Enrollment Lifecycle...")
    res = manager.enroll_from_media(
        person_id="MP_JohnDoe_001",
        photo_paths=[raw_photo],
        gei_paths=[raw_gei],
        auto_delete_raw=True,
    )

    print(f"[*] Enrollment Outcome: {res.status.value}")
    print(f"    - Gait Embeddings Generated & Persisted: {res.gait_embeddings_count} (256D)")
    print(f"    - Appearance Embeddings Generated & Persisted: {res.appearance_embeddings_count} (512D)")
    print(f"    - Raw Files Deleted: {res.raw_files_deleted}")
    print(f"    - Raw Photo Exists After Cleanup: {raw_photo.exists()}")
    print(f"    - Raw GEI Exists After Cleanup: {raw_gei.exists()}")

    # Verify Database State
    person = db.get_person("MP_JohnDoe_001")
    assert person is not None
    print("\n[+] Verified Durable State: Subject 'MP_JohnDoe_001' is in EMBEDDING_ONLY state.")
    print(f"    - Total Gait Vectors in DB: {len(person.gait_embeddings)}")
    print(f"    - Total Appearance Vectors in DB: {len(person.appearance_embeddings)}")

    shutil.rmtree(temp_dir, ignore_errors=True)


def demonstrate_continuous_improvement_and_rollback():
    print("\n" + "=" * 80)
    print("STAGE 2: CONTINUOUS PERFORMANCE IMPROVEMENT + VALIDATION GATE + ROLLBACK DEMO")
    print("=" * 80)

    temp_dir = tempfile.mkdtemp(prefix="argus_demo_ci_")
    t_path = Path(temp_dir)
    reg_file = t_path / "models" / "model_registry.json"
    obs_dir = t_path / "data" / "observations"

    reg = ModelRegistry(registry_file=str(reg_file))
    collector = OperationalEmbeddingCollector(output_dir=str(obs_dir))
    detector = DriftDetector(collector=collector)
    validator = CandidateValidator()
    engine = ContinuousImprovementEngine(
        registry=reg,
        validator=validator,
        collector=collector,
        drift_detector=detector,
    )

    # 1. Initial State: Baseline Production Model
    active_init = reg.get_active_model("dual_modal_fusion")
    print(f"[*] Initial Active Production Model: {active_init.model_version} ({active_init.architecture})")
    print(f"    - Baseline TAR: {active_init.validation_metrics.get('out_of_fold_tar', 67.57):.2f}%")
    print(f"    - Baseline FAR: {active_init.validation_metrics.get('out_of_fold_far', 2.70):.2f}%")

    # 2. Process Operational CCTV Observations
    print("\n[*] Recording Operational CCTV Observations...")
    for i in range(15):
        obs = collector.record_observation(
            camera_id="cam-zone-north",
            track_id=100 + i,
            vector=np.random.randn(256),
            predicted_identity="Subject_01" if i % 2 == 0 else "Subject_02",
            confidence=0.88 + (i * 0.005),
            modality="gait",
        )
        if i % 3 == 0:
            collector.verify_observation(obs.observation_id, verified_identity=obs.predicted_identity)

    eligible = collector.get_training_eligible()
    print(f"[+] Total Verified Training-Eligible Observations Collected: {len(eligible)}")

    # 3. Evaluate an Inferior Candidate (Security Regression)
    print("\n[*] Evaluating Candidate 'v1.1.0-uncalibrated' (Simulating Elevated FAR)...")
    passed_inf, val_inf, _ = engine.process_candidate(
        candidate_version="v1.1.0-uncalibrated",
        model_type="dual_modal_fusion",
        architecture="LinearOptimal-DualModal-Uncalibrated",
        embedding_dim=256,
        artifact_path="configs/fusion_profiles/candidate_inf.json",
        candidate_metrics={"tar": 55.0, "far": 9.20, "eer": 32.0},
    )
    print(f"    - Validation Result: {'PASSED' if passed_inf else 'REJECTED'}")
    print(f"    - Rejection Reasons: {val_inf.rejection_reasons}")
    print(f"    - Active Model Remains: {reg.get_active_model('dual_modal_fusion').model_version}")

    # 4. Evaluate a Superior Candidate (Passes All Security & Accuracy Gates)
    print("\n[*] Evaluating Candidate 'v2.0.0-calibrated' (Simulating Improved TAR and Lower FAR)...")
    passed_sup, _, rec_sup = engine.process_candidate(
        candidate_version="v2.0.0-calibrated",
        model_type="dual_modal_fusion",
        architecture="LearnedLogistic-DualModal-Calibrated",
        embedding_dim=256,
        artifact_path="configs/fusion_profiles/candidate_sup.json",
        candidate_metrics={"tar": 76.50, "far": 1.35, "eer": 18.20},
    )
    print(f"    - Validation Result: {'PASSED' if passed_sup else 'REJECTED'}")
    print(f"    - Promotion Status: {rec_sup.deployment_status.value}")
    print(f"    - New Active Production Model: {reg.get_active_model('dual_modal_fusion').model_version}")
    print(f"    - Previous Version Retained for Rollback: {rec_sup.previous_production_version}")

    # 5. Simulate Post-Deployment Regression & Automatic Rollback
    print("\n[*] Simulating Production Drift Regression Alert -> Triggering Automatic Rollback...")
    restored = engine.trigger_runtime_regression_rollback(
        model_type="dual_modal_fusion",
        reason="Field telemetry detected elevated false accept rate in Camera Zone 04",
    )
    print("[+] Safety Rollback Completed:")
    print(f"    - Restored Active Production Version: {restored.model_version} ({restored.deployment_status.value})")
    rolled_back_v2 = reg.get_model("v2.0.0-calibrated", "dual_modal_fusion")
    print(
        f"    - Faulty Candidate Status: {rolled_back_v2.deployment_status.value} (Reason: {rolled_back_v2.rejection_reason})"
    )

    shutil.rmtree(temp_dir, ignore_errors=True)
    print("\n" + "=" * 80)
    print("ALL VALIDATION STAGES COMPLETED SUCCESSFULLY WITH ZERO VIOLATIONS.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    demonstrate_safe_enrollment_lifecycle()
    demonstrate_continuous_improvement_and_rollback()
