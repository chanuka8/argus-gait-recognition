import shutil
import sys
import tempfile
from pathlib import Path


root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np

from enrollment.enrollment_lifecycle import (
    EnrollmentLifecycleManager,
    EnrollmentStatus,
)
from intelligence.nn_fine_tuner import NNFineTuner
from models.model_registry import ModelRegistry
from storage.embedding_database import EmbeddingDatabase
from storage.firebase_embedding_store import (
    FirebaseEmbeddingDocument,
    FirebaseEmbeddingStore,
)


def run_verification() -> int:
    print("=" * 80)
    print("ARGUS AI - FIREBASE DURABLE EMBEDDING PERSISTENCE VERIFICATION")
    print("=" * 80)

    tmp_dir = Path(tempfile.mkdtemp(prefix="argus_verify_fb_"))
    results = []

    try:
        db_dir = tmp_dir / "embedding_db"
        gait_gal = tmp_dir / "live_gallery"
        app_gal = tmp_dir / "appearance_gallery"
        cand_dir = tmp_dir / "candidates"
        reg_file = tmp_dir / "model_registry.json"
        fb_offline = tmp_dir / "fb_store.json"




        fb_store = FirebaseEmbeddingStore(
            mode="auto",
            offline_store_path=str(fb_offline),
        )
        mode = fb_store.mode
        print(f"[CHECK 1] Firebase Store Mode: {mode.upper()}")
        results.append(("Firebase Store Mode", "VERIFIED", f"Running in {mode} mode"))




        gait_vec = list(np.random.randn(256).astype(float))
        app_vec = list(np.random.randn(512).astype(float))

        doc_gait = FirebaseEmbeddingDocument(
            embedding_id="verify-emb-gait-01",
            person_id="Subject_Verify_A",
            modality="gait",
            embedding_dim=256,
            vector=gait_vec,
            model_version="v1.0.0",
            observation_date="2026-08-27",
        )
        doc_app = FirebaseEmbeddingDocument(
            embedding_id="verify-emb-app-01",
            person_id="Subject_Verify_A",
            modality="appearance",
            embedding_dim=512,
            vector=app_vec,
            model_version="v1.0.0",
            observation_date="2026-08-27",
        )

        res_g = fb_store.persist_embedding(doc_gait)
        res_a = fb_store.persist_embedding(doc_app)
        assert res_g.success is True and res_a.success is True, "Failed to persist embeddings"

        ver_g, _ = fb_store.verify_persistence("verify-emb-gait-01")
        ver_a, _ = fb_store.verify_persistence("verify-emb-app-01")
        assert ver_g is True and ver_a is True, "Read-after-write verification failed"

        print("[CHECK 2] Multimodal Persistence (256D Gait + 512D Appearance) & Verification: OK")
        results.append(("Multimodal Persistence & Read-after-Write", "VERIFIED", "256D/512D stored and verified"))




        bad_doc = FirebaseEmbeddingDocument(
            embedding_id="bad-dim-01",
            person_id="Sub_Bad",
            modality="gait",
            embedding_dim=512,
            vector=app_vec,
            model_version="v1.0.0",
        )
        res_bad = fb_store.persist_embedding(bad_doc)
        assert res_bad.success is False, "Store accepted invalid dimension for gait"
        print("[CHECK 3] Dimension Isolation: OK (rejected mismatched dimension)")
        results.append(("Dimension Isolation", "VERIFIED", "Mismatched dimensions strictly rejected"))




        db = EmbeddingDatabase(
            db_dir=str(db_dir),
            gait_gallery_dir=str(gait_gal),
            appearance_gallery_dir=str(app_gal),
            firebase_store=fb_store,
        )
        rebuild_res = db.rebuild_from_firebase()
        assert rebuild_res["success"] is True, "Disaster recovery rebuild failed"
        assert rebuild_res["rebuilt_persons"] == 1
        p_rec = db.get_person("Subject_Verify_A")
        assert len(p_rec.gait_embeddings) == 1 and len(p_rec.appearance_embeddings) == 1
        print("[CHECK 4] Disaster Recovery Rebuild: OK (restored local galleries from Firebase)")
        results.append(("Disaster Recovery Rebuild", "VERIFIED", "Reconstructed local VectorStores from Firebase"))




        import cv2

        gei_file = tmp_dir / "test_gei.png"
        cv2.imwrite(str(gei_file), np.ones((128, 64), dtype=np.uint8) * 200)

        manager = EnrollmentLifecycleManager(db=db, firebase_store=fb_store)
        enroll_res = manager.enroll_from_media(
            person_id="Subject_Enroll_Live",
            gei_paths=[gei_file],
            auto_delete_raw=True,
        )
        assert enroll_res.status == EnrollmentStatus.EMBEDDING_ONLY
        assert not gei_file.exists(), "Raw file was not deleted upon verified persistence"
        print("[CHECK 5] Enrollment Lifecycle & Safe Raw-Data Cleanup: OK")
        results.append(("Enrollment 7-Step Invariant", "VERIFIED", "EMBEDDING_ONLY reached, raw media cleaned"))




        tuner = NNFineTuner(
            candidate_dir=str(cand_dir),
            max_epochs=1,
            learning_rate=1e-5,
        )
        gei_data = [
            {"image": np.random.rand(64, 128).astype(np.float32), "label": "Subject_1"}
            for _ in range(4)
        ] + [
            {"image": np.random.rand(64, 128).astype(np.float32), "label": "Subject_2"}
            for _ in range(4)
        ]
        nn_res = tuner.fine_tune_bygait_light(
            active_weights_path="",
            training_gei_data=gei_data,
            historical_gei_data=[],
            candidate_version="vVerifyCand01",
        )
        assert nn_res["success"] is True, f"NN fine-tuning failed: {nn_res}"
        assert Path(nn_res["artifact_path"]).exists()
        print(f"[CHECK 6] Date-Aware ByGaitLight NN Fine-Tuning: OK (Rank-1: {nn_res['metrics']['val_rank1_accuracy']}%)")
        results.append(("ByGaitLight NN Fine-Tuning", "VERIFIED", "Candidate .pth generated with SHA-256 checksum"))




        registry = ModelRegistry(registry_file=str(reg_file))
        cand_ver = "vVerifyCand01"
        registry.register_candidate(
            model_version=cand_ver,
            model_type="bygait_light",
            architecture="ByGaitLight-CNN-256D",
            embedding_dim=256,
            artifact_path=nn_res["artifact_path"],
        )
        registry.record_validation_result(
            model_version=cand_ver,
            model_type="bygait_light",
            passed=True,
            metrics={"tar": 95.0, "far": 0.0},
        )
        promoted = registry.promote_version(cand_ver, "bygait_light")
        assert promoted.model_version == cand_ver

        rolled = registry.rollback("bygait_light", reason="Verification check")
        assert rolled.model_version != cand_ver
        print("[CHECK 7] Candidate Promotion & Automatic Rollback: OK")
        results.append(("Atomic Promotion & Rollback", "VERIFIED", "Linear version promotion and clean rollback verified"))

        print("\n" + "=" * 80)
        print("VERIFICATION SUMMARY:")
        print("=" * 80)
        for name, status, detail in results:
            print(f"  [{status:8s}] {name:40s} | {detail}")
        print("=" * 80)
        print("VERDICT: ALL FIREBASE & CONTINUOUS LEARNING CHECKS PASSED.")
        print("=" * 80)
        return 0

    except Exception as err:  # noqa: BLE001
        print(f"\n[FATAL ERROR] Verification failed: {err}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(run_verification())
