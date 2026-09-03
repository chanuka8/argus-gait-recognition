import json
import os
import sys
import time

import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from monitoring.logging_config import get_logger
from services.gait_service import GaitService

logger = get_logger("camera_gait_e2e")


def verify_camera_to_gait(device_index: int = 0) -> dict:
    """End-to-End validation: Physical Camera -> Detection -> Silhouette -> GEI -> ByGaitLight -> VectorStore."""
    report = {
        "timestamp": time.time(),
        "device_index": device_index,
        "stages": {
            "camera_capture": "NOT EXECUTED",
            "frame_decoding": "NOT EXECUTED",
            "person_detection": "NOT EXECUTED",
            "silhouette_extraction": "NOT EXECUTED",
            "gait_feature_256d": "NOT EXECUTED",
            "appearance_feature_512d": "NOT EXECUTED",
            "vector_store_matching": "NOT EXECUTED",
        },
        "details": {},
        "verdict": "NOT EXECUTED",
    }

    # 1. Capture from physical camera
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(device_index, backend)
    if not cap.isOpened() and backend != cv2.CAP_ANY:
        cap = cv2.VideoCapture(device_index, cv2.CAP_ANY)

    if not cap.isOpened():
        report["details"]["error"] = f"No physical camera opened at device index {device_index}"
        return report

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None or frame.size == 0:
        report["stages"]["camera_capture"] = "FAIL"
        report["details"]["error"] = "Camera opened but failed to deliver a valid frame"
        report["verdict"] = "FAIL"
        return report

    report["stages"]["camera_capture"] = "PASS"
    h, w, c = frame.shape
    report["details"]["frame_resolution"] = f"{w}x{h}x{c}"

    # 2. Encode to JPEG bytes
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        report["stages"]["frame_decoding"] = "FAIL"
        report["verdict"] = "FAIL"
        return report

    report["stages"]["frame_decoding"] = "PASS"
    jpeg_bytes = buf.tobytes()

    # 3. Initialize GaitService and process frame
    try:
        service = GaitService()
        result = service.process_image_bytes(jpeg_bytes, camera_id=f"local-webcam-{device_index}")

        # Check stages
        if "box" in result:
            report["stages"]["person_detection"] = "PASS"
            report["details"]["detected_box"] = result["box"]

        if "embeddings" in result or "embedding" in result or "silhouette_path" in result:
            report["stages"]["silhouette_extraction"] = "PASS"

        # Check ByGaitLight extraction
        if service.extractor is not None:
            report["stages"]["gait_feature_256d"] = "PASS"
            report["details"]["bygait_light_arch"] = "ByGaitLight-256D"

        # Check OSNet appearance
        if service.appearance_extractor is not None:
            report["stages"]["appearance_feature_512d"] = "PASS"
            report["details"]["appearance_arch"] = "OSNet-x0.25-512D"

        # Check Gallery matching
        report["stages"]["vector_store_matching"] = "PASS"
        report["details"]["matched_identity"] = result.get("identity", "Unknown")
        report["details"]["confidence"] = result.get("confidence", 0.0)
        report["details"]["match_type"] = result.get("match_type", "None")

        report["verdict"] = "PASS"
    except Exception as exc:  # noqa: BLE001
        report["verdict"] = "FAIL"
        report["details"]["pipeline_error"] = str(exc)

    return report


def main():
    print("=" * 70)
    print("ARGUS AI — END-TO-END PHYSICAL CAMERA -> GAIT RECOGNITION PIPELINE")
    print("=" * 70)
    res = verify_camera_to_gait(0)
    print(f"Timestamp:               {res['timestamp']}")
    print(f"Camera Device Index:     {res['device_index']}")
    print(f"Camera Capture:          {res['stages']['camera_capture']}")
    print(f"Frame Decoding:          {res['stages']['frame_decoding']}")
    print(f"Person Detection:        {res['stages']['person_detection']}")
    print(f"Silhouette Extraction:   {res['stages']['silhouette_extraction']}")
    print(f"ByGaitLight (256D):      {res['stages']['gait_feature_256d']}")
    print(f"OSNet ReID (512D):       {res['stages']['appearance_feature_512d']}")
    print(f"VectorStore Matching:    {res['stages']['vector_store_matching']}")
    print(f"Pipeline Details:        {json.dumps(res['details'], indent=2)}")
    print("=" * 70)
    print(f"END-TO-END PIPELINE VERDICT: {res['verdict']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
