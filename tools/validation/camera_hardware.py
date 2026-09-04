import argparse
import json
import os
import sys
import time
from typing import Any

import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from monitoring.logging_config import get_logger
from security_layer.credentials import sanitize_rtsp_url

logger = get_logger("camera_hardware_validator")


def probe_opencv_camera(
    device_index: int,
    num_frames: int = 10,
    preferred_backend: int = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY,
) -> dict[str, Any]:
    """Test physical camera capture on a given hardware index."""
    result: dict[str, Any] = {
        "device_index": device_index,
        "status": "NOT EXECUTED",
        "backend": "CAP_DSHOW" if preferred_backend == cv2.CAP_DSHOW else "DEFAULT",
        "resolution": None,
        "measured_fps": 0.0,
        "frames_captured": 0,
        "reconnect_status": "NOT EXECUTED",
        "error": None,
    }

    try:
        cap = cv2.VideoCapture(device_index, preferred_backend)
    except Exception as exc:  # noqa: BLE001
        result["status"] = "FAIL"
        result["error"] = f"Exception opening camera index {device_index}: {exc}"
        return result

    if not cap.isOpened() and preferred_backend != cv2.CAP_ANY:
        cap = cv2.VideoCapture(device_index, cv2.CAP_ANY)
        result["backend"] = "CAP_ANY"

    if not cap.isOpened():
        result["status"] = "NOT EXECUTED"
        result["error"] = f"No physical camera opened at device index {device_index}"
        return result

    try:
        # Read frames to warm up sensor and measure throughput
        start_time = time.time()
        captured = 0
        w, h = 0, 0
        last_frame = None

        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                break
            captured += 1
            h, w = frame.shape[:2]
            last_frame = frame

        duration = max(0.001, time.time() - start_time)

        if captured == 0:
            result["status"] = "FAIL"
            result["error"] = "Camera opened but delivered 0 valid frames"
            cap.release()
            return result

        result["status"] = "PASS"
        result["frames_captured"] = captured
        result["resolution"] = f"{w}x{h}"
        result["measured_fps"] = round(captured / duration, 2)

        # Test JPEG snapshot generation
        if last_frame is not None:
            ok, buf = cv2.imencode(".jpg", last_frame)
            if not ok or len(buf) == 0:
                result["status"] = "FAIL"
                result["error"] = "JPEG compression failed on captured frame"
                cap.release()
                return result

        # Test Reconnect / Release lifecycle
        cap.release()
        time.sleep(0.3)
        re_cap = cv2.VideoCapture(device_index, preferred_backend)
        if re_cap.isOpened():
            re_ret, re_frame = re_cap.read()
            if re_ret and re_frame is not None:
                result["reconnect_status"] = "PASS"
            else:
                result["reconnect_status"] = "FAIL"
            re_cap.release()
        else:
            result["reconnect_status"] = "FAIL"

    except Exception as exc:  # noqa: BLE001
        result["status"] = "FAIL"
        result["error"] = f"Unexpected runtime error during capture: {exc}"
        if cap.isOpened():
            cap.release()

    return result


def probe_rtsp_camera(
    rtsp_url: str,
    timeout_sec: float = 5.0,
    num_frames: int = 10,
) -> dict[str, Any]:
    """Test live RTSP camera connection and frame decoding."""
    safe_url = sanitize_rtsp_url(rtsp_url)
    result: dict[str, Any] = {
        "url": safe_url,
        "status": "NOT EXECUTED",
        "resolution": None,
        "measured_fps": 0.0,
        "frames_captured": 0,
        "reconnect_status": "NOT EXECUTED",
        "error": None,
    }

    if not rtsp_url or not rtsp_url.strip():
        result["status"] = "NOT EXECUTED"
        result["error"] = "No RTSP URL provided"
        return result

    try:
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout_sec * 1000))
    except Exception as exc:  # noqa: BLE001
        result["status"] = "FAIL"
        result["error"] = f"Exception opening RTSP stream: {exc}"
        return result

    if not cap.isOpened():
        result["status"] = "FAIL"
        result["error"] = f"Could not connect to RTSP stream at {safe_url}"
        return result

    try:
        start_time = time.time()
        captured = 0
        w, h = 0, 0
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            captured += 1
            h, w = frame.shape[:2]

        duration = max(0.001, time.time() - start_time)

        if captured == 0:
            result["status"] = "FAIL"
            result["error"] = "RTSP stream opened but produced 0 decoded frames"
            cap.release()
            return result

        result["status"] = "PASS"
        result["frames_captured"] = captured
        result["resolution"] = f"{w}x{h}"
        result["measured_fps"] = round(captured / duration, 2)

        # Test reconnect
        cap.release()
        time.sleep(0.5)
        re_cap = cv2.VideoCapture(rtsp_url)
        if re_cap.isOpened():
            re_ret, re_frame = re_cap.read()
            result["reconnect_status"] = "PASS" if re_ret and re_frame is not None else "FAIL"
            re_cap.release()
        else:
            result["reconnect_status"] = "FAIL"

    except Exception as exc:  # noqa: BLE001
        result["status"] = "FAIL"
        result["error"] = f"RTSP streaming error: {exc}"
        if cap.isOpened():
            cap.release()

    return result


def main():
    parser = argparse.ArgumentParser(description="ARGUS AI Physical Camera Hardware Validation")
    parser.add_argument("--webcam-index", type=int, default=0, help="Local webcam device index (default: 0)")
    parser.add_argument("--usb-index", type=int, default=None, help="External USB webcam device index (default: None)")
    parser.add_argument("--rtsp-url", type=str, default=None, help="Live RTSP stream URL")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "timestamp": time.time(),
        "local_webcam": None,
        "usb_webcam": None,
        "rtsp_stream": None,
        "summary": "NOT EXECUTED",
    }

    # 1. Local Webcam Probe
    local_res = probe_opencv_camera(args.webcam_index)
    report["local_webcam"] = local_res

    # 2. USB Webcam Probe
    usb_index = args.usb_index
    if usb_index is None and os.environ.get("ARGUS_TEST_USB_INDEX"):
        try:
            usb_index = int(os.environ["ARGUS_TEST_USB_INDEX"])
        except ValueError:
            usb_index = None

    if usb_index is not None and usb_index != args.webcam_index:
        usb_res = probe_opencv_camera(usb_index)
    else:
        # Check if an external camera exists at index 1
        usb_res = probe_opencv_camera(1)
        if usb_res["status"] == "NOT EXECUTED":
            usb_res["error"] = "No secondary USB webcam hardware connected at index 1"

    report["usb_webcam"] = usb_res

    # 3. RTSP Camera Probe
    rtsp_url = args.rtsp_url or os.environ.get("ARGUS_TEST_RTSP_URL")
    if rtsp_url:
        rtsp_res = probe_rtsp_camera(rtsp_url)
    else:
        rtsp_res = {
            "url": None,
            "status": "NOT EXECUTED",
            "resolution": None,
            "measured_fps": 0.0,
            "frames_captured": 0,
            "reconnect_status": "NOT EXECUTED",
            "error": "No RTSP stream URL configured (--rtsp-url or ARGUS_TEST_RTSP_URL)",
        }
    report["rtsp_stream"] = rtsp_res

    statuses = [local_res["status"], usb_res["status"], rtsp_res["status"]]
    if any(s == "FAIL" for s in statuses):
        report["summary"] = "FAIL"
    elif any(s == "PASS" for s in statuses):
        report["summary"] = "PASS"
    else:
        report["summary"] = "NOT EXECUTED"

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print("=" * 70)
    print("ARGUS AI — PHYSICAL CAMERA HARDWARE VALIDATION")
    print("=" * 70)
    print(f"1. Local Webcam (index {args.webcam_index}):")
    print(f"   Status:          {local_res['status']}")
    print(f"   Backend:         {local_res['backend']}")
    print(f"   Resolution:      {local_res['resolution'] or 'N/A'}")
    print(f"   Measured FPS:    {local_res['measured_fps']}")
    print(f"   Captured Frames: {local_res['frames_captured']}")
    print(f"   Reconnect Test:  {local_res['reconnect_status']}")
    if local_res["error"]:
        print(f"   Notes:           {local_res['error']}")

    print(f"\n2. USB Webcam (index {usb_index if usb_index is not None else 1}):")
    print(f"   Status:          {usb_res['status']}")
    print(f"   Resolution:      {usb_res['resolution'] or 'N/A'}")
    print(f"   Measured FPS:    {usb_res['measured_fps']}")
    print(f"   Captured Frames: {usb_res['frames_captured']}")
    print(f"   Reconnect Test:  {usb_res['reconnect_status']}")
    if usb_res["error"]:
        print(f"   Notes:           {usb_res['error']}")

    print("\n3. Live RTSP Camera:")
    print(f"   Status:          {rtsp_res['status']}")
    print(f"   Stream URL:      {rtsp_res['url'] or 'N/A'}")
    print(f"   Resolution:      {rtsp_res['resolution'] or 'N/A'}")
    print(f"   Measured FPS:    {rtsp_res['measured_fps']}")
    print(f"   Captured Frames: {rtsp_res['frames_captured']}")
    print(f"   Reconnect Test:  {rtsp_res['reconnect_status']}")
    if rtsp_res["error"]:
        print(f"   Notes:           {rtsp_res['error']}")

    print("=" * 70)
    print(f"HARDWARE VALIDATION VERDICT: {report['summary']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
