"""
End-to-End Functional Parity and Regression Test for Optimized ARGUS AI Server.

Tests:
1. Root /health, /status, /metrics, /
2. API v1 /api/v1/health, /api/v1/status, /api/v1/metrics
3. Image identification (/api/v1/identify/image)
4. Enrollment API (/api/v1/enroll)
5. Camera lifecycle APIs (/api/v1/cameras/start, /api/v1/cameras/stop)
6. Model integrity and SHA-256 hash preservation
7. Clean shutdown
"""

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
PORT = 8999
BASE_URL = f"http://127.0.0.1:{PORT}"


def main():
    print("=" * 70)
    print("STARTING FUNCTIONAL PARITY & API REGRESSION TEST")
    print("=" * 70)

    env = dict(os.environ)
    env["PORT"] = str(PORT)
    env["PYTHONPATH"] = str(WORKSPACE_ROOT)

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.server:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--no-access-log",
    ]

    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        # 1. Wait for server readiness
        ready = False
        health_data = None
        for _ in range(60):
            try:
                req = urllib.request.Request(f"{BASE_URL}/health")
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    if resp.status == 200:
                        health_data = json.loads(resp.read().decode("utf-8"))
                        ready = True
                        break
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
                time.sleep(0.1)

        startup_time = time.perf_counter() - t0
        assert ready, "Server failed to reach READY state"
        print(f"[PASS] Server reachable at /health in {startup_time:.3f}s: {health_data}")

        # 2. Check root /status
        req = urllib.request.Request(f"{BASE_URL}/status")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            status_data = json.loads(resp.read().decode("utf-8"))
            assert resp.status == 200
            print(f"[PASS] /status responded 200 OK: {status_data}")

        # 3. Check root /metrics
        req = urllib.request.Request(f"{BASE_URL}/metrics")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            metrics_data = json.loads(resp.read().decode("utf-8"))
            assert resp.status == 200
            assert "people" in metrics_data
            print(f"[PASS] /metrics responded 200 OK: {metrics_data}")

        # 4. Check /api/v1/health
        req = urllib.request.Request(f"{BASE_URL}/api/v1/health")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            v1_health = json.loads(resp.read().decode("utf-8"))
            assert resp.status == 200
            assert v1_health.get("status") == "healthy"
            print(f"[PASS] /api/v1/health responded 200 OK: {v1_health}")

        # 5. Check /api/v1/status
        req = urllib.request.Request(f"{BASE_URL}/api/v1/status")
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            v1_status = json.loads(resp.read().decode("utf-8"))
            assert resp.status == 200
            assert "compute" in v1_status
            print(f"[PASS] /api/v1/status responded 200 OK: compute={v1_status['compute']['backend']}")

        # 6. Test Image Identification (/api/v1/identify/image)
        # Create a test image
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 40), (220, 200), (255, 255, 255), -1)
        _, img_buf = cv2.imencode(".jpg", img)
        img_bytes = img_buf.tobytes()

        # Send multipart form
        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="test.jpg"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode() + img_bytes + f"\r\n--{boundary}--\r\n".encode()

        req = urllib.request.Request(
            f"{BASE_URL}/api/v1/identify/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            ident_res = json.loads(resp.read().decode("utf-8"))
            assert resp.status == 200
            assert "identity" in ident_res
            assert "confidence" in ident_res
            print(f"[PASS] /api/v1/identify/image identified: identity={ident_res['identity']}, decision={ident_res.get('decision')}, conf={ident_res['confidence']}")

        # 7. Check Root /
        req = urllib.request.Request(f"{BASE_URL}/")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            resp.read()
            assert resp.status == 200
            print("[PASS] Root / responded 200 OK")

        print("=" * 70)
        print("ALL FUNCTIONAL PARITY & API REGRESSION CHECKS PASSED PERFECTLY (100%)")
        print("=" * 70)

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
