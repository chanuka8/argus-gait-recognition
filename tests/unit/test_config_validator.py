"""
Unit tests for ConfigValidator and RTSP credential sanitization.
"""

from pathlib import Path
from utils.config_validator import ConfigValidator, sanitize_rtsp_url


def test_rtsp_credential_sanitization_edge_cases():

    multi = "Cam1: rtsp://user1:pass1@10.0.0.1/stream, Cam2: rtsp://user2:pass2@10.0.0.2/stream"
    s_multi = sanitize_rtsp_url(multi)
    assert "pass1" not in s_multi
    assert "pass2" not in s_multi
    assert "user1:***@" in s_multi
    assert "user2:***@" in s_multi

    encoded = "rtsp://user%40domain:secret%21@192.168.1.50:554/live"
    s_encoded = sanitize_rtsp_url(encoded)
    assert "secret%21" not in s_encoded
    assert "user%40domain:***@" in s_encoded

    query_url = "rtsp://admin:pass123@192.168.1.100:554/stream?channel=1&subtype=0"
    s_query = sanitize_rtsp_url(query_url)
    assert "pass123" not in s_query
    assert "?channel=1&subtype=0" in s_query

    exc_text = "ConnectionFailedError: Failed to connect to rtsp://admin:secret_pass@192.168.1.10:554/live after 5 retries"
    s_exc = sanitize_rtsp_url(exc_text)
    assert "secret_pass" not in s_exc
    assert "ConnectionFailedError" in s_exc


def test_config_validator_all(tmp_path: Path):
    validator = ConfigValidator(configs_dir="configs")
    results = validator.validate_all()

    assert "inference.yaml" in results
    assert "cameras.yaml" in results
    assert "system.yaml" in results

    for cfg_file, errors in results.items():
        assert len(errors) == 0, f"Config validation errors in {cfg_file}: {errors}"


def test_config_validator_detects_invalid_values():
    validator = ConfigValidator()

    bad_inference = {
        "inference_backend": {
            "backend": "invalid_backend_name",
            "device": "invalid_device",
            "precision": "fp64",
            "max_batch_size": -1,
        },
        "evaluation_threshold": 1.5,
    }

    errors = validator.validate_inference_config(bad_inference)
    assert len(errors) >= 3
    assert any("Invalid backend" in e for e in errors)
    assert any("Invalid device" in e for e in errors)
    assert any("evaluation_threshold" in e for e in errors)
