# Security Layer

The `security_layer` package implements security event classification, decision severity evaluation, thread-safe audit log logging, and secure RTSP/camera credentials handling for ARGUS AI.

## Responsibilities

- Evaluating recognition results against security policy rules (generating ALLOW, REVIEW_REQUIRED, SECURITY_ALERT decisions).
- Maintaining thread-safe, structured audit logs at `outputs/logs/security/security_events.csv`.
- Managing encrypted and plaintext camera stream credentials safely.
- Boundaries: Does not run object detection, video decoding, or CNN model inference.

## Key Modules

<!-- BEGIN SYNC: KEY_MODULES -->
| Module | Purpose |
|---|---|
| [credentials.py](file:///e:/ARGUS_AI/security_layer/credentials.py) | Encryption and credentials storage manager for RTSP stream passwords |
| [security_engine.py](file:///e:/ARGUS_AI/security_layer/security_engine.py) | Security rule engine classifying recognition scores into severity decision tiers |
| [security_logger.py](file:///e:/ARGUS_AI/security_layer/security_logger.py) | Thread-safe CSV logger persisting security audit events to `outputs/logs/security/security_events.csv` |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

Recognition Match Output → `security_layer/security_engine.py` → Security Decision Tier → `security_layer/security_logger.py` → `outputs/logs/security/security_events.csv`.

## Configuration

- [configs/system.yaml](file:///e:/ARGUS_AI/configs/system.yaml): `recognition.security_threshold`
- [configs/cameras.yaml](file:///e:/ARGUS_AI/configs/cameras.yaml): RTSP camera credentials

## Public Interfaces

- `SecurityEngine`: Security rule evaluator in [security_layer/security_engine.py](file:///e:/ARGUS_AI/security_layer/security_engine.py).
- `SecurityLogger`: Audit logger in [security_layer/security_logger.py](file:///e:/ARGUS_AI/security_layer/security_logger.py).
- `CredentialsManager`: Credentials manager in [security_layer/credentials.py](file:///e:/ARGUS_AI/security_layer/credentials.py).

## Tests

- [tests/test_audit_verification.py](file:///e:/ARGUS_AI/tests/test_audit_verification.py)
- [tests/test_rtsp_credentials.py](file:///e:/ARGUS_AI/tests/test_rtsp_credentials.py)
- [tests/unit/test_output_layout.py](file:///e:/ARGUS_AI/tests/unit/test_output_layout.py)

## Related Documentation

- [Root README](file:///e:/ARGUS_AI/README.md)
- [Monitoring Documentation](file:///e:/ARGUS_AI/monitoring/README.md)
