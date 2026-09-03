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
| --- | --- |
| [auth.py](auth.py) | Module/resource file auth.py |
| [authorization.py](authorization.py) | Module/resource file authorization.py |
| [credentials.py](credentials.py) | Encryption and credentials storage manager for RTSP stream passwords |
| [password_hasher.py](password_hasher.py) | Module/resource file password_hasher.py |
| [security_engine.py](security_engine.py) | Security rule engine classifying recognition scores into severity decision tiers |
| [security_logger.py](security_logger.py) | Thread-safe CSV logger persisting security audit events to `outputs/logs/security/security_events.csv` |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

Recognition Match Output → `security_layer/security_engine.py` → Security Decision Tier → `security_layer/security_logger.py` → `outputs/logs/security/security_events.csv`.

## Configuration

- [configs/system.yaml](../configs/system.yaml): `recognition.security_threshold`
- [configs/cameras.yaml](../configs/cameras.yaml): RTSP camera credentials

## Public Interfaces

- `SecurityEngine`: Security rule evaluator in [security_layer/security_engine.py](security_engine.py).
- `SecurityLogger`: Audit logger in [security_layer/security_logger.py](security_logger.py).
- `CredentialsManager`: Credentials manager in [security_layer/credentials.py](credentials.py).

## Tests

- [tests/test_audit_verification.py](../tests/test_audit_verification.py)
- [tests/test_rtsp_credentials.py](../tests/test_rtsp_credentials.py)
- [tests/unit/test_output_layout.py](../tests/unit/test_output_layout.py)

## Related Documentation

- [Root README](../README.md)
- [Monitoring Documentation](../monitoring/README.md)
