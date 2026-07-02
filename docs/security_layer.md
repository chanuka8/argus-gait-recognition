# Security Layer

This document details the real-time incident threat evaluation layer, classification rules, and security audit log generation.

---

## 1. Threat Classification Architecture

The security layer acts as an automated risk assessment layer on top of raw biometric matches. Rather than passing simple similarity scores to security operators, the system maps each prediction to a severity ranking and a safety action decision.

The evaluation logic, located in [security_layer/security_engine.py](file:///e:/ARGUS_AI/security_layer/security_engine.py), operates on the following policy matrix:

| Matching Outcome | Score Range | Threat Severity | Action Decision | Overlay Color |
| :--- | :--- | :---: | :---: | :---: |
| **Confirmed Match** | Score $\ge 0.92$ | `INFO` | `ALLOW` | **Green** |
| **Verified Match** | $0.85 \le \text{Score} < 0.92$ | `INFO` | `ALLOW` | **Green** |
| **Low Confidence Match** | $0.70 \le \text{Score} < 0.85$ | `MEDIUM` | `REVIEW_REQUIRED` | **Orange** |
| **Unregistered Person** | Score $< 0.70$ (or matched label `UNKNOWN`) | `HIGH` | `SECURITY_ALERT` | **Red** |

---

## 2. Security Decision Engine Rules

The [SecurityEngine](file:///e:/ARGUS_AI/security_layer/security_engine.py#L4) implements the following decision process:
```python
def evaluate(self, track_id, identity, score, camera_id="default"):
    severity = "INFO"
    decision = "ALLOW"

    if identity == "UNKNOWN":
        severity = "HIGH"
        decision = "SECURITY_ALERT"
    elif score < self.confidence_threshold:
        severity = "MEDIUM"
        decision = "REVIEW_REQUIRED"
        
    # Logs decision and severity to the audit file
    self.logger.log(...)
```

---

## 3. Incident Audit Logging: `security_events.csv`

Every evaluated match outcome is appended to an audit file. The [SecurityLogger](file:///e:/ARGUS_AI/security_layer/security_logger.py#L7) handles writing to the CSV spreadsheet, utilizing a thread lock (`threading.Lock()`) to prevent resource conflicts during parallel thread writes.

The audit file is saved at `outputs/security_logs/security_events.csv` and contains the following column keys:

1. **`timestamp`:** ISO-8601 formatting timestamp (e.g. `2026-06-15T05:52:36.421580`).
2. **`track_id`:** Persistent track ID tracking the target across the stream.
3. **`identity`:** The matched subject ID code (e.g. `034`) or `UNKNOWN`.
4. **`score`:** Similarity matching score value rounded to 4 decimals (e.g. `0.9421`).
5. **`severity`:** Risk rating classification (`INFO`, `MEDIUM`, `HIGH`).
6. **`decision`:** Action directive (`ALLOW`, `REVIEW_REQUIRED`, `SECURITY_ALERT`).
7. **`camera_id`:** The source identifier of the camera feed (defaults to `cam_01` or `default`).
