from security_layer.security_logger import SecurityLogger


class SecurityEngine:

    def __init__(
        self,
        confidence_threshold=0.80,
    ):

        self.confidence_threshold = confidence_threshold
        self.logger = SecurityLogger()

    def evaluate(
        self,
        track_id,
        identity,
        score,
        camera_id="default",
    ):

        severity = "INFO"
        decision = "ALLOW"

        if identity == "UNKNOWN":

            severity = "HIGH"
            decision = "SECURITY_ALERT"

        elif score < self.confidence_threshold:

            severity = "MEDIUM"
            decision = "REVIEW_REQUIRED"

        self.logger.log(
            track_id=track_id,
            identity=identity,
            score=score,
            severity=severity,
            decision=decision,
            camera_id=camera_id,
        )

        return {
            "severity": severity,
            "decision": decision,
        }