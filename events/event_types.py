from enum import Enum


class EventType(str, Enum):

    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"

    INFERENCE_STARTED = "inference_started"
    INFERENCE_COMPLETED = "inference_completed"

    ENROLLMENT_STARTED = "enrollment_started"
    ENROLLMENT_COMPLETED = "enrollment_completed"

    TRAINING_STARTED = "training_started"
    TRAINING_COMPLETED = "training_completed"

    ALERT_TRIGGERED = "alert_triggered"

    ERROR_OCCURRED = "error_occurred"