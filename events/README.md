# Events

The `events` package implements an in-memory publish-subscribe event bus and dispatcher pattern for asynchronous event handling and module decoupling in ARGUS AI.

## Responsibilities

- Providing event queue abstractions for pipeline, security, and alert notifications.
- Dispatching events to registered listener callbacks without blocking pipeline threads.
- Defining structured event type data contracts across system components.
- Boundaries: Does not write events to disk directly (handled by `utils/event_logger.py` and `security_layer/security_logger.py`).

## Key Modules

<!-- BEGIN SYNC: KEY_MODULES -->
| Module | Purpose |
|---|---|
| [dispatcher.py](file:///E:/ARGUS_AI/events/dispatcher.py) | Asynchronous event dispatcher managing callback execution pools |
| [event_bus.py](file:///E:/ARGUS_AI/events/event_bus.py) | Centralized thread-safe event bus for publishing and subscribing to topics |
| [event_types.py](file:///E:/ARGUS_AI/events/event_types.py) | Data classes and enum definitions for system, recognition, and alert events |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

Pipeline Step / Alert → `events/event_bus.py` → `events/dispatcher.py` → Registered Listener Handlers.

## Configuration

- [configs/system.yaml](file:///e:/ARGUS_AI/configs/system.yaml): system event settings

## Public Interfaces

- `EventBus`: Central pub-sub message broker in [events/event_bus.py](file:///e:/ARGUS_AI/events/event_bus.py).
- `EventDispatcher`: Threaded event dispatcher in [events/dispatcher.py](file:///e:/ARGUS_AI/events/dispatcher.py).
- `Event`, `EventType`: Data structures in [events/event_types.py](file:///e:/ARGUS_AI/events/event_types.py).

## Tests

- [tests/unit/test_output_layout.py](file:///e:/ARGUS_AI/tests/unit/test_output_layout.py)
- [tests/test_audit_verification.py](file:///e:/ARGUS_AI/tests/test_audit_verification.py)

## Related Documentation

- [Root README](file:///e:/ARGUS_AI/README.md)
- [Utils Documentation](file:///e:/ARGUS_AI/utils/README.md)
