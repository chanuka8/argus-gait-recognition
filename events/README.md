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
| --- | --- |
| [dispatcher.py](dispatcher.py) | Asynchronous event dispatcher managing callback execution pools |
| [event_bus.py](event_bus.py) | Centralized thread-safe event bus for publishing and subscribing to topics |
| [event_types.py](event_types.py) | Data classes and enum definitions for system, recognition, and alert events |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

Pipeline Step / Alert → `events/event_bus.py` → `events/dispatcher.py` → Registered Listener Handlers.

## Configuration

- [configs/system.yaml](../configs/system.yaml): system event settings

## Public Interfaces

- `EventBus`: Central pub-sub message broker in [events/event_bus.py](event_bus.py).
- `EventDispatcher`: Threaded event dispatcher in [events/dispatcher.py](dispatcher.py).
- `Event`, `EventType`: Data structures in [events/event_types.py](event_types.py).

## Tests

- [tests/unit/test_output_layout.py](../tests/unit/test_output_layout.py)
- [tests/test_audit_verification.py](../tests/test_audit_verification.py)

## Related Documentation

- [Root README](../README.md)
- [Utils Documentation](../utils/README.md)
