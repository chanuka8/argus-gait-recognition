import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from events.event_bus import EventBus


def on_system_started(data):
    print(f"[EVENT RECEIVED] SYSTEM STARTED -> {data}")


def main():
    bus = EventBus()

    bus.subscribe("system_started", on_system_started)

    bus.publish(
        "system_started",
        {
            "mode": "inference"
        }
    )


if __name__ == "__main__":
    main()