import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from security_layer.security_engine import SecurityEngine


def main():
    engine = SecurityEngine()

    print(
        engine.evaluate(
            track_id=1,
            identity="027",
            score=0.92,
        )
    )

    print(
        engine.evaluate(
            track_id=2,
            identity="027",
            score=0.65,
        )
    )

    print(
        engine.evaluate(
            track_id=3,
            identity="UNKNOWN",
            score=0.31,
        )
    )


if __name__ == "__main__":
    main()