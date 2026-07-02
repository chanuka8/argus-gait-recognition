import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from streaming.buffer_queue import BufferQueue
from streaming.frame_dropper import FrameDropper


def main() -> None:
    queue = BufferQueue(maxsize=3)
    dropper = FrameDropper(max_queue_size=3)

    for frame_id in range(10):
        if dropper.should_drop(queue.size()):
            dropper.record_drop()
            print(f"Dropped frame {frame_id}")
            continue

        pushed = queue.push(f"frame_{frame_id}")

        if pushed:
            print(f"Queued frame {frame_id}")
        else:
            dropper.record_drop()
            print(f"Queue full. Dropped frame {frame_id}")

    print("\nQueue size:", queue.size())
    print("Drop stats:", dropper.stats())

    print("\nReading frames:")
    while not queue.is_empty():
        print(queue.pop())


if __name__ == "__main__":
    main()