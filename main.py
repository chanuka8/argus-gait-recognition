from app.core.thread_limits import configure as _configure_thread_limits

_configure_thread_limits()  # must run before torch/cv2/numpy are imported anywhere below

from app.core.system import ArgusSystem


def main() -> None:
    system = ArgusSystem(mode="inference")
    system.start()


if __name__ == "__main__":
    main()
