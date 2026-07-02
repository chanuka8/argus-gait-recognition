from core.system import ArgusSystem


def main() -> None:
    system = ArgusSystem(mode="inference")
    system.start()


if __name__ == "__main__":
    main()