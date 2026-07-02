import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from enrollment.enrollment_manager import EnrollmentManager


def prepare_demo_person() -> Path:
    source_dir = Path("data/casia_processed/gei/001")
    target_dir = Path("data/new_input/demo_person_001")

    target_dir.mkdir(parents=True, exist_ok=True)

    for image_path in list(source_dir.glob("*.png"))[:5]:
        shutil.copy(image_path, target_dir / image_path.name)

    return target_dir


def main() -> None:
    person_folder = prepare_demo_person()

    manager = EnrollmentManager()
    result = manager.enroll_person(str(person_folder))

    print("\n=== ENROLLMENT TEST ===")
    print(result)


if __name__ == "__main__":
    main()