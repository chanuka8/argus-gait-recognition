import argparse
import os
import shutil
from pathlib import Path


def get_destination_mapping(relative_path: Path) -> Path:
    parts = relative_path.parts
    if not parts:
        return relative_path

    first = parts[0]

    if first == "camera_stats":
        return Path("monitoring/camera_stats") / Path(*parts[1:])
    elif first == "detection_reports":
        return Path("media/detections") / Path(*parts[1:])
    elif first == "eval_reports":
        return Path("reports/evaluation") / Path(*parts[1:])
    elif first == "evaluation_charts":
        return Path("reports/evaluation/charts") / Path(*parts[1:])
    elif first == "events":
        return Path("logs/events") / Path(*parts[1:])
    elif first == "security_logs":
        return Path("logs/security") / Path(*parts[1:])
    elif first == "videos":
        return Path("media/videos") / Path(*parts[1:])
    elif first == "images":
        return Path("media/images") / Path(*parts[1:])
    elif first == "logs":
        if len(parts) > 1 and "camera" in parts[1].lower():
            return Path("logs/camera") / Path(*parts[1:])
        return Path("logs/system") / Path(*parts[1:])
    elif first == "reports":
        if len(parts) == 1:
            return Path("reports")
        sub = parts[1]
        filename = sub.lower()
        if "benchmark" in filename:
            return Path("reports/benchmark") / Path(*parts[1:])
        elif "explainable" in filename or "lineage" in filename:
            return Path("reports/explainable") / Path(*parts[1:])
        elif "timeline" in filename:
            return Path("reports/timelines") / Path(*parts[1:])
        elif "export" in filename or "topology" in filename:
            return Path("reports/exports") / Path(*parts[1:])
        elif sub in ("explainable", "timelines", "benchmark", "evaluation", "exports"):
            return Path("reports") / Path(*parts[1:])
        elif sub.endswith(".log"):
            return Path("logs/system") / Path(*parts[1:])
        else:
            return Path("reports/evaluation") / Path(*parts[1:])
    elif first in ("argus.pid", "pid"):
        return Path("temporary") / Path(*parts[1:])

    return relative_path


def resolve_conflict_path(dest_path: Path) -> Path:
    if not dest_path.exists():
        return dest_path

    stem = dest_path.stem
    suffix = dest_path.suffix
    parent = dest_path.parent
    counter = 1

    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def migrate_outputs(outputs_dir: Path, dry_run: bool = False) -> list[tuple[Path, Path]]:
    if not outputs_dir.exists():
        return []

    moved_records: list[tuple[Path, Path]] = []

    new_top_level = {
        "reports",
        "logs",
        "monitoring",
        "media",
        "watchlist",
        "temporary",
    }

    legacy_items: list[Path] = []
    for item in outputs_dir.iterdir():
        if item.name in new_top_level:
            continue
        if item.name.startswith("."):
            continue
        legacy_items.append(item)

    for sub in ("logs", "reports"):
        sub_path = outputs_dir / sub
        if sub_path.is_dir():
            for child in sub_path.iterdir():
                if child.is_file() and not child.name.startswith("."):
                    legacy_items.append(child)

    for item in legacy_items:
        if item.is_file():
            rel = item.relative_to(outputs_dir)
            target_rel = get_destination_mapping(rel)
            if target_rel == rel:
                continue

            target_full = outputs_dir / target_rel
            target_file = resolve_conflict_path(target_full)

            print(f"[MOVE] {item} -> {target_file}")
            if not dry_run:
                target_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(item), str(target_file))

            moved_records.append((item, target_file))

        elif item.is_dir():
            for root, dirs, files in os.walk(item):
                for f in files:
                    if f.startswith("."):
                        continue
                    src_file = Path(root) / f
                    rel = src_file.relative_to(outputs_dir)
                    target_rel = get_destination_mapping(rel)
                    target_full = outputs_dir / target_rel
                    target_file = resolve_conflict_path(target_full)

                    print(f"[MOVE] {src_file} -> {target_file}")
                    if not dry_run:
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(src_file), str(target_file))

                    moved_records.append((src_file, target_file))

            if not dry_run:
                try:
                    shutil.rmtree(str(item))
                except OSError:
                    pass

    return moved_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate ARGUS AI output layout.")
    parser.add_argument("--dry-run", action="store_true", help="Log movements without executing.")
    parser.add_argument("--outputs-dir", default="outputs", help="Path to outputs directory.")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    print(f"Starting output layout migration (dry_run={args.dry_run}) in {outputs_dir}...")
    records = migrate_outputs(outputs_dir, dry_run=args.dry_run)
    print(f"Migration complete. Total files moved: {len(records)}")


if __name__ == "__main__":
    main()
