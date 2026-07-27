"""Automated README synchronization script for ARGUS AI package folders.

Validates and synchronizes package folder README.md files with the active source modules in each folder.

Usage:
    python scripts/sync_folder_readmes.py         # Automatically update folder READMEs and index
    python scripts/sync_folder_readmes.py --check # Check if folder READMEs are synchronized (CI mode)
    python scripts/sync_folder_readmes.py --update# Explicitly update folder READMEs
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

TARGET_FOLDERS = [
    "api",
    "configs",
    "core",
    "enrollment",
    "evaluation",
    "events",
    "intelligence",
    "models",
    "monitoring",
    "pipeline",
    "preprocessing",
    "security_layer",
    "services",
    "storage",
    "streaming",
    "tests",
    "training",
    "utils",
]

REQUIRED_SECTIONS = [
    "Responsibilities",
    "Key Modules",
    "Data Flow",
    "Configuration",
    "Public Interfaces",
    "Tests",
    "Related Documentation",
]


def get_active_files_for_folder(folder_path: Path) -> List[str]:
    """Get active files/modules that should be documented in Key Modules."""
    folder_name = folder_path.name

    if folder_name == "configs":
        files = [
            f.name
            for f in folder_path.iterdir()
            if f.is_file() and f.suffix in (".yaml", ".yml", ".json")
        ]
    elif folder_name == "models":
        items = ["architectures/bygait_light.py"]
        for sub in [
            "active",
            "appearance_gallery",
            "candidates",
            "gallery",
            "live_gallery",
            "reid",
            "rollback",
            "weights",
        ]:
            if (folder_path / sub).exists():
                items.append(f"{sub}/")
        return sorted(items)
    else:
        items = []
        for f in folder_path.iterdir():
            if f.is_file() and f.suffix == ".py" and not f.name.startswith("__") and f.name != "README.md":
                items.append(f.name)
            elif f.is_dir() and not f.name.startswith("__") and not f.name.startswith("."):
                if folder_name == "api" and f.name == "routes":
                    for sub in f.iterdir():
                        if sub.is_file() and sub.suffix == ".py" and not sub.name.startswith("__"):
                            items.append(f"routes/{sub.name}")
                elif folder_name in ("pipeline", "tests"):
                    items.append(f"{f.name}/")

        files = items

    return sorted(files)


def check_folder_readme(folder_path: Path) -> Tuple[bool, List[str]]:
    """Check if README.md exists, contains required sections, and lists active modules."""
    readme_path = folder_path / "README.md"
    issues = []

    if not readme_path.exists():
        return False, [f"Missing {readme_path}"]

    content = readme_path.read_text(encoding="utf-8")

    # Verify sections
    for sec in REQUIRED_SECTIONS:
        if f"## {sec}" not in content:
            issues.append(f"Missing section '## {sec}' in {readme_path.name}")

    # Check key modules presence
    active_files = get_active_files_for_folder(folder_path)
    for file_name in active_files:
        base_name = file_name.rstrip("/")
        if base_name not in content:
            issues.append(f"Module/File '{file_name}' not listed in {readme_path.name}")

    return len(issues) == 0, issues


def check_readme_index(root_dir: Path) -> Tuple[bool, List[str]]:
    """Check if docs/README_INDEX.md exists and contains links to all package READMEs."""
    index_path = root_dir / "docs" / "README_INDEX.md"
    issues = []

    if not index_path.exists():
        return False, [f"Missing {index_path}"]

    content = index_path.read_text(encoding="utf-8")
    for folder in TARGET_FOLDERS:
        if f"{folder}/README.md" not in content:
            issues.append(f"Missing entry for {folder}/README.md in docs/README_INDEX.md")

    return len(issues) == 0, issues


def update_folder_readme(folder_path: Path) -> bool:
    """Synchronize Key Modules section in folder README if comment markers are present."""
    readme_path = folder_path / "README.md"
    if not readme_path.exists():
        return False

    content = readme_path.read_text(encoding="utf-8")
    marker_start = "<!-- BEGIN SYNC: KEY_MODULES -->"
    marker_end = "<!-- END SYNC: KEY_MODULES -->"

    if marker_start not in content or marker_end not in content:
        return False

    active_files = get_active_files_for_folder(folder_path)
    
    # Extract current descriptions dictionary if existing
    desc_map: Dict[str, str] = {}
    table_lines = content.split(marker_start)[1].split(marker_end)[0].strip().split("\n")
    for line in table_lines:
        if line.startswith("|") and not line.startswith("| Module") and not line.startswith("|---"):
            cols = [c.strip() for c in line.split("|")[1:-1]]
            if len(cols) >= 2:
                mod_link = cols[0]
                desc = cols[1]
                m = re.search(r"\[(.*?)\]", mod_link)
                key = m.group(1) if m else mod_link
                desc_map[key] = desc

    # Rebuild table
    new_lines = ["| Module | Purpose |", "|---|---|"]
    for f in active_files:
        desc = desc_map.get(f, f"Module/resource file {f}")
        if "/" in f and not f.endswith("/"):
            link = f"[{f}]({f})"
        elif f.endswith("/"):
            link = f"`{f}`"
        else:
            link = f"[{f}]({f})"
        new_lines.append(f"| {link} | {desc} |")

    new_table_str = "\n".join(new_lines)
    pattern = re.escape(marker_start) + r".*?" + re.escape(marker_end)
    replacement = f"{marker_start}\n{new_table_str}\n{marker_end}"

    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    if new_content != content:
        readme_path.write_text(new_content, encoding="utf-8")
        return True

    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Synchronize ARGUS AI folder README files.")
    parser.add_argument("--check", action="store_true", help="Check if folder READMEs are synchronized (CI mode).")
    parser.add_argument("--update", action="store_true", help="Update folder README key modules tables.")
    parser.add_argument("--root-dir", default=".", help="Root workspace directory.")
    args = parser.parse_args()

    # Default execution without flags triggers update mode automatically
    should_update = args.update or not args.check
    root_dir = Path(args.root_dir)
    total_issues = 0
    updated_count = 0

    print("Checking ARGUS AI folder documentation alignment...")

    for folder_name in TARGET_FOLDERS:
        folder_path = root_dir / folder_name
        if not folder_path.exists():
            continue

        if should_update:
            if update_folder_readme(folder_path):
                print(f"[UPDATED] Synchronized {folder_name}/README.md")
                updated_count += 1

        is_valid, issues = check_folder_readme(folder_path)
        if not is_valid:
            total_issues += len(issues)
            for issue in issues:
                print(f"[WARN] {issue}")
        else:
            print(f"[OK] {folder_name}/README.md is synchronized and valid.")

    idx_valid, idx_issues = check_readme_index(root_dir)
    if not idx_valid:
        total_issues += len(idx_issues)
        for issue in idx_issues:
            print(f"[WARN] {issue}")
    else:
        print("[OK] docs/README_INDEX.md is valid.")

    if args.check and total_issues > 0:
        print(f"\n[FAIL] Synchronization check failed with {total_issues} issue(s).")
        return 1

    print(f"\n[SUCCESS] All package READMEs synchronized cleanly ({updated_count} updated).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
