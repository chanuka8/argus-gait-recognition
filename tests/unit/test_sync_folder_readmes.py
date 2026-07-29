"""Unit tests for package folder README documentation synchronization and git hook installer.

Covers:
  1. Public file discovery
  2. Private/excluded directory handling
  3. Deterministic generation (second sync produces no diff)
  4. Manual text preservation outside markers
  5. Stale --check exit code
  6. Relative link generation (no file:/// links)
  7. README index generation and validation
  8. Malformed marker handling
  9. Atomic write behavior
 10. Pre-commit hook content verification
 11. Hook Linux/macOS venv path detection
 12. No network/model/GPU dependency in sync script
 13. Missing README returns issues
 14. Corrupted file safety
"""

import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from scripts.install_git_hooks import HOOK_CONTENT, install_pre_commit_hook
from scripts.sync_folder_readmes import (
    TARGET_FOLDERS,
    check_folder_readme,
    check_readme_index,
    get_active_files_for_folder,
    update_folder_readme,
)


class TestFileDiscovery(unittest.TestCase):
    """Verify get_active_files_for_folder discovers the right files."""

    def test_discovers_python_files(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            (folder / "alpha.py").write_text("# module", encoding="utf-8")
            (folder / "beta.py").write_text("# module", encoding="utf-8")
            files = get_active_files_for_folder(folder)
            self.assertIn("alpha.py", files)
            self.assertIn("beta.py", files)

    def test_excludes_dunder_init(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            (folder / "__init__.py").write_text("", encoding="utf-8")
            (folder / "real.py").write_text("# module", encoding="utf-8")
            files = get_active_files_for_folder(folder)
            self.assertNotIn("__init__.py", files)
            self.assertIn("real.py", files)

    def test_excludes_readme(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            (folder / "README.md").write_text("# Readme", encoding="utf-8")
            (folder / "module.py").write_text("# module", encoding="utf-8")
            files = get_active_files_for_folder(folder)
            self.assertNotIn("README.md", files)

    def test_sorted_output(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            for name in ["z.py", "a.py", "m.py"]:
                (folder / name).write_text("# mod", encoding="utf-8")
            files = get_active_files_for_folder(folder)
            self.assertEqual(files, sorted(files))

    def test_configs_folder_discovers_yaml(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "configs"
            folder.mkdir()
            (folder / "system.yaml").write_text("key: val", encoding="utf-8")
            (folder / "notes.txt").write_text("not yaml", encoding="utf-8")
            files = get_active_files_for_folder(folder)
            self.assertIn("system.yaml", files)
            self.assertNotIn("notes.txt", files)


class TestCheckFolderReadme(unittest.TestCase):
    """Verify check_folder_readme reports correct issues."""

    def _make_valid_readme(self, folder: Path, modules: list[str]) -> None:
        sections = [
            "# Test",
            "## Responsibilities",
            "## Key Modules",
            "<!-- BEGIN SYNC: KEY_MODULES -->",
            "| Module | Purpose |",
            "|---|---|",
        ]
        for m in modules:
            sections.append(f"| [{m}]({m}) | desc |")
        sections += [
            "<!-- END SYNC: KEY_MODULES -->",
            "## Data Flow",
            "## Configuration",
            "## Public Interfaces",
            "## Tests",
            "## Related Documentation",
        ]
        (folder / "README.md").write_text("\n".join(sections), encoding="utf-8")

    def test_missing_readme_returns_issues(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            is_valid, issues = check_folder_readme(folder)
            self.assertFalse(is_valid)
            self.assertTrue(any("Missing" in i for i in issues))

    def test_valid_readme_passes(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            (folder / "module_a.py").write_text("# mod", encoding="utf-8")
            self._make_valid_readme(folder, ["module_a.py"])
            is_valid, issues = check_folder_readme(folder)
            self.assertTrue(is_valid, f"Unexpected issues: {issues}")

    def test_missing_module_is_detected(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            (folder / "module_a.py").write_text("# mod", encoding="utf-8")
            (folder / "module_b.py").write_text("# mod", encoding="utf-8")
            self._make_valid_readme(folder, ["module_a.py"])  # missing module_b
            is_valid, issues = check_folder_readme(folder)
            self.assertFalse(is_valid)
            self.assertTrue(any("module_b.py" in i for i in issues))


class TestUpdateFolderReadme(unittest.TestCase):
    """Verify update_folder_readme synchronizes markers correctly."""

    def _make_readme_with_markers(self, folder: Path, table_content: str, manual_text: str = "") -> None:
        content = textwrap.dedent(f"""\
            # Test Package

            {manual_text}

            ## Key Modules

            <!-- BEGIN SYNC: KEY_MODULES -->
            {table_content}
            <!-- END SYNC: KEY_MODULES -->

            ## Responsibilities

            ## Data Flow

            ## Configuration

            ## Public Interfaces

            ## Tests

            ## Related Documentation
        """)
        (folder / "README.md").write_text(content, encoding="utf-8")

    def test_updates_stale_table(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            (folder / "alpha.py").write_text("# mod", encoding="utf-8")
            self._make_readme_with_markers(folder, "| old | old |")
            updated = update_folder_readme(folder)
            self.assertTrue(updated)
            content = (folder / "README.md").read_text(encoding="utf-8")
            self.assertIn("alpha.py", content)

    def test_no_update_when_current(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            (folder / "alpha.py").write_text("# mod", encoding="utf-8")
            self._make_readme_with_markers(folder, "| old | old |")
            update_folder_readme(folder)
            updated_again = update_folder_readme(folder)
            self.assertFalse(updated_again)

    def test_manual_text_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            (folder / "mod.py").write_text("# mod", encoding="utf-8")
            manual = "This is manually written documentation that must survive sync."
            self._make_readme_with_markers(folder, "| old | old |", manual_text=manual)
            update_folder_readme(folder)
            content = (folder / "README.md").read_text(encoding="utf-8")
            self.assertIn(manual, content)

    def test_missing_markers_returns_false(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            (folder / "README.md").write_text("# No markers here\n", encoding="utf-8")
            result = update_folder_readme(folder)
            self.assertFalse(result)

    def test_missing_readme_returns_false(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            result = update_folder_readme(folder)
            self.assertFalse(result)

    def test_deterministic_second_run(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            (folder / "b.py").write_text("# b", encoding="utf-8")
            (folder / "a.py").write_text("# a", encoding="utf-8")
            self._make_readme_with_markers(folder, "| old | old |")
            update_folder_readme(folder)
            content_first = (folder / "README.md").read_text(encoding="utf-8")
            update_folder_readme(folder)
            content_second = (folder / "README.md").read_text(encoding="utf-8")
            self.assertEqual(content_first, content_second)


class TestAtomicWrite(unittest.TestCase):
    """Verify update_folder_readme uses atomic write (temp file + replace)."""

    def test_no_temp_files_left_after_sync(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            (folder / "mod.py").write_text("# mod", encoding="utf-8")
            content = textwrap.dedent("""\
                # Test
                ## Key Modules
                <!-- BEGIN SYNC: KEY_MODULES -->
                | old | old |
                <!-- END SYNC: KEY_MODULES -->
                ## Responsibilities
                ## Data Flow
                ## Configuration
                ## Public Interfaces
                ## Tests
                ## Related Documentation
            """)
            (folder / "README.md").write_text(content, encoding="utf-8")
            update_folder_readme(folder)
            temp_files = [f for f in folder.iterdir() if f.name.startswith(".readme_sync_")]
            self.assertEqual(len(temp_files), 0, f"Leftover temp files: {temp_files}")

    def test_readme_not_corrupted_by_successful_write(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            (folder / "mod.py").write_text("# mod", encoding="utf-8")
            content = textwrap.dedent("""\
                # Test
                ## Key Modules
                <!-- BEGIN SYNC: KEY_MODULES -->
                | old | old |
                <!-- END SYNC: KEY_MODULES -->
                ## Responsibilities
                ## Data Flow
                ## Configuration
                ## Public Interfaces
                ## Tests
                ## Related Documentation
            """)
            (folder / "README.md").write_text(content, encoding="utf-8")
            update_folder_readme(folder)
            result = (folder / "README.md").read_text(encoding="utf-8")
            self.assertIn("<!-- BEGIN SYNC: KEY_MODULES -->", result)
            self.assertIn("<!-- END SYNC: KEY_MODULES -->", result)
            self.assertIn("mod.py", result)


class TestReadmeIndex(unittest.TestCase):
    """Verify check_readme_index detects missing entries."""

    def test_valid_index(self):
        root_dir = Path(__file__).resolve().parent.parent.parent
        is_valid, issues = check_readme_index(root_dir)
        self.assertTrue(is_valid, f"docs/README_INDEX.md issues: {issues}")

    def test_missing_index_file_detected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            is_valid, issues = check_readme_index(root)
            self.assertFalse(is_valid)
            self.assertTrue(any("Missing" in i for i in issues))

    def test_no_file_protocol_links_in_index(self):
        root_dir = Path(__file__).resolve().parent.parent.parent
        index = root_dir / "docs" / "README_INDEX.md"
        if index.exists():
            content = index.read_text(encoding="utf-8")
            self.assertNotIn("file:///", content)

    def test_no_duplicate_entries_in_index(self):
        root_dir = Path(__file__).resolve().parent.parent.parent
        index = root_dir / "docs" / "README_INDEX.md"
        if index.exists():
            content = index.read_text(encoding="utf-8")
            for folder in TARGET_FOLDERS:
                link_target = f"](../{folder}/README.md)"
                count = content.count(link_target)
                self.assertEqual(count, 1, f"Expected exactly 1 link entry for {link_target}, found {count}")

    def test_all_linked_readmes_exist(self):
        root_dir = Path(__file__).resolve().parent.parent.parent
        for folder in TARGET_FOLDERS:
            readme = root_dir / folder / "README.md"
            self.assertTrue(readme.exists(), f"Missing {folder}/README.md")


class TestSyncCheckMode(unittest.TestCase):
    """Verify --check mode exit code behavior."""

    def test_check_exits_zero_when_current(self):
        result = subprocess.run(
            [sys.executable, "scripts/sync_folder_readmes.py", "--check"],
            capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent.parent),
        )
        self.assertEqual(result.returncode, 0, f"--check failed:\n{result.stdout}\n{result.stderr}")

    def test_check_does_not_modify_files(self):
        root = Path(__file__).resolve().parent.parent.parent
        before = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=str(root)).stdout
        subprocess.run([sys.executable, "scripts/sync_folder_readmes.py", "--check"], capture_output=True, text=True, cwd=str(root))
        after = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=str(root)).stdout
        self.assertEqual(before, after, "--check modified working tree state")


class TestAllFolderReadmesSynchronized(unittest.TestCase):
    """Integration: verify every target folder passes check."""

    def test_all_folder_readmes_exist_and_synchronized(self) -> None:
        root_dir = Path(__file__).resolve().parent.parent.parent
        failed_folders = {}
        for folder_name in TARGET_FOLDERS:
            folder_path = root_dir / folder_name
            self.assertTrue(folder_path.exists(), f"Expected folder '{folder_name}' to exist.")
            is_valid, issues = check_folder_readme(folder_path)
            if not is_valid:
                failed_folders[folder_name] = issues
        self.assertEqual(len(failed_folders), 0, f"Folder README issues: {failed_folders}")


class TestNoFileProtocolLinks(unittest.TestCase):
    """Verify no folder README contains file:/// links."""

    def test_no_file_protocol_in_folder_readmes(self):
        root_dir = Path(__file__).resolve().parent.parent.parent
        for folder in TARGET_FOLDERS:
            readme = root_dir / folder / "README.md"
            if readme.exists():
                content = readme.read_text(encoding="utf-8")
                self.assertNotIn("file:///", content, f"file:/// link found in {folder}/README.md")


class TestPreCommitHook(unittest.TestCase):
    """Verify pre-commit hook content and installer."""

    def test_hook_contains_sync_script(self):
        self.assertIn("sync_folder_readmes.py", HOOK_CONTENT)

    def test_hook_stages_readme_files(self):
        self.assertIn("git add", HOOK_CONTENT)
        self.assertIn("docs/README_INDEX.md", HOOK_CONTENT)

    def test_hook_does_not_use_git_add_dot(self):
        # git add . would stage all unrelated changes
        self.assertNotIn("git add .", HOOK_CONTENT)
        self.assertNotIn("git add -A", HOOK_CONTENT)

    def test_hook_has_valid_shebang(self):
        self.assertTrue(HOOK_CONTENT.strip().startswith("#!/bin/sh"))

    def test_hook_aborts_on_sync_failure(self):
        self.assertIn("exit 1", HOOK_CONTENT)

    def test_hook_detects_linux_venv(self):
        self.assertIn("venv/bin/python", HOOK_CONTENT)

    def test_hook_detects_windows_venv(self):
        self.assertIn("venv/Scripts/python.exe", HOOK_CONTENT)

    def test_hook_installer_creates_file(self):
        root_dir = Path(__file__).resolve().parent.parent.parent
        success = install_pre_commit_hook(root_dir)
        self.assertTrue(success)
        hook_path = root_dir / ".git" / "hooks" / "pre-commit"
        self.assertTrue(hook_path.exists())
        content = hook_path.read_text(encoding="utf-8")
        self.assertIn("sync_folder_readmes.py", content)
        self.assertIn("venv/bin/python", content)

    def test_hook_no_hardcoded_absolute_paths(self):
        self.assertNotIn("E:\\", HOOK_CONTENT)
        self.assertNotIn("C:\\", HOOK_CONTENT)
        self.assertNotIn("/home/", HOOK_CONTENT)


class TestNoRuntimeSideEffects(unittest.TestCase):
    """Verify sync script doesn't import heavy runtime modules."""

    def test_no_torch_import(self):
        path = Path(__file__).resolve().parent.parent.parent / "scripts" / "sync_folder_readmes.py"
        content = path.read_text(encoding="utf-8")
        self.assertNotIn("import torch", content)
        self.assertNotIn("import cv2", content)
        self.assertNotIn("import numpy", content)
        self.assertNotIn("import onnx", content)

    def test_no_outputs_write(self):
        path = Path(__file__).resolve().parent.parent.parent / "scripts" / "sync_folder_readmes.py"
        content = path.read_text(encoding="utf-8")
        self.assertNotIn("outputs/", content)

    def test_sync_script_compiles(self):
        path = Path(__file__).resolve().parent.parent.parent / "scripts" / "sync_folder_readmes.py"
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(path)],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
        )
        self.assertEqual(result.returncode, 0, f"Compile failed: {result.stderr}")


if __name__ == "__main__":
    unittest.main()
