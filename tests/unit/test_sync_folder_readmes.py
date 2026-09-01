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

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.install_git_hooks import HOOK_CONTENT, install_pre_commit_hook
from scripts.sync_folder_readmes import (
    TARGET_FOLDERS,
    _atomic_write_file,
    _get_script_category,
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
            self._make_valid_readme(folder, ["module_a.py"])
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
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
            check=False,
        )
        self.assertEqual(result.returncode, 0, f"--check failed:\n{result.stdout}\n{result.stderr}")

    def test_check_does_not_modify_files(self):
        root = Path(__file__).resolve().parent.parent.parent
        before = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=str(root), check=False
        ).stdout
        subprocess.run(
            [sys.executable, "scripts/sync_folder_readmes.py", "--check"],
            capture_output=True,
            text=True,
            cwd=str(root),
            check=False,
        )
        after = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=str(root), check=False
        ).stdout
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
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
            check=False,
        )
        self.assertEqual(result.returncode, 0, f"Compile failed: {result.stderr}")


class TestLineEndingPreservation(unittest.TestCase):
    """Verify update_folder_readme preserves original line endings (CRLF vs LF)."""

    def _make_readme(self, folder: Path, newline: str) -> None:
        lines = [
            "# Test Package",
            "## Key Modules",
            "<!-- BEGIN SYNC: KEY_MODULES -->",
            "| old | old |",
            "<!-- END SYNC: KEY_MODULES -->",
            "## Responsibilities",
            "## Data Flow",
            "## Configuration",
            "## Public Interfaces",
            "## Tests",
            "## Related Documentation",
        ]
        text = newline.join(lines) + newline
        (folder / "README.md").write_bytes(text.encode("utf-8"))

    def test_preserves_crlf_line_endings(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            (folder / "mod_a.py").write_text("# mod a", encoding="utf-8")
            self._make_readme(folder, "\r\n")

            updated = update_folder_readme(folder)
            self.assertTrue(updated)

            raw = (folder / "README.md").read_bytes()
            self.assertIn(b"\r\n", raw)
            self.assertIn(b"mod_a.py", raw)

    def test_preserves_lf_line_endings(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            (folder / "mod_b.py").write_text("# mod b", encoding="utf-8")
            self._make_readme(folder, "\n")

            updated = update_folder_readme(folder)
            self.assertTrue(updated)

            raw = (folder / "README.md").read_bytes()
            self.assertNotIn(b"\r\n", raw)
            self.assertIn(b"mod_b.py", raw)


class TestWindowsRetryAndLockHandling(unittest.TestCase):
    """Verify atomic write retry logic and temporary file cleanup under transient lock conditions."""

    def test_recovers_from_transient_permission_error(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "README.md"
            target.write_text("# Initial", encoding="utf-8")

            real_replace = os.replace
            call_count = 0

            def flaky_replace(src, dst):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise PermissionError(13, "Permission denied (simulated Windows lock)")
                return real_replace(src, dst)

            with patch("os.replace", side_effect=flaky_replace):
                _atomic_write_file(target, "# Updated Content", max_retries=5)

            self.assertEqual(call_count, 3)
            self.assertEqual(target.read_text(encoding="utf-8"), "# Updated Content")

            temp_files = [f for f in Path(td).iterdir() if f.name.startswith(".readme_sync_")]
            self.assertEqual(len(temp_files), 0, f"Leftover temp files: {temp_files}")

    def test_temp_file_cleaned_up_on_permanent_failure(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "README.md"
            target.write_text("# Initial", encoding="utf-8")

            def perm_denied(src, dst):
                raise PermissionError(13, "Simulated permanent lock")

            with patch("os.replace", side_effect=perm_denied), self.assertRaises(OSError):
                _atomic_write_file(target, "# Broken", max_retries=3)

            temp_files = [f for f in Path(td).iterdir() if f.name.startswith(".readme_sync_")]
            self.assertEqual(len(temp_files), 0, f"Temp files left after failure: {temp_files}")


class TestScriptCategorySynchronization(unittest.TestCase):
    """Verify category synchronization accurately classifies renamed and active scripts."""

    def test_validation_and_dataset_script_categories(self):
        self.assertEqual(_get_script_category("demo_confidence_scorer.py"), "Validation")
        self.assertEqual(_get_script_category("demo_gei.py"), "Validation")
        self.assertEqual(_get_script_category("run_tracking.py"), "Validation")
        self.assertEqual(_get_script_category("generate_visualizer_charts.py"), "Validation")
        self.assertEqual(_get_script_category("build_gallery.py"), "Dataset")
        self.assertEqual(_get_script_category("export_bygait_onnx.py"), "Conversion")
        self.assertEqual(_get_script_category("activate_venv.ps1"), "Environment")
        self.assertEqual(_get_script_category("sync_folder_readmes.py"), "Documentation")


class TestDocsCheckImmutabilityAndSafety(unittest.TestCase):
    """Verify cli docs-check is strictly read-only and leaves the repository byte-for-byte unchanged."""

    def test_docs_check_leaves_docs_tree_and_readmes_byte_for_byte_identical(self):
        import hashlib

        root_dir = Path(__file__).resolve().parent.parent.parent

        def snapshot_files():
            snapshot = {}

            docs_dir = root_dir / "docs"
            if docs_dir.exists():
                for p in docs_dir.rglob("*.md"):
                    if p.is_file():
                        rel = str(p.relative_to(root_dir))
                        snapshot[rel] = hashlib.sha256(p.read_bytes()).hexdigest()


            root_readme = root_dir / "README.md"
            if root_readme.is_file():
                snapshot["README.md"] = hashlib.sha256(root_readme.read_bytes()).hexdigest()

            for folder in TARGET_FOLDERS:
                f_readme = root_dir / folder / "README.md"
                if f_readme.is_file():
                    rel = str(f_readme.relative_to(root_dir))
                    snapshot[rel] = hashlib.sha256(f_readme.read_bytes()).hexdigest()

            return snapshot

        before_snapshot = snapshot_files()

        result = subprocess.run(
            [sys.executable, "cli.py", "--mode", "docs-check"],
            capture_output=True,
            text=True,
            cwd=str(root_dir),
            check=False,
        )
        self.assertEqual(result.returncode, 0, f"docs-check failed:\n{result.stdout}\n{result.stderr}")

        after_snapshot = snapshot_files()

        self.assertEqual(
            set(before_snapshot.keys()),
            set(after_snapshot.keys()),
            "docs-check added or removed documentation files!",
        )

        for rel_path, before_hash in before_snapshot.items():
            after_hash = after_snapshot[rel_path]
            self.assertEqual(
                before_hash,
                after_hash,
                f"docs-check modified documentation file: {rel_path}",
            )

    def test_docs_check_does_not_unlink_custom_documentation_files(self):
        root_dir = Path(__file__).resolve().parent.parent.parent
        custom_file = root_dir / "docs" / "temp_audit_regression_protection_test.md"
        custom_file.write_text("# Custom Protected Document\nContent that must not be unlinked.\n", encoding="utf-8")

        try:
            result = subprocess.run(
                [sys.executable, "cli.py", "--mode", "docs-check"],
                capture_output=True,
                text=True,
                cwd=str(root_dir),
                check=False,
            )
            self.assertEqual(result.returncode, 0, f"docs-check failed:\n{result.stdout}\n{result.stderr}")
            self.assertTrue(custom_file.exists(), "docs-check unlinked custom markdown documentation file!")
            self.assertEqual(
                custom_file.read_text(encoding="utf-8"),
                "# Custom Protected Document\nContent that must not be unlinked.\n",
            )
        finally:
            if custom_file.exists():
                try:
                    custom_file.unlink()
                except OSError:
                    pass


class TestSyncIdempotency(unittest.TestCase):
    """Verify repeated update and check executions are strictly idempotent."""

    def test_repeated_update_produces_zero_further_diff(self):
        root_dir = Path(__file__).resolve().parent.parent.parent


        res1 = subprocess.run(
            [sys.executable, "scripts/sync_folder_readmes.py", "--update"],
            capture_output=True,
            text=True,
            cwd=str(root_dir),
            check=False,
        )
        self.assertEqual(res1.returncode, 0)


        res2 = subprocess.run(
            [sys.executable, "scripts/sync_folder_readmes.py", "--update"],
            capture_output=True,
            text=True,
            cwd=str(root_dir),
            check=False,
        )
        self.assertEqual(res2.returncode, 0)
        self.assertIn("0 updated", res2.stdout)


        res3 = subprocess.run(
            [sys.executable, "scripts/sync_folder_readmes.py", "--check"],
            capture_output=True,
            text=True,
            cwd=str(root_dir),
            check=False,
        )
        self.assertEqual(res3.returncode, 0)
        self.assertIn("All package READMEs synchronized cleanly", res3.stdout)


if __name__ == "__main__":
    unittest.main()


