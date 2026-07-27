"""Unit tests for package folder README documentation synchronization and git hook installer."""

import unittest
from pathlib import Path

from scripts.install_git_hooks import install_pre_commit_hook
from scripts.sync_folder_readmes import TARGET_FOLDERS, check_folder_readme, check_readme_index


class TestFolderReadmeSync(unittest.TestCase):
    """Verify that all major package folders contain compliant and synchronized README.md files."""

    def test_all_folder_readmes_exist_and_synchronized(self) -> None:
        root_dir = Path(__file__).resolve().parent.parent.parent
        failed_folders = {}

        for folder_name in TARGET_FOLDERS:
            folder_path = root_dir / folder_name
            self.assertTrue(
                folder_path.exists(),
                f"Expected target package folder '{folder_name}' to exist.",
            )
            is_valid, issues = check_folder_readme(folder_path)
            if not is_valid:
                failed_folders[folder_name] = issues

        self.assertEqual(
            len(failed_folders),
            0,
            f"Folder README synchronization failed for: {failed_folders}",
        )

    def test_readme_index_validity(self) -> None:
        root_dir = Path(__file__).resolve().parent.parent.parent
        is_valid, issues = check_readme_index(root_dir)
        self.assertTrue(is_valid, f"docs/README_INDEX.md issues: {issues}")

    def test_pre_commit_hook_installer(self) -> None:
        root_dir = Path(__file__).resolve().parent.parent.parent
        success = install_pre_commit_hook(root_dir)
        self.assertTrue(success)
        hook_path = root_dir / ".git" / "hooks" / "pre-commit"
        self.assertTrue(hook_path.exists())
        content = hook_path.read_text(encoding="utf-8")
        self.assertIn("sync_folder_readmes.py", content)
        self.assertIn("git add docs/README_INDEX.md", content)


if __name__ == "__main__":
    unittest.main()
