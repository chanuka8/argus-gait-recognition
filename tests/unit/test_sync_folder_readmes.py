"""Unit test for package folder README documentation synchronization."""

import unittest
from pathlib import Path

from scripts.sync_folder_readmes import TARGET_FOLDERS, check_folder_readme


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


if __name__ == "__main__":
    unittest.main()
