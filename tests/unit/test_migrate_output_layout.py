"""Deterministic unit tests for scripts/migrate_output_layout.py."""

import tempfile
import unittest
from pathlib import Path

from scripts.migrate_output_layout import migrate_outputs


class TestMigrateOutputLayout(unittest.TestCase):
    """Test one-time output layout migration logic."""

    def test_migration_dry_run(self) -> None:
        """Verify dry-run mode logs moves without mutating the filesystem."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir) / "outputs"
            legacy_events = outputs_dir / "events"
            legacy_events.mkdir(parents=True, exist_ok=True)
            dummy_file = legacy_events / "recognition_log.csv"
            dummy_file.write_text("timestamp,camera_id\n", encoding="utf-8")

            moved = migrate_outputs(outputs_dir, dry_run=True)
            self.assertEqual(len(moved), 1)
            self.assertTrue(dummy_file.exists())

    def test_migration_execution_and_conflict_resolution(self) -> None:
        """Verify actual file move, conflict resolution (_1 suffix), and clean removal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir) / "outputs"

            # Create legacy file
            legacy_sec = outputs_dir / "security_logs"
            legacy_sec.mkdir(parents=True, exist_ok=True)
            sec_file = legacy_sec / "security_events.csv"
            sec_file.write_text("event_1\n", encoding="utf-8")

            # Create existing destination file to trigger conflict resolution
            dest_sec = outputs_dir / "logs" / "security"
            dest_sec.mkdir(parents=True, exist_ok=True)
            existing_dest = dest_sec / "security_events.csv"
            existing_dest.write_text("existing_event\n", encoding="utf-8")

            moved = migrate_outputs(outputs_dir, dry_run=False)
            self.assertEqual(len(moved), 1)

            # Verification
            self.assertFalse(sec_file.exists())
            self.assertTrue(existing_dest.exists())
            conflict_file = dest_sec / "security_events_1.csv"
            self.assertTrue(conflict_file.exists())
            self.assertEqual(conflict_file.read_text(encoding="utf-8"), "event_1\n")


if __name__ == "__main__":
    unittest.main()
