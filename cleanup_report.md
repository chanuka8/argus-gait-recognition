# Cleanup and Quarantine Audit Report

This report outlines the files and directories identified for cleanup quarantine, those preserved as risky or essential, and the estimated space savings.

## Safe Quarantine Candidates

These candidates are confirmed safe to move to quarantine (`_cleanup_quarantine/`) as they are caches, temporary files, or empty unused files.

| File / Folder Path | Type | Reason | Size (Est.) |
| :--- | :--- | :--- | :--- |
| `.pytest_cache/` | Cache Folder | Internal pytest cache; generated automatically. | ~100 KB |
| `.qodo/` | Cache Folder | IDE helper cache files; generated automatically. | ~10 KB |
| `__pycache__/` (multiple folders) | Cache Folders | Python compiled bytecode (`.pyc`) caches; safe to recreate on execution. | ~1.5 MB |
| `pipeline/steps/silhouette_alignment.py` | Empty File (0B) | Unused, contains no code, and is not referenced in the project imports. | 0 bytes |

## Risky Candidates (Preserved)

These files/folders were audited but preserved in their original locations to ensure system integrity, Python module resolution, and version control structures.

| File / Folder Path | Type | Reason | Status |
| :--- | :--- | :--- | :--- |
| All package `__init__.py` files | Init Files | Required by Python to resolve packages and discover submodules. | **PRESERVED** |
| All `.gitkeep` files | Placeholder Files | Required by Git to track empty output or config directories. | **PRESERVED** |
| `outputs/reports/argus.log` | Log File | Contains active telemetry and system boot logs. | **PRESERVED** |

## Keep List

All project runtime, configuration, test, and documentation files are preserved:
- `cli.py`, `main.py`, `Makefile`, `README.md`, `VERSION`
- `requirements.txt`, `requirements-dev.txt`
- `configs/` directory and all contents
- `pipeline/`, `training/`, `evaluation/`, `preprocessing/`, `storage/`, `security_layer/`, `streaming/`, `utils/` source directories
- `tests/` and `scripts/` directories
- `docs/` folder contents
- Model assets and datasets inside `models/`, `runs/`, and `data/` directories

---

## Space Audit Summary

- **Total items moved to quarantine:** 2 cache directories, 23 compiled bytecode directories, and 1 empty source file.
- **Estimated disk space saved/quarantined:** **~1.6 MB**
