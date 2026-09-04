# Tools

The `tools` package provides operational CLI utilities, benchmarks, maintenance scripts, database migrations, and validation harnesses for ARGUS AI.

## Responsibilities

- Providing deployment smoke testing, environment checks, and diagnostic tools (`tools/validation/`).
- Benchmarking inference backends, frame rates, and latency under load (`tools/benchmark/`).
- Managing biometric gallery lifecycle, cleanup, and git hook installations (`tools/maintenance/`).
- Running password encryption and output structure database migrations (`tools/migration/`).
- Preprocessing dataset skeletons and managing pretrained weight assets (`tools/data/`).
- Boundaries: Tools are operator-facing and developer utilities; production core runtime does not import from `tools`.

## Key Modules

<!-- BEGIN SYNC: KEY_MODULES -->
| Module | Purpose |
| --- | --- |
| [activate_venv.ps1](activate_venv.ps1) | Module/resource file activate_venv.ps1 |
| `benchmark/` | Module/resource file benchmark/ |
| [bootstrap_env.ps1](bootstrap_env.ps1) | Module/resource file bootstrap_env.ps1 |
| `data/` | Module/resource file data/ |
| [dev.js](dev.js) | Module/resource file dev.js |
| `maintenance/` | Module/resource file maintenance/ |
| [manage_venv.ps1](manage_venv.ps1) | Module/resource file manage_venv.ps1 |
| `migration/` | Module/resource file migration/ |
| [start_system.bat](start_system.bat) | Module/resource file start_system.bat |
| [start_system.sh](start_system.sh) | Module/resource file start_system.sh |
| `validation/` | Module/resource file validation/ |
<!-- END SYNC: KEY_MODULES -->

## Data Flow

Operator / CLI Invocation → `tools/<category>/<tool>.py` → Target System Inspection / Orchestration → Output Diagnostics / Reports.

## Configuration

- Runtime YAML configurations in `configs/system.yaml` and `configs/inference.yaml`.

## Public Interfaces

- Run diagnostic health check: `python tools/validation/doctor.py`
- Run deployment smoke test: `python tools/validation/deployment_smoke_test.py`
- Run inference backend benchmark: `python tools/benchmark/inference_backends.py`
- Run gallery maintenance: `python tools/maintenance/build_gallery.py`

## Tests

- Tested via `tests/unit/backend/` and `tests/performance/inference/`.

## Related Documentation

- [Root README](../README.md)
- [Deployment Documentation](../docs/README_INDEX.md)
