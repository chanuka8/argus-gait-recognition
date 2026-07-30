# ARGUS AI Runtime Manifest

This manifest defines the build vs runtime asset separation for native deployment packaging.

## Runtime Required Assets

- `main.py`
- `cli.py`
- `VERSION`
- `core/system.py`
- `core/boot.py`
- `core/orchestrator.py`
- `models/architectures/bygait_light.py`
- `models/inference/backend.py`
- `models/inference/pytorch_backend.py`
- `pipeline/live_recognition.py`
- `pipeline/video_recognition.py`
- `pipeline/folder_recognition.py`
- `storage/vector_store.py`
- `monitoring/logging_config.py`
- `deployment/startup_validator.py`
- `deployment/backend_summary.py`
- `deployment/build_metadata.py`
- `deployment/shutdown_manager.py`
- `configs/system.yaml`
- `configs/inference.yaml`
- `configs/cameras.yaml`
- `scripts/doctor.py`

## Build-Only Assets (Excluded from Production Runtime)

- `tests`
- `training`
- `evaluation`
- `automation`
- `dataconnect`
- `scripts/export_bygait_onnx.py`
- `scripts/benchmark_inference_backends.py`
- `scripts/sync_folder_readmes.py`
- `requirements.txt`
- `ruff.toml`
- `pytest.ini`
- `Makefile`
- `docs`
- `.github`
- `.qodo`
- `.agents`

## Excluded Security & Development Patterns

- `venv`
- `.git`
- `.pytest_cache`
- `.ruff_cache`
- `.vscode`
- `.env`
- `secrets`
- `__pycache__`
- `*.pyc`
- `*.pyo`
- `outputs/reports/*.json`
- `outputs/reports/*.md`
