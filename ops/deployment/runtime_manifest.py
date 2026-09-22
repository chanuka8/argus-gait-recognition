import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

BUILD_TIME_ASSETS = [
    "tests",
    "ml_platform/training",
    "ml_platform/evaluation",
    "ops/automation",
    "dataconnect",
    "ops/tools/maintenance/export_bygait_onnx.py",
    "ops/tools/benchmark/inference_backends.py",
    "ops/tools/maintenance/sync_folder_readmes.py",
    "requirements.txt",
    "ruff.toml",
    "pytest.ini",
    "Makefile",
    "docs",
    ".github",
    ".qodo",
    ".agents",
]

RUNTIME_ONLY_ASSETS = [
    "main.py",
    "cli.py",
    "VERSION",
    "app/core/system.py",
    "app/core/boot.py",
    "app/core/orchestrator.py",
    "ml_platform/models/architectures/bygait_light.py",
    "ml_platform/models/inference/backend.py",
    "ml_platform/models/inference/pytorch_backend.py",
    "app/pipeline/live_recognition.py",
    "app/pipeline/video_recognition.py",
    "app/pipeline/folder_recognition.py",
    "app/storage/vector_store.py",
    "app/monitoring/logging_config.py",
    "ops/deployment/startup_validator.py",
    "ops/deployment/backend_summary.py",
    "ops/deployment/build_metadata.py",
    "ops/deployment/shutdown_manager.py",
    "configs/system.yaml",
    "configs/inference.yaml",
    "configs/cameras.yaml",
    "ops/tools/validation/doctor.py",
]

EXCLUDED_PATTERNS = [
    "venv",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".vscode",
    ".env",
    "secrets",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "outputs/reports/*.json",
    "outputs/reports/*.md",
]


from app.core.paths import get_app_root, resolve_runtime_path


@dataclass
class RuntimeManifest:
    application_name: str = "ARGUS AI"
    runtime_assets: list = field(default_factory=lambda: list(RUNTIME_ONLY_ASSETS))
    build_assets: list = field(default_factory=lambda: list(BUILD_TIME_ASSETS))
    excluded_patterns: list = field(default_factory=lambda: list(EXCLUDED_PATTERNS))

    def validate_runtime_assets(self, repo_root: str | Path | None = None) -> dict:
        root = get_app_root(repo_root) if repo_root is None else Path(repo_root).resolve()
        checked = []
        missing = []

        for asset_rel in self.runtime_assets:
            checked.append(asset_rel)
            asset_path = root / asset_rel
            if not asset_path.exists():
                missing.append(asset_rel)

        return {
            "valid": len(missing) == 0,
            "missing": missing,
            "checked": checked,
        }

    def to_dict(self) -> dict:
        return asdict(self)

    def export_json(self, output_path: str | Path = "ops/deployment/runtime_manifest.json") -> Path:
        path = resolve_runtime_path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()
        raw_json = json.dumps(data, indent=4)
        if "C:\\Users" in raw_json or "/home/" in raw_json:
            raise ValueError("Absolute user-home path detected in runtime manifest export")

        with open(path, "w", encoding="utf-8") as f:
            f.write(raw_json + "\n")

        return path

    def export_markdown(self, output_path: str | Path = "ops/deployment/runtime_manifest.md") -> Path:
        path = resolve_runtime_path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        content = [
            "# ARGUS AI Runtime Manifest",
            "",
            "This manifest defines the build vs runtime asset separation for native deployment packaging.",
            "",
            "## Runtime Required Assets",
            "",
        ]
        for asset in self.runtime_assets:
            content.append(f"- `{asset}`")

        content.extend(
            [
                "",
                "## Build-Only Assets (Excluded from Production Runtime)",
                "",
            ]
        )
        for asset in self.build_assets:
            content.append(f"- `{asset}`")

        content.extend(
            [
                "",
                "## Excluded Security & Development Patterns",
                "",
            ]
        )
        for pattern in self.excluded_patterns:
            content.append(f"- `{pattern}`")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(content) + "\n")

        return path


def get_runtime_manifest() -> RuntimeManifest:
    return RuntimeManifest()


def generate_runtime_manifest_artifacts(
    json_path: str | Path = "ops/deployment/runtime_manifest.json",
    md_path: str | Path = "ops/deployment/runtime_manifest.md",
) -> dict:
    manifest = get_runtime_manifest()
    jp = manifest.export_json(output_path=json_path)
    mp = manifest.export_markdown(output_path=md_path)
    val = manifest.validate_runtime_assets()

    return {
        "json_path": str(jp),
        "markdown_path": str(mp),
        "validation": val,
    }
