"""
Deployment Runtime Manifest and Packaging Separator for ARGUS AI.

Defines the structure for distinguishing build-time assets vs runtime-only assets,
ensuring Windows-native deployment packages contain all required runtime modules
while excluding tests, development tools, caches, and sensitive credentials.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

# Authoritative lists of repository-relative asset paths

BUILD_TIME_ASSETS = [
    "tests",
    "training",
    "evaluation",
    "automation",
    "dataconnect",
    "scripts/export_bygait_onnx.py",
    "scripts/benchmark_inference_backends.py",
    "scripts/sync_folder_readmes.py",
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
    "core/system.py",
    "core/boot.py",
    "core/orchestrator.py",
    "models/architectures/bygait_light.py",
    "models/inference/backend.py",
    "models/inference/pytorch_backend.py",
    "pipeline/live_recognition.py",
    "pipeline/video_recognition.py",
    "pipeline/folder_recognition.py",
    "storage/vector_store.py",
    "monitoring/logging_config.py",
    "deployment/startup_validator.py",
    "deployment/backend_summary.py",
    "deployment/build_metadata.py",
    "deployment/shutdown_manager.py",
    "configs/system.yaml",
    "configs/inference.yaml",
    "configs/cameras.yaml",
    "scripts/doctor.py",
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


@dataclass
class RuntimeManifest:
    """Structured container for runtime vs build deployment manifest."""

    application_name: str = "ARGUS AI"
    runtime_assets: list = field(default_factory=lambda: list(RUNTIME_ONLY_ASSETS))
    build_assets: list = field(default_factory=lambda: list(BUILD_TIME_ASSETS))
    excluded_patterns: list = field(default_factory=lambda: list(EXCLUDED_PATTERNS))

    def validate_runtime_assets(self, repo_root: str = ".") -> dict:
        """
        Validate presence of essential runtime assets against the filesystem.

        Returns a dictionary containing 'valid' (bool), 'missing' (list), and 'checked' (list).
        """
        root = Path(repo_root).resolve()
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
        """Export manifest data as a dictionary with repository-relative paths only."""
        return asdict(self)

    def export_json(self, output_path: str = "deployment/runtime_manifest.json") -> Path:
        """Write manifest JSON artifact with repository-relative paths."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()
        # Verify no absolute paths or credentials
        raw_json = json.dumps(data, indent=4)
        if "C:\\Users" in raw_json or "/home/" in raw_json:
            raise ValueError("Absolute user-home path detected in runtime manifest export")

        with open(path, "w", encoding="utf-8") as f:
            f.write(raw_json + "\n")

        return path

    def export_markdown(self, output_path: str = "deployment/runtime_manifest.md") -> Path:
        """Write human-readable runtime manifest markdown documentation."""
        path = Path(output_path)
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

        content.extend([
            "",
            "## Build-Only Assets (Excluded from Production Runtime)",
            "",
        ])
        for asset in self.build_assets:
            content.append(f"- `{asset}`")

        content.extend([
            "",
            "## Excluded Security & Development Patterns",
            "",
        ])
        for pattern in self.excluded_patterns:
            content.append(f"- `{pattern}`")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(content) + "\n")

        return path


def get_runtime_manifest() -> RuntimeManifest:
    """Factory helper to return default RuntimeManifest."""
    return RuntimeManifest()


def generate_runtime_manifest_artifacts(
    json_path: str = "deployment/runtime_manifest.json",
    md_path: str = "deployment/runtime_manifest.md",
) -> dict:
    """Generate both JSON and Markdown runtime manifest files."""
    manifest = get_runtime_manifest()
    jp = manifest.export_json(output_path=json_path)
    mp = manifest.export_markdown(output_path=md_path)
    val = manifest.validate_runtime_assets()

    return {
        "json_path": str(jp),
        "markdown_path": str(mp),
        "validation": val,
    }
