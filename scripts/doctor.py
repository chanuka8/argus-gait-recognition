import importlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage.vector_store import validate_gallery_files
from utils.config_validator import ConfigValidator, sanitize_rtsp_url

STATUS_READY = "READY_FOR_CONTROLLED_GAIT_RECOGNITION_TESTING"
STATUS_WARNINGS = "READY_WITH_WARNINGS"
STATUS_NOT_READY = "NOT_READY"
STATUS_UNABLE = "UNABLE_TO_VERIFY"


def _to_rel(path: str | Path) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def run_doctor(
    json_path: str = "outputs/reports/health_report.json",
    md_path: str = "outputs/reports/health_report.md",
) -> tuple[int, dict]:
    try:
        return _execute_doctor_checks(json_path=json_path, md_path=md_path)
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        err_msg = sanitize_rtsp_url(str(e))
        report_data = {
            "overall_status": STATUS_UNABLE,
            "exit_code": 2,
            "blocking_issues": [f"Doctor internal checker error: {err_msg}"],
            "warnings": [],
            "checks": [],
        }
        return 2, report_data


def _execute_doctor_checks(json_path: str, md_path: str) -> tuple[int, dict]:
    checks: list[dict] = []
    blocking_issues: list[str] = []
    warnings: list[str] = []

    py_ver = sys.version.split()[0]
    py_ok = sys.version_info >= (3, 9)
    checks.append(
        {
            "name": "python_version",
            "category": "python",
            "status": "PASS" if py_ok else "FAIL",
            "details": f"Python {py_ver} ({_to_rel(sys.executable)})",
        }
    )
    if not py_ok:
        blocking_issues.append(f"Python version {py_ver} is unsupported (minimum 3.9 required)")

    required_packages = ["torch", "onnx", "onnxruntime", "cv2", "numpy", "yaml"]
    for pkg in required_packages:
        try:
            importlib.import_module(pkg)
            checks.append(
                {
                    "name": f"import_{pkg}",
                    "category": "dependencies",
                    "status": "PASS",
                    "details": f"Package '{pkg}' import succeeded",
                }
            )
        except ImportError as e:
            status = "FAIL" if pkg in {"torch", "numpy", "yaml"} else "WARN"
            checks.append(
                {
                    "name": f"import_{pkg}",
                    "category": "dependencies",
                    "status": status,
                    "details": f"Package '{pkg}' not available: {e}",
                }
            )
            if status == "FAIL":
                blocking_issues.append(f"Required package '{pkg}' missing")
            else:
                warnings.append(f"Optional package '{pkg}' missing")

    ckpt_path = ROOT / "runs" / "exp_001" / "best_model.pth"
    onnx_path = ROOT / "models" / "engines" / "bygait_light.onnx"

    ckpt_exists = ckpt_path.exists()
    onnx_exists = onnx_path.exists()

    checks.append(
        {
            "name": "pytorch_checkpoint_exists",
            "category": "model",
            "status": "PASS" if ckpt_exists else "WARN",
            "details": f"PyTorch checkpoint at '{_to_rel(ckpt_path)}' ("
            + ("found" if ckpt_exists else "missing")
            + ")",
        }
    )
    if not ckpt_exists:
        warnings.append(f"PyTorch checkpoint missing at {_to_rel(ckpt_path)}")

    checks.append(
        {
            "name": "onnx_model_exists",
            "category": "model",
            "status": "PASS" if onnx_exists else "WARN",
            "details": f"ONNX model at '{_to_rel(onnx_path)}' (" + ("found" if onnx_exists else "missing") + ")",
        }
    )

    if onnx_exists:
        try:
            import onnx

            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            checks.append(
                {
                    "name": "onnx_model_integrity",
                    "category": "model",
                    "status": "PASS",
                    "details": "ONNX model loaded and verified structurally",
                }
            )
        except (RuntimeError, ValueError, TypeError, OSError) as e:
            checks.append(
                {
                    "name": "onnx_model_integrity",
                    "category": "model",
                    "status": "FAIL",
                    "details": f"ONNX model corruption: {e}",
                }
            )
            blocking_issues.append(f"Corrupted ONNX model file at {_to_rel(onnx_path)}")

    try:
        from models.inference.backend import BackendValidator, get_inference_backend

        backend_inst = get_inference_backend()
        validator = BackendValidator(backend_inst.config)
        smoke_ok = validator.run_smoke_test(backend_inst)

        checks.append(
            {
                "name": "backend_initialization",
                "category": "backend",
                "status": "PASS" if smoke_ok else "FAIL",
                "details": f"Active backend '{backend_inst.active_backend}' (Provider: {backend_inst.execution_provider}, Smoke test: {'PASSED' if smoke_ok else 'FAILED'})",
            }
        )
        if not smoke_ok:
            blocking_issues.append(f"Active inference backend '{backend_inst.active_backend}' failed smoke test")
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        checks.append(
            {
                "name": "backend_initialization",
                "category": "backend",
                "status": "FAIL",
                "details": f"Backend initialization error: {e}",
            }
        )
        blocking_issues.append(f"Inference backend failed to initialize: {e}")

    gallery_dir = ROOT / "models" / "gallery"
    g_valid, g_err, g_count = validate_gallery_files(gallery_dir=gallery_dir, expected_dim=256)
    if g_valid:
        checks.append(
            {
                "name": "gallery_integrity",
                "category": "gallery",
                "status": "PASS",
                "details": f"Gallery features and labels ({g_count} identities) loaded safely with allow_pickle=False",
            }
        )
    elif "files missing" in (g_err or "").lower():
        checks.append(
            {
                "name": "gallery_integrity",
                "category": "gallery",
                "status": "WARN",
                "details": f"Gallery state notice: {g_err}",
            }
        )
        warnings.append(f"Gallery state notice: {g_err}")
    else:
        checks.append(
            {
                "name": "gallery_integrity",
                "category": "gallery",
                "status": "FAIL",
                "details": f"Gallery defect: {g_err}",
            }
        )
        blocking_issues.append(f"Gallery defect: {g_err}")

    cfg_validator = ConfigValidator(configs_dir=ROOT / "configs")
    cfg_results = cfg_validator.validate_all()
    all_cfg_ok = True
    for cfg_name, cfg_errs in cfg_results.items():
        if cfg_errs:
            all_cfg_ok = False
            for err in cfg_errs:
                sanitized_err = sanitize_rtsp_url(err)
                blocking_issues.append(f"Config error in {cfg_name}: {sanitized_err}")

    checks.append(
        {
            "name": "configuration_files",
            "category": "config",
            "status": "PASS" if all_cfg_ok else "FAIL",
            "details": "All YAML configurations loaded and validated cleanly"
            if all_cfg_ok
            else "Configuration validation errors detected",
        }
    )

    out_dir = ROOT / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_file = out_dir / ".doctor_probe.tmp"
    writable = False
    try:
        probe_file.write_text("write_probe", encoding="utf-8")
        writable = probe_file.exists()
        if probe_file.exists():
            probe_file.unlink()
    except OSError:
        writable = False

    checks.append(
        {
            "name": "storage_writability",
            "category": "storage",
            "status": "PASS" if writable else "FAIL",
            "details": f"Output report directory '{_to_rel(out_dir)}' writable: {writable}",
        }
    )
    if not writable:
        blocking_issues.append(f"Output directory {_to_rel(out_dir)} is not writable")

    _total, _used, free = shutil.disk_usage(ROOT)
    free_mb = free / (1024 * 1024)
    disk_ok = free_mb > 500
    checks.append(
        {
            "name": "disk_space",
            "category": "storage",
            "status": "PASS" if disk_ok else "WARN",
            "details": f"Available disk space: {free_mb:.1f} MB (minimum 500 MB recommended)",
        }
    )
    if not disk_ok:
        warnings.append(f"Low disk space: {free_mb:.1f} MB remaining")

    try:
        from monitoring.logging_config import get_logger

        logger = get_logger("doctor")
        logger.info("Doctor health check diagnostic log test.")
        checks.append(
            {
                "name": "logging_initialization",
                "category": "logging",
                "status": "PASS",
                "details": "Logging system initialized successfully",
            }
        )
    except (RuntimeError, ValueError, TypeError, OSError) as e:
        checks.append(
            {
                "name": "logging_initialization",
                "category": "logging",
                "status": "WARN",
                "details": f"Logging initialization warning: {e}",
            }
        )
        warnings.append(f"Logging initialization issue: {e}")

    if blocking_issues:
        overall_status = STATUS_NOT_READY
        exit_code = 1
    elif warnings:
        overall_status = STATUS_WARNINGS
        exit_code = 0
    else:
        overall_status = STATUS_READY
        exit_code = 0

    report_data = {
        "overall_status": overall_status,
        "exit_code": exit_code,
        "blocking_issues": blocking_issues,
        "warnings": warnings,
        "checks": checks,
    }

    j_path = ROOT / json_path
    j_path.parent.mkdir(parents=True, exist_ok=True)
    with open(j_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)

    m_path = ROOT / md_path
    m_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for c in checks:
        rows.append(f"| `{c['name']}` | `{c['category']}` | **{c['status']}** | {sanitize_rtsp_url(c['details'])} |")

    md_content = (
        f"""# Doctor System Health Report

## Executive Summary

- **Overall Status**: `{overall_status}`
- **Exit Code**: `{exit_code}`
- **Blocking Issues Count**: `{len(blocking_issues)}`
- **Warnings Count**: `{len(warnings)}`

## Diagnostic Checks

| Check Name | Category | Status | Details |
| :--- | :--- | :--- | :--- |
"""
        + "\n".join(rows)
        + "\n"
    )

    if blocking_issues:
        md_content += (
            "\n## Confirmed Blocking Defects\n\n"
            + "\n".join(f"- {sanitize_rtsp_url(b)}" for b in blocking_issues)
            + "\n"
        )
    if warnings:
        md_content += "\n## Non-Blocking Warnings\n\n" + "\n".join(f"- {sanitize_rtsp_url(w)}" for w in warnings) + "\n"

    with open(m_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return exit_code, report_data


def main() -> None:
    exit_code, report = run_doctor()
    print(f"[DOCTOR] Health check finished. Status: {report['overall_status']}, Exit Code: {exit_code}")
    if report["blocking_issues"]:
        print(f"[DOCTOR] Blocking issues found: {len(report['blocking_issues'])}")
        for b in report["blocking_issues"]:
            print(f"  - [DEFECT] {sanitize_rtsp_url(b)}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
