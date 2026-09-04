import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import numpy as np

from deployment.backend_summary import BackendStartupSummary
from deployment.build_metadata import extract_build_metadata
from deployment.runtime_manifest import get_runtime_manifest
from deployment.shutdown_manager import ShutdownManager
from deployment.startup_validator import DeploymentStartupValidator
from storage.vector_store import validate_gallery_files


def run_deployment_smoke_test(
    output_dir: str = "outputs/reports",
) -> tuple[int, dict]:
    report = {
        "status": "FAILED",
        "exit_code": 2,
        "checks": {},
        "defects": [],
        "build_metadata": {},
    }

    try:
        manifest = get_runtime_manifest()
        m_val = manifest.validate_runtime_assets()
        report["checks"]["runtime_manifest"] = "PASSED" if m_val["valid"] else "FAILED"
        if not m_val["valid"]:
            report["defects"].append(f"Missing runtime manifest assets: {m_val['missing']}")

        validator = DeploymentStartupValidator()
        s_res = validator.validate_startup(raise_on_failure=False)
        report["checks"]["startup_validator"] = "PASSED" if s_res["success"] else "FAILED"
        if not s_res["success"]:
            for defect in s_res["blocking_issues"]:
                report["defects"].append(f"Startup validator issue: {defect}")

        backend = s_res.get("backend")
        if backend is None:
            report["checks"]["backend_initialization"] = "FAILED"
            report["defects"].append("Inference backend failed to initialize")
        else:
            report["checks"]["backend_initialization"] = "PASSED"
            b_meta = getattr(backend, "metadata", {})
            report["checks"]["backend_metadata"] = "PASSED" if (isinstance(b_meta, dict) and b_meta) else "WARNING"

            summary_obj = BackendStartupSummary(
                backend=backend,
                startup_status=s_res.get("status", "READY_FOR_CONTROLLED_GAIT_RECOGNITION_TESTING"),
            )
            summary_obj.emit(print_cli=True)

        if backend is not None:
            try:
                dummy_gei = np.zeros((1, 1, 128, 64), dtype=np.float32)
                embedding = backend.predict(dummy_gei)

                is_shape_ok = isinstance(embedding, np.ndarray) and embedding.shape == (1, 256)
                is_finite = np.isfinite(embedding).all()
                norm = float(np.linalg.norm(embedding))
                is_norm_ok = np.isclose(norm, 1.0, atol=1e-4)

                if is_shape_ok and is_finite and is_norm_ok:
                    report["checks"]["synthetic_inference"] = "PASSED"
                else:
                    report["checks"]["synthetic_inference"] = "FAILED"
                    report["defects"].append(
                        f"Synthetic inference assertion failed (shape={getattr(embedding, 'shape', None)}, norm={norm})"
                    )
            except (RuntimeError, ValueError, TypeError, OSError) as e:
                report["checks"]["synthetic_inference"] = "FAILED"
                report["defects"].append(f"Synthetic inference exception: {e}")

        g_valid, g_err, _g_count = validate_gallery_files(gallery_dir=Path("models/gallery"), expected_dim=256)
        report["checks"]["gallery_validation"] = (
            "PASSED" if g_valid or "files missing" in (g_err or "").lower() else "FAILED"
        )
        if not g_valid and "files missing" not in (g_err or "").lower():
            report["defects"].append(f"Gallery validation defect: {g_err}")

        build_meta = extract_build_metadata(backend=backend)
        report["build_metadata"] = build_meta.to_dict()

        sm = ShutdownManager()
        sd_ok = sm.shutdown()
        report["checks"]["graceful_shutdown"] = "PASSED" if sd_ok else "FAILED"
        if not sd_ok:
            report["defects"].append("Shutdown manager returned failure during smoke test")

        if len(report["defects"]) == 0:
            report["status"] = "PASSED"
            report["exit_code"] = 0
            exit_code = 0
        else:
            report["status"] = "FAILED"
            report["exit_code"] = 1
            exit_code = 1

    except (RuntimeError, ValueError, TypeError, OSError) as e:
        report["status"] = "INTERNAL_FAILURE"
        report["exit_code"] = 2
        report["defects"].append(f"Unhandled smoke test exception: {e}")
        exit_code = 2

    try:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        json_file = out_p / "deployment_smoke_test.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        md_file = out_p / "deployment_smoke_test.md"
        md_lines = [
            "# ARGUS AI Deployment Smoke Test Report",
            "",
            f"**Status:** {report['status']}",
            f"**Exit Code:** {report['exit_code']}",
            "",
            "## Verification Checks",
            "",
        ]
        for check, res in report["checks"].items():
            md_lines.append(f"- **{check}**: `{res}`")

        if report["defects"]:
            md_lines.extend(["", "## Confirmed Defects", ""])
            for d in report["defects"]:
                md_lines.append(f"- :x: {d}")
        else:
            md_lines.extend(["", ":white_check_mark: Zero deployment defects identified."])

        md_file.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        report["checks"]["report_generation"] = "PASSED"
    except (OSError, ValueError) as e:
        report["defects"].append(f"Failed writing smoke test report: {e}")
        if exit_code == 0:
            exit_code = 1
            report["exit_code"] = 1

    return exit_code, report


def main() -> None:
    code, rep = run_deployment_smoke_test()
    print(f"\n[DEPLOYMENT SMOKE TEST] Completed with status={rep['status']} (Exit Code {code})")
    sys.exit(code)


if __name__ == "__main__":
    main()
