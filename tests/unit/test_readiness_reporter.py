from pathlib import Path

from deployment.readiness_reporter import ALLOWED_OVERALL_STATUSES, DeploymentReadinessReporter


def test_readiness_reporter_generation(tmp_path: Path):
    reporter = DeploymentReadinessReporter()

    json_path = tmp_path / "deployment_readiness.json"
    md_path = tmp_path / "deployment_readiness.md"

    report_data = reporter.generate_reports(
        json_path=str(json_path),
        md_path=str(md_path),
    )

    assert json_path.exists()
    assert md_path.exists()

    assert report_data.get("schema_version") == "1.0.0"
    assert "timestamp" in report_data
    overall = report_data.get("overall_status")
    assert overall in ALLOWED_OVERALL_STATUSES
    assert "python_readiness" in report_data
    assert "pytorch_readiness" in report_data
    assert "onnx_readiness" in report_data
    assert "backend_readiness" in report_data
    assert "metadata" in report_data["backend_readiness"]
    assert "exact_verification_evidence" in report_data
