"""
Unit tests for DeploymentStartupValidator.
"""

from deployment.startup_validator import DeploymentStartupValidator, StartupValidationError
import pytest


def test_startup_validator_success():
    validator = DeploymentStartupValidator()
    summary = validator.validate_startup(raise_on_failure=False)

    assert summary["success"] is True
    assert len(summary["blocking_issues"]) == 0


def test_startup_validator_raises_on_failure(monkeypatch: pytest.MonkeyPatch):
    validator = DeploymentStartupValidator()

    def fake_validate_all():
        return {"inference.yaml": ["Simulated invalid config parameter"]}

    monkeypatch.setattr(validator.config_validator, "validate_all", fake_validate_all)

    with pytest.raises(StartupValidationError, match="Simulated invalid config parameter"):
        validator.validate_startup(raise_on_failure=True)
