from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def sample_gei_path():
    return (
        PROJECT_ROOT
        / "data"
        / "casia_processed"
        / "gei"
        / "034"
        / "034_nm-01_126.png"
    )


@pytest.fixture
def enrollment_sample_folder():
    path = (
        PROJECT_ROOT
        / "data"
        / "new_input"
        / "api_test_person"
    )
    if not path.exists():
        disabled_path = (
            PROJECT_ROOT
            / "data"
            / "new_input"
            / "_disabled_api_test_person"
        )
        if disabled_path.exists():
            return disabled_path
    return path


@pytest.fixture
def security_log_path():
    return (
        PROJECT_ROOT
        / "outputs"
        / "security_logs"
        / "security_events.csv"
    )


@pytest.fixture
def benchmark_report_path():
    return (
        PROJECT_ROOT
        / "outputs"
        / "reports"
        / "benchmark_report.json"
    )