import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def sample_gei_path():
    return PROJECT_ROOT / "data" / "casia_processed" / "gei" / "034" / "034_nm-01_126.png"


@pytest.fixture
def enrollment_sample_folder():
    path = PROJECT_ROOT / "data" / "new_input" / "api_test_person"
    if not path.exists():
        disabled_path = PROJECT_ROOT / "data" / "new_input" / "_disabled_api_test_person"
        if disabled_path.exists():
            return disabled_path
    return path


@pytest.fixture
def security_log_path():
    return PROJECT_ROOT / "outputs" / "logs" / "security" / "security_events.csv"


@pytest.fixture
def benchmark_report_path():
    return PROJECT_ROOT / "outputs" / "reports" / "benchmark" / "benchmark_report.json"


@pytest.fixture(autouse=True)
def setup_test_auth_headers(request, monkeypatch):
    """Provide valid admin session token to non-security test clients so regression tests pass."""
    fspath = str(getattr(request, "fspath", "")).replace("\\", "/")
    security_test_modules = {
        "test_auth_bypass.py",
        "test_camera_auth_flow.py",
        "test_rbac_and_bola.py",
        "test_admin_bootstrap.py",
        "test_password_migration.py",
        "test_firebase_account_connectivity.py",
        "test_investigator_camera_flow.py",
        "test_async_missing_person.py",
    }
    if any(sec in fspath for sec in security_test_modules) or "tests/security" in fspath:
        return

    from fastapi.testclient import TestClient

    from security_layer.auth import get_session_store

    session = get_session_store().create_session(
        operator_id="test_admin_auto",
        username="test_admin_auto",
        role="admin",
    )
    admin_auth = f"Bearer {session.token}"

    original_request = TestClient.request

    def authenticated_request(self, method, url, *args, **kwargs):
        headers = kwargs.get("headers")
        if headers is None:
            kwargs["headers"] = {"Authorization": admin_auth}
        elif isinstance(headers, dict) and "Authorization" not in headers:
            kwargs["headers"] = {"Authorization": admin_auth, **headers}
        return original_request(self, method, url, *args, **kwargs)

    monkeypatch.setattr(TestClient, "request", authenticated_request)


@pytest.fixture(autouse=True)
def isolate_operator_storage(tmp_path, monkeypatch):
    """Hermetically isolate operator store in all tests to prevent mutating production/offline disk files."""
    monkeypatch.setenv("ARGUS_OPERATOR_STORE_MODE", "offline")
    from security_layer.auth import get_operator_store

    op_store = get_operator_store()
    orig_path = op_store.offline_store_path
    isolated_file = tmp_path / "test_operator_store.json"
    op_store.offline_store_path = isolated_file
    try:
        yield isolated_file
    finally:
        op_store.offline_store_path = orig_path
