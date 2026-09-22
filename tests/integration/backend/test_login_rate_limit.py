"""SEC-06: Login Rate Limiting and Brute-Force Protection Tests.

Verifies:
  A. Normal login flows succeed and create valid sessions.
  B. Failed logins return 401 until threshold is reached, then return 429 with Retry-After.
  C. Recovery after throttle expiration allows legitimate logins again.
  D. Successful login resets failed-attempt counters.
  E. Account isolation: attacking User A does not throttle User B.
  F. IP isolation: abusive IP is throttled without affecting clean IPs.
  G. Spoofing resistance: spoofed X-Forwarded-For headers are ignored by default.
  H. Username enumeration resistance: consistent 429 responses for known and unknown accounts.
  I. Suspended account handling remains secure and throttled appropriately.
  J. Memory bounding and LRU eviction under large volume of distinct identifiers.
  K. Concurrent requests cannot bypass the threshold.
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.server import app
from app.security_layer.auth import get_operator_store, get_session_store
from app.security_layer.password_hasher import get_password_hasher
from app.security_layer.rate_limiter import (
    LoginRateLimiter,
    extract_client_ip,
    get_login_rate_limiter,
)


class MockClock:
    """Controllable monotonic clock for deterministic rate-limit testing."""

    def __init__(self, start_time: float = 1000.0) -> None:
        self.current_time = start_time

    def __call__(self) -> float:
        return self.current_time

    def advance(self, seconds: float) -> None:
        self.current_time += seconds


@pytest.fixture(autouse=True)
def setup_sec06_fixtures(tmp_path: Path):
    """Set up clean operator store, session store, and rate limiter for each test."""
    session_store = get_session_store()
    session_store.clear()

    rate_limiter = get_login_rate_limiter()
    rate_limiter.clear()
    rate_limiter.reset_clock()

    op_store = get_operator_store()
    hasher = get_password_hasher()

    initial_operators = {
        "admins": {
            "sec06_admin": {
                "name": "SEC06 Admin",
                "username": "sec06_admin",
                "password_hash": hasher.hash("AdminSecret@2026!"),
                "role": "admin",
                "status": "Active",
            }
        },
        "investigators": {
            "sec06_alice": {
                "name": "Alice Investigator",
                "username": "sec06_alice",
                "password_hash": hasher.hash("AliceSecret@2026!"),
                "role": "investigator",
                "status": "Active",
            },
            "sec06_bob": {
                "name": "Bob Investigator",
                "username": "sec06_bob",
                "password_hash": hasher.hash("BobSecret@2026!"),
                "role": "investigator",
                "status": "Active",
            },
            "sec06_suspended": {
                "name": "Suspended Operator",
                "username": "sec06_suspended",
                "password_hash": hasher.hash("SuspendedSecret@2026!"),
                "role": "investigator",
                "status": "Suspended",
            },
        },
    }
    op_store._save_offline_store(initial_operators)

    yield

    rate_limiter.clear()
    rate_limiter.reset_clock()
    session_store.clear()


# ==============================================================================
# A. NORMAL LOGIN FLOW
# ==============================================================================


class TestNormalLogin:
    """Verify normal, legitimate logins operate as expected."""

    def test_valid_credentials_succeed(self):
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/auth/login",
                json={"username": "sec06_alice", "password": "AliceSecret@2026!"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert "token" in data
            assert len(data["token"]) >= 32
            assert data["operator"]["username"] == "sec06_alice"
            assert data["operator"]["role"] == "investigator"

    def test_valid_session_accesses_me_profile(self):
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/auth/login",
                json={"username": "sec06_alice", "password": "AliceSecret@2026!"},
            )
            token = resp.json()["token"]

            me_resp = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
            assert me_resp.status_code == 200
            assert me_resp.json()["username"] == "sec06_alice"


# ==============================================================================
# B. FAILED LOGIN & RATE LIMITING (HTTP 429)
# ==============================================================================


class TestFailedLoginRateLimiting:
    """Verify failed attempts trigger rate limiting with HTTP 429."""

    def test_first_failure_returns_401(self):
        with TestClient(app) as client:
            resp = client.post(
                "/api/v1/auth/login",
                json={"username": "sec06_alice", "password": "WrongPassword1!"},
            )
            assert resp.status_code == 401
            assert "Invalid" in resp.json()["detail"]

    def test_repeated_failures_trigger_http_429(self):
        rate_limiter = get_login_rate_limiter()
        mock_clock = MockClock(1000.0)
        rate_limiter.set_clock(mock_clock)

        with TestClient(app) as client:
            # 5 consecutive failed attempts
            for i in range(5):
                resp = client.post(
                    "/api/v1/auth/login",
                    json={"username": "sec06_alice", "password": f"WrongPass_{i}"},
                )
                assert resp.status_code == 401

            # 6th attempt must be throttled with HTTP 429
            resp_throttled = client.post(
                "/api/v1/auth/login",
                json={"username": "sec06_alice", "password": "WrongPass_Again"},
            )
            assert resp_throttled.status_code == 429
            data = resp_throttled.json()
            assert "Too many failed login attempts" in data["detail"]
            assert "Retry-After" in resp_throttled.headers
            retry_after = int(resp_throttled.headers["Retry-After"])
            assert retry_after > 0

    def test_throttled_request_blocks_even_correct_password(self):
        """While locked, an attacker cannot guess the password even with the correct one."""
        rate_limiter = get_login_rate_limiter()
        mock_clock = MockClock(1000.0)
        rate_limiter.set_clock(mock_clock)

        with TestClient(app) as client:
            for _ in range(5):
                client.post(
                    "/api/v1/auth/login",
                    json={"username": "sec06_alice", "password": "BadPassword!"},
                )

            # Attempt with CORRECT password while throttled
            resp = client.post(
                "/api/v1/auth/login",
                json={"username": "sec06_alice", "password": "AliceSecret@2026!"},
            )
            assert resp.status_code == 429


# ==============================================================================
# C. RECOVERY AFTER THROTTLE PERIOD
# ==============================================================================


class TestRecoveryAfterExpiration:
    """Verify that after the lockout duration, logins are permitted again."""

    def test_lockout_expires_and_login_succeeds(self):
        rate_limiter = get_login_rate_limiter()
        mock_clock = MockClock(1000.0)
        rate_limiter.set_clock(mock_clock)

        with TestClient(app) as client:
            # Trigger 5 failures -> locked for 60s
            for _ in range(5):
                client.post(
                    "/api/v1/auth/login",
                    json={"username": "sec06_alice", "password": "BadPassword!"},
                )

            # Confirm 429
            throttled = client.post(
                "/api/v1/auth/login",
                json={"username": "sec06_alice", "password": "BadPassword!"},
            )
            assert throttled.status_code == 429
            retry_after = int(throttled.headers["Retry-After"])

            # Advance clock past the retry period (e.g. 65 seconds)
            mock_clock.advance(retry_after + 5.0)

            # Legitimate login now succeeds
            success_resp = client.post(
                "/api/v1/auth/login",
                json={"username": "sec06_alice", "password": "AliceSecret@2026!"},
            )
            assert success_resp.status_code == 200
            assert "token" in success_resp.json()


# ==============================================================================
# D. SUCCESSFUL LOGIN RESETS FAILED STATE
# ==============================================================================


class TestSuccessResetsState:
    """Verify that a successful login resets the failure counter."""

    def test_success_resets_failed_counter(self):
        rate_limiter = get_login_rate_limiter()
        mock_clock = MockClock(1000.0)
        rate_limiter.set_clock(mock_clock)

        with TestClient(app) as client:
            # 4 failed attempts (1 below threshold)
            for _ in range(4):
                resp = client.post(
                    "/api/v1/auth/login",
                    json={"username": "sec06_alice", "password": "WrongPassword!"},
                )
                assert resp.status_code == 401

            # 5th attempt is successful
            success_resp = client.post(
                "/api/v1/auth/login",
                json={"username": "sec06_alice", "password": "AliceSecret@2026!"},
            )
            assert success_resp.status_code == 200

            # Subsequent failures start from 0 again (should get 401, not 429)
            retry_fail = client.post(
                "/api/v1/auth/login",
                json={"username": "sec06_alice", "password": "WrongPasswordAgain!"},
            )
            assert retry_fail.status_code == 401


# ==============================================================================
# E. ACCOUNT ISOLATION
# ==============================================================================


class TestAccountIsolation:
    """Verify that attacking User A does not lock out User B."""

    def test_locking_user_a_leaves_user_b_unaffected(self):
        rate_limiter = get_login_rate_limiter()
        mock_clock = MockClock(1000.0)
        rate_limiter.set_clock(mock_clock)

        with TestClient(app) as client:
            # Attacker locks sec06_alice with 5 bad attempts
            for _ in range(5):
                client.post(
                    "/api/v1/auth/login",
                    json={"username": "sec06_alice", "password": "BadPassword!"},
                )

            # Verify sec06_alice is locked
            resp_alice = client.post(
                "/api/v1/auth/login",
                json={"username": "sec06_alice", "password": "BadPassword!"},
            )
            assert resp_alice.status_code == 429

            # Legitimate sec06_bob can still log in without hindrance
            resp_bob = client.post(
                "/api/v1/auth/login",
                json={"username": "sec06_bob", "password": "BobSecret@2026!"},
            )
            assert resp_bob.status_code == 200
            assert "token" in resp_bob.json()


# ==============================================================================
# F. IP ISOLATION & SPRAY ATTACK PROTECTION
# ==============================================================================


class TestIPIsolation:
    """Verify IP-level rate limiting prevents credential stuffing across accounts."""

    def test_distributed_user_spray_from_same_ip_is_throttled(self):
        rate_limiter = LoginRateLimiter(
            account_max_attempts=5,
            ip_max_attempts=10,  # lower threshold for test
            ip_base_lockout_seconds=60.0,
        )
        clock = MockClock(1000.0)
        rate_limiter.set_clock(clock)

        ip = "192.168.1.100"

        # Attacker tries 10 different usernames from the same IP (1 attempt each)
        for i in range(10):
            user = f"user_{i}"
            allowed, _, _ = rate_limiter.acquire_slot(user, ip)
            assert allowed is True
            rate_limiter.release_slot(user, ip, success=False)

        # 11th attempt from this IP must be throttled by IP limit
        allowed, retry_after, reason = rate_limiter.acquire_slot("any_user", ip)
        assert allowed is False
        assert reason in ("ip_threshold_reached", "ip_locked")
        assert retry_after > 0

        # Different clean IP is unaffected
        clean_ip = "10.0.0.5"
        clean_allowed, _, _ = rate_limiter.acquire_slot("any_user", clean_ip)
        assert clean_allowed is True


# ==============================================================================
# G. SPOOFING RESISTANCE
# ==============================================================================


class TestSpoofingResistance:
    """Verify untrusted X-Forwarded-For headers cannot be used to bypass rate limits."""

    def test_spoofed_x_forwarded_for_ignored_by_default(self, monkeypatch):
        # Ensure proxy trust is disabled
        monkeypatch.delenv("ARGUS_TRUST_PROXY_HEADERS", raising=False)

        rate_limiter = get_login_rate_limiter()
        mock_clock = MockClock(1000.0)
        rate_limiter.set_clock(mock_clock)

        with TestClient(app) as client:
            # Attacker attempts 5 logins with spoofed X-Forwarded-For headers
            for i in range(5):
                client.post(
                    "/api/v1/auth/login",
                    headers={"X-Forwarded-For": f"10.99.{i}.1"},
                    json={"username": "sec06_alice", "password": "BadPassword!"},
                )

            # 6th attempt with yet another spoofed IP must still be throttled
            resp = client.post(
                "/api/v1/auth/login",
                headers={"X-Forwarded-For": "10.99.99.99"},
                json={"username": "sec06_alice", "password": "BadPassword!"},
            )
            assert resp.status_code == 429

    def test_extract_client_ip_unit(self, monkeypatch):
        class DummyRequest:
            def __init__(self, host: str, headers: dict[str, str] | None = None):
                self.client = type("Client", (), {"host": host})()
                self.headers = headers or {}

        # 1. Default: ignores X-Forwarded-For
        monkeypatch.delenv("ARGUS_TRUST_PROXY_HEADERS", raising=False)
        req = DummyRequest("192.168.1.50", {"X-Forwarded-For": "1.2.3.4"})
        assert extract_client_ip(req) == "192.168.1.50"

        # 2. Enabled but untrusted proxy host: still uses direct client host
        monkeypatch.setenv("ARGUS_TRUST_PROXY_HEADERS", "1")
        monkeypatch.setenv("ARGUS_TRUSTED_PROXIES", "127.0.0.1")
        req = DummyRequest("192.168.1.50", {"X-Forwarded-For": "1.2.3.4"})
        assert extract_client_ip(req) == "192.168.1.50"

        # 3. Enabled and host is trusted proxy: trusts first IP in X-Forwarded-For
        req = DummyRequest("127.0.0.1", {"X-Forwarded-For": "203.0.113.195, 10.0.0.1"})
        assert extract_client_ip(req) == "203.0.113.195"


# ==============================================================================
# H. USERNAME ENUMERATION RESISTANCE
# ==============================================================================


class TestUsernameEnumerationResistance:
    """Verify rate limit responses do not leak whether an account exists."""

    def test_nonexistent_user_throttled_with_identical_generic_429(self):
        rate_limiter = get_login_rate_limiter()
        mock_clock = MockClock(1000.0)
        rate_limiter.set_clock(mock_clock)

        ghost_user = "ghost_nonexistent_operator"

        with TestClient(app) as client:
            for _ in range(5):
                client.post(
                    "/api/v1/auth/login",
                    json={"username": ghost_user, "password": "AnyPassword!"},
                )

            # 6th attempt on nonexistent user returns 429
            resp = client.post(
                "/api/v1/auth/login",
                json={"username": ghost_user, "password": "AnyPassword!"},
            )
            assert resp.status_code == 429
            assert "Too many failed login attempts" in resp.json()["detail"]


# ==============================================================================
# I. SUSPENDED ACCOUNT HANDLING
# ==============================================================================


class TestSuspendedAccount:
    """Verify suspended accounts are properly rejected and rate-limited."""

    def test_suspended_account_fails_and_rate_limits(self):
        rate_limiter = get_login_rate_limiter()
        mock_clock = MockClock(1000.0)
        rate_limiter.set_clock(mock_clock)

        with TestClient(app) as client:
            # 1st attempt returns 401 with suspended message
            resp1 = client.post(
                "/api/v1/auth/login",
                json={"username": "sec06_suspended", "password": "SuspendedSecret@2026!"},
            )
            assert resp1.status_code == 401
            assert "suspended" in resp1.json()["detail"].lower()

            # Repeat 4 more times
            for _ in range(4):
                client.post(
                    "/api/v1/auth/login",
                    json={"username": "sec06_suspended", "password": "SuspendedSecret@2026!"},
                )

            # 6th attempt is throttled with 429
            resp_throttled = client.post(
                "/api/v1/auth/login",
                json={"username": "sec06_suspended", "password": "SuspendedSecret@2026!"},
            )
            assert resp_throttled.status_code == 429


# ==============================================================================
# J. MEMORY BOUNDS & EVICTION
# ==============================================================================


class TestMemoryBoundsAndEviction:
    """Verify tracking structures are bounded and cannot cause memory exhaustion."""

    def test_entries_expire_and_are_cleaned_up(self):
        rate_limiter = LoginRateLimiter(
            account_max_attempts=5,
            account_window_seconds=60.0,
            account_base_lockout_seconds=30.0,
            max_entries=100,
        )
        clock = MockClock(1000.0)
        rate_limiter.set_clock(clock)

        # Record failures for 10 users
        for i in range(10):
            user = f"user_{i}"
            allowed, _, _ = rate_limiter.acquire_slot(user, "127.0.0.1")
            assert allowed is True
            rate_limiter.release_slot(user, "127.0.0.1", success=False)

        assert len(rate_limiter._accounts) == 10

        # Advance clock past idle window
        clock.advance(70.0)

        # Checking another user triggers cleanup
        rate_limiter.check_rate_limit("new_user", "127.0.0.1")
        # All expired entries must have been cleaned up
        assert len(rate_limiter._accounts) <= 1

    def test_lru_eviction_when_max_entries_exceeded(self):
        max_capacity = 20
        rate_limiter = LoginRateLimiter(
            account_max_attempts=5,
            max_entries=max_capacity,
        )
        clock = MockClock(1000.0)
        rate_limiter.set_clock(clock)

        # Add 50 unique users (exceeding max_capacity)
        for i in range(50):
            clock.advance(1.0)
            user = f"burst_user_{i}"
            rate_limiter.acquire_slot(user, "127.0.0.1")
            rate_limiter.release_slot(user, "127.0.0.1", success=False)

        # Store size must never exceed max_entries
        assert len(rate_limiter._accounts) <= max_capacity


# ==============================================================================
# K. CONCURRENCY SAFETY
# ==============================================================================


class TestConcurrency:
    """Verify concurrent requests cannot bypass the rate limit threshold."""

    def test_concurrent_requests_do_not_bypass_threshold(self):
        rate_limiter = LoginRateLimiter(
            account_max_attempts=5,
            account_window_seconds=300.0,
            account_base_lockout_seconds=60.0,
        )
        clock = MockClock(1000.0)
        rate_limiter.set_clock(clock)

        target_user = "concurrency_victim"
        client_ip = "127.0.0.1"

        def attempt_login(attempt_id: int) -> bool:
            allowed, _, _ = rate_limiter.acquire_slot(target_user, client_ip)
            if not allowed:
                return False
            # Simulate slight delay (like hashing)
            rate_limiter.release_slot(target_user, client_ip, success=False)
            return True

        # Launch 20 concurrent login attempts
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(attempt_login, i) for i in range(20)]
            results = [f.result() for f in futures]

        allowed_count = sum(1 for r in results if r is True)
        rejected_count = sum(1 for r in results if r is False)

        # At most 5 attempts should have been allowed
        assert allowed_count <= 5
        assert rejected_count >= 15

    def test_exponential_backoff_escalation(self):
        limiter = LoginRateLimiter(
            account_max_attempts=3,
            account_base_lockout_seconds=60.0,
            account_max_lockout_seconds=300.0,
        )
        clock = MockClock(1000.0)
        limiter.set_clock(clock)

        user = "escalation_test"
        ip = "127.0.0.1"

        # 1st lockout
        for _ in range(3):
            limiter.acquire_slot(user, ip)
            limiter.release_slot(user, ip, success=False)

        _, retry1, _ = limiter.check_rate_limit(user, ip)
        assert retry1 == 60.0

        # Advance past 1st lockout
        clock.advance(61.0)

        # 2nd lockout
        for _ in range(3):
            limiter.acquire_slot(user, ip)
            limiter.release_slot(user, ip, success=False)

        _, retry2, _ = limiter.check_rate_limit(user, ip)
        # Should escalate to 120s
        assert retry2 == 120.0
