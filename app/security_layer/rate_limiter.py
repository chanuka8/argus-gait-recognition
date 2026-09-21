"""ARGUS AI — Login Rate Limiting and Brute-Force Protection Module.

Security Architecture:
  - Multi-dimensional tracking:
      1. Per-Account (normalized username) - prevents targeted password brute-forcing
      2. Per-Client-IP - prevents distributed credential stuffing / password spraying
  - In-flight attempt limiting to defeat concurrent race condition attacks
  - Expiring, time-bounded failure history (no permanent lockout)
  - Memory bounds with LRU / TTL eviction to prevent denial-of-service memory exhaustion
  - Secure client IP extraction: ignores unverified X-Forwarded-For headers unless
    explicitly configured behind a trusted reverse proxy
  - Pluggable/injectable monotonic clock for deterministic testing
  - Full thread safety with lightweight synchronization (RLock)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from fastapi import Request

logger = logging.getLogger("ARGUS.RateLimiter")


@dataclass
class BucketRecord:
    """Tracks failed authentication attempts and lock state for an identifier."""

    failure_timestamps: deque[float] = field(default_factory=deque)
    locked_until: float = 0.0
    consecutive_lockouts: int = 0
    in_flight: int = 0
    last_activity: float = 0.0


class LoginRateLimiter:
    """Thread-safe, bounded, multi-dimensional rate limiter for operator login."""

    def __init__(
        self,
        account_max_attempts: int = 5,
        account_window_seconds: float = 300.0,
        account_base_lockout_seconds: float = 60.0,
        account_max_lockout_seconds: float = 300.0,
        ip_max_attempts: int = 20,
        ip_window_seconds: float = 300.0,
        ip_base_lockout_seconds: float = 60.0,
        ip_max_lockout_seconds: float = 300.0,
        max_entries: int = 10000,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.account_max_attempts = account_max_attempts
        self.account_window_seconds = account_window_seconds
        self.account_base_lockout_seconds = account_base_lockout_seconds
        self.account_max_lockout_seconds = account_max_lockout_seconds

        self.ip_max_attempts = ip_max_attempts
        self.ip_window_seconds = ip_window_seconds
        self.ip_base_lockout_seconds = ip_base_lockout_seconds
        self.ip_max_lockout_seconds = ip_max_lockout_seconds

        self.max_entries = max_entries
        self._clock: Callable[[], float] = clock or time.monotonic

        self._lock = threading.RLock()
        self._accounts: dict[str, BucketRecord] = {}
        self._ips: dict[str, BucketRecord] = {}

    def set_clock(self, clock: Callable[[], float]) -> None:
        """Inject a custom clock function (for deterministic testing)."""
        with self._lock:
            self._clock = clock

    def reset_clock(self) -> None:
        """Reset to standard monotonic clock."""
        with self._lock:
            self._clock = time.monotonic

    def now(self) -> float:
        return self._clock()

    def check_rate_limit(self, username: str, client_ip: str) -> tuple[bool, float, str]:
        """Check if a login request is permitted.

        Returns:
            (is_allowed: bool, retry_after: float, reason: str)
        """
        norm_user = username.strip().lower()
        now = self.now()

        with self._lock:
            self._cleanup_locked(now)

            # Check account lockout
            acc_rec = self._accounts.get(norm_user)
            if acc_rec:
                if acc_rec.locked_until > now:
                    retry_after = max(1.0, acc_rec.locked_until - now)
                    return False, retry_after, "account_locked"
                elif acc_rec.locked_until > 0:
                    acc_rec.locked_until = 0.0

                # Check sliding window failures + in_flight
                self._prune_bucket_timestamps(acc_rec, now, self.account_window_seconds)
                active_failures = len(acc_rec.failure_timestamps)
                if (active_failures + acc_rec.in_flight) >= self.account_max_attempts:
                    retry_after = max(1.0, self._compute_lockout(acc_rec.consecutive_lockouts, is_account=True))
                    return False, retry_after, "account_threshold_reached"

            # Check IP lockout
            ip_rec = self._ips.get(client_ip)
            if ip_rec:
                if ip_rec.locked_until > now:
                    retry_after = max(1.0, ip_rec.locked_until - now)
                    return False, retry_after, "ip_locked"
                elif ip_rec.locked_until > 0:
                    ip_rec.locked_until = 0.0

                self._prune_bucket_timestamps(ip_rec, now, self.ip_window_seconds)
                active_ip_failures = len(ip_rec.failure_timestamps)
                if (active_ip_failures + ip_rec.in_flight) >= self.ip_max_attempts:
                    retry_after = max(1.0, self._compute_lockout(ip_rec.consecutive_lockouts, is_account=False))
                    return False, retry_after, "ip_threshold_reached"

            return True, 0.0, "ok"

    def acquire_slot(self, username: str, client_ip: str) -> tuple[bool, float, str]:
        """Atomically check and reserve an in-flight slot.

        Returns:
            (is_allowed: bool, retry_after: float, reason: str)
        """
        norm_user = username.strip().lower()
        now = self.now()

        with self._lock:
            allowed, retry_after, reason = self.check_rate_limit(norm_user, client_ip)
            if not allowed:
                return False, retry_after, reason

            # Reserve in-flight slot
            acc_rec = self._get_or_create_bucket(self._accounts, norm_user, now)
            acc_rec.in_flight += 1
            acc_rec.last_activity = now

            ip_rec = self._get_or_create_bucket(self._ips, client_ip, now)
            ip_rec.in_flight += 1
            ip_rec.last_activity = now

            return True, 0.0, "ok"

    def release_slot(self, username: str, client_ip: str, success: bool) -> None:
        """Release the in-flight slot and record success or failure."""
        norm_user = username.strip().lower()
        now = self.now()

        with self._lock:
            # 1. Update Account Bucket
            acc_rec = self._accounts.get(norm_user)
            if acc_rec:
                acc_rec.in_flight = max(0, acc_rec.in_flight - 1)
                acc_rec.last_activity = now
                if success:
                    # Reset failures and lock on success
                    acc_rec.failure_timestamps.clear()
                    acc_rec.locked_until = 0.0
                    acc_rec.consecutive_lockouts = 0
                else:
                    acc_rec.failure_timestamps.append(now)
                    self._prune_bucket_timestamps(acc_rec, now, self.account_window_seconds)
                    if len(acc_rec.failure_timestamps) >= self.account_max_attempts:
                        lockout_duration = self._compute_lockout(acc_rec.consecutive_lockouts, is_account=True)
                        acc_rec.locked_until = now + lockout_duration
                        acc_rec.consecutive_lockouts += 1
                        acc_rec.failure_timestamps.clear()

            # 2. Update IP Bucket
            ip_rec = self._ips.get(client_ip)
            if ip_rec:
                ip_rec.in_flight = max(0, ip_rec.in_flight - 1)
                ip_rec.last_activity = now
                if success:
                    # On successful login from IP, reduce pressure if not locked
                    pass
                else:
                    ip_rec.failure_timestamps.append(now)
                    self._prune_bucket_timestamps(ip_rec, now, self.ip_window_seconds)
                    if len(ip_rec.failure_timestamps) >= self.ip_max_attempts:
                        lockout_duration = self._compute_lockout(ip_rec.consecutive_lockouts, is_account=False)
                        ip_rec.locked_until = now + lockout_duration
                        ip_rec.consecutive_lockouts += 1
                        ip_rec.failure_timestamps.clear()

    def clear(self) -> None:
        """Reset all rate limiter state (for tests)."""
        with self._lock:
            self._accounts.clear()
            self._ips.clear()

    # --- Internal helpers ---
    def _compute_lockout(self, consecutive_lockouts: int, is_account: bool) -> float:
        base = self.account_base_lockout_seconds if is_account else self.ip_base_lockout_seconds
        maximum = self.account_max_lockout_seconds if is_account else self.ip_max_lockout_seconds
        duration = base * (2 ** min(consecutive_lockouts, 4))
        return min(duration, maximum)

    def _prune_bucket_timestamps(self, bucket: BucketRecord, now: float, window: float) -> None:
        while bucket.failure_timestamps and (now - bucket.failure_timestamps[0]) > window:
            bucket.failure_timestamps.popleft()

    def _get_or_create_bucket(self, store: dict[str, BucketRecord], key: str, now: float) -> BucketRecord:
        rec = store.get(key)
        if rec is None:
            if len(store) >= self.max_entries:
                self._evict_lru(store)
            rec = BucketRecord(last_activity=now)
            store[key] = rec
        return rec

    def _cleanup_locked(self, now: float) -> None:
        self._cleanup_store(self._accounts, now, self.account_window_seconds)
        self._cleanup_store(self._ips, now, self.ip_window_seconds)

    def _cleanup_store(self, store: dict[str, BucketRecord], now: float, window: float) -> None:
        to_delete = []
        for key, rec in store.items():
            if rec.in_flight > 0:
                continue
            if rec.locked_until > now:
                continue
            self._prune_bucket_timestamps(rec, now, window)
            if len(rec.failure_timestamps) == 0 and (now - rec.last_activity) > window:
                to_delete.append(key)
        for key in to_delete:
            del store[key]

    def _evict_lru(self, store: dict[str, BucketRecord]) -> None:
        idle_items = [(k, v.last_activity) for k, v in store.items() if v.in_flight == 0]
        if not idle_items:
            return
        idle_items.sort(key=lambda x: x[1])
        num_to_evict = max(1, len(idle_items) // 10)
        for k, _ in idle_items[:num_to_evict]:
            store.pop(k, None)


# Global singleton instance
_DEFAULT_RATE_LIMITER: LoginRateLimiter | None = None


def get_login_rate_limiter() -> LoginRateLimiter:
    global _DEFAULT_RATE_LIMITER
    if _DEFAULT_RATE_LIMITER is None:
        _DEFAULT_RATE_LIMITER = LoginRateLimiter()
    return _DEFAULT_RATE_LIMITER


def extract_client_ip(request: Request) -> str:
    """Safely extract client IP address without blindly trusting spoofable proxy headers.

    Security rule:
      - By default (local/on-prem), use request.client.host.
      - X-Forwarded-For is ONLY trusted if ARGUS_TRUST_PROXY_HEADERS is explicitly
        enabled ('1', 'true', 'yes') AND request.client.host is in ARGUS_TRUSTED_PROXIES.
    """
    raw_host = request.client.host if request.client else "127.0.0.1"

    trust_headers = os.environ.get("ARGUS_TRUST_PROXY_HEADERS", "").strip().lower() in ("1", "true", "yes")
    if not trust_headers:
        return raw_host

    trusted_proxies_str = os.environ.get("ARGUS_TRUSTED_PROXIES", "127.0.0.1,::1")
    trusted_proxies = {p.strip() for p in trusted_proxies_str.split(",") if p.strip()}

    if raw_host in trusted_proxies:
        xff = request.headers.get("X-Forwarded-For")
        if xff:
            client_ip = xff.split(",")[0].strip()
            if client_ip:
                return client_ip

    return raw_host
