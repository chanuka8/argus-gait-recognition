import hmac

try:
    import argon2
    from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

    _HASHER = argon2.PasswordHasher(
        time_cost=3,
        memory_cost=65536,
        parallelism=4,
        hash_len=32,
        type=argon2.Type.ID,
    )
except ImportError:
    _HASHER = None
    InvalidHashError = ValueError
    VerificationError = ValueError
    VerifyMismatchError = ValueError


class PasswordHasher:
    """Argon2id password hashing and verification with transparent legacy migration support.

    Safety:
      - Never exposes raw passwords in exceptions or logs.
      - Uses constant-time comparison for legacy plaintext strings.
      - Detects when an existing hash requires parameter upgrades or when legacy plaintext
        needs re-hashing with Argon2id.
    """

    def __init__(self) -> None:
        if _HASHER is None:
            raise RuntimeError(
                "argon2-cffi is not installed or failed to initialize. "
                "Ensure argon2-cffi is installed in the active environment."
            )
        self._hasher = _HASHER

    def hash(self, password: str) -> str:
        """Hash a plaintext password using Argon2id."""
        if not password or not isinstance(password, str):
            raise ValueError("Password must be a non-empty string")
        return self._hasher.hash(password)

    def verify(self, password: str, stored_credential: str) -> tuple[bool, bool]:
        """Verify a password against a stored credential (Argon2id hash or legacy plaintext).

        Returns:
            (is_valid, needs_rehash)
            - is_valid: True if password matches the stored credential.
            - needs_rehash: True if the credential was verified via legacy plaintext or
              Argon2id parameters require updating, meaning the caller should persist
              a freshly computed Argon2id hash.
        """
        if not password or not stored_credential:
            return False, False

        # 1. Check if stored credential is an Argon2 hash
        if stored_credential.startswith("$argon2"):
            try:
                self._hasher.verify(stored_credential, password)
                needs_rehash = self._hasher.check_needs_rehash(stored_credential)
                return True, needs_rehash
            except (VerifyMismatchError, VerificationError, InvalidHashError):
                return False, False

        # 2. Legacy plaintext verification (Fail-safe migration path)
        # Uses constant-time HMAC comparison to prevent timing attacks.
        try:
            password_bytes = password.encode("utf-8")
            stored_bytes = stored_credential.encode("utf-8")
            is_match = hmac.compare_digest(password_bytes, stored_bytes)
            if is_match:
                # Verified against legacy plaintext; caller MUST re-hash with Argon2id.
                return True, True
            return False, False
        except Exception:  # noqa: BLE001
            return False, False

    def is_argon2_hash(self, value: str) -> bool:
        """Check whether a string format matches an Argon2 hash."""
        return isinstance(value, str) and value.startswith("$argon2")


# Global singleton instance
_DEFAULT_HASHER: PasswordHasher | None = None


def get_password_hasher() -> PasswordHasher:
    global _DEFAULT_HASHER
    if _DEFAULT_HASHER is None:
        _DEFAULT_HASHER = PasswordHasher()
    return _DEFAULT_HASHER
