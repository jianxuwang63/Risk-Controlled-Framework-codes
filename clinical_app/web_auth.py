from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections import defaultdict, deque
from threading import Lock


SESSION_COOKIE = "mip_pilot_session"


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def create_session_token(
    secret: str,
    *,
    lifetime_seconds: int,
    now: int | None = None,
) -> str:
    issued_at = int(time.time() if now is None else now)
    payload = {
        "iat": issued_at,
        "exp": issued_at + lifetime_seconds,
        "nonce": secrets.token_urlsafe(12),
    }
    encoded = _b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signature = hmac.new(
        secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256
    ).digest()
    return f"{encoded}.{_b64encode(signature)}"


def verify_session_token(
    token: str | None,
    secret: str,
    *,
    now: int | None = None,
) -> bool:
    if not token:
        return False
    try:
        encoded, supplied_signature = token.split(".", 1)
        expected_signature = hmac.new(
            secret.encode("utf-8"), encoded.encode("ascii"), hashlib.sha256
        ).digest()
        if not hmac.compare_digest(
            _b64decode(supplied_signature), expected_signature
        ):
            return False
        payload = json.loads(_b64decode(encoded))
        current = int(time.time() if now is None else now)
        return (
            isinstance(payload.get("iat"), int)
            and isinstance(payload.get("exp"), int)
            and payload["iat"] <= current + 60
            and payload["exp"] > current
        )
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


class LoginThrottle:
    """Small in-memory limiter for one application process."""

    def __init__(self, *, max_failures: int, window_seconds: int):
        self.max_failures = max_failures
        self.window_seconds = window_seconds
        self._failures: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def _prune(self, key: str, now: float) -> deque[float]:
        failures = self._failures[key]
        cutoff = now - self.window_seconds
        while failures and failures[0] <= cutoff:
            failures.popleft()
        if not failures:
            self._failures.pop(key, None)
            return deque()
        return failures

    def retry_after(self, key: str, *, now: float | None = None) -> int:
        current = time.time() if now is None else now
        with self._lock:
            failures = self._prune(key, current)
            if len(failures) < self.max_failures:
                return 0
            return max(1, int(self.window_seconds - (current - failures[0])))

    def record_failure(self, key: str, *, now: float | None = None) -> int:
        current = time.time() if now is None else now
        with self._lock:
            failures = self._prune(key, current)
            failures.append(current)
            self._failures[key] = failures
            if len(failures) < self.max_failures:
                return 0
            return max(1, int(self.window_seconds - (current - failures[0])))

    def clear(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)
