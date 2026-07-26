import tempfile
import unittest
from pathlib import Path

from clinical_app.public_setup import initialize_secrets
from clinical_app.web_auth import (
    LoginThrottle,
    create_session_token,
    verify_session_token,
)


class BrowserAuthenticationTests(unittest.TestCase):
    def test_signed_session_is_bound_to_secret_and_expiry(self):
        token = create_session_token(
            "a" * 32,
            lifetime_seconds=120,
            now=1_000,
        )
        self.assertTrue(verify_session_token(token, "a" * 32, now=1_100))
        self.assertFalse(verify_session_token(token, "b" * 32, now=1_100))
        self.assertFalse(verify_session_token(token, "a" * 32, now=1_120))
        self.assertFalse(verify_session_token(token + "tampered", "a" * 32, now=1_100))

    def test_login_failures_are_limited_and_can_be_cleared(self):
        throttle = LoginThrottle(max_failures=3, window_seconds=60)
        self.assertEqual(throttle.record_failure("client", now=100), 0)
        self.assertEqual(throttle.record_failure("client", now=110), 0)
        self.assertGreater(throttle.record_failure("client", now=120), 0)
        self.assertGreater(throttle.retry_after("client", now=125), 0)
        throttle.clear("client")
        self.assertEqual(throttle.retry_after("client", now=125), 0)
        self.assertEqual(throttle.record_failure("client", now=200), 0)

    def test_public_password_is_created_once_and_preserved(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            first = initialize_secrets(root)
            password = (root / "public_access.key").read_text(encoding="utf-8")
            session = (root / "session.key").read_text(encoding="utf-8")
            second = initialize_secrets(root)
            self.assertTrue(first["access_password"])
            self.assertTrue(first["session_secret"])
            self.assertFalse(second["access_password"])
            self.assertFalse(second["session_secret"])
            self.assertEqual(
                (root / "public_access.key").read_text(encoding="utf-8"),
                password,
            )
            self.assertEqual(
                (root / "session.key").read_text(encoding="utf-8"),
                session,
            )
            self.assertGreaterEqual(len(password.strip()), 32)
            self.assertGreaterEqual(len(session.strip()), 64)
