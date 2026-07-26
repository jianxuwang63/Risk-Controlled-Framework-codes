from __future__ import annotations

import argparse
import os
import secrets
from pathlib import Path


def _create_once(path: Path, value: str) -> bool:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.write("\n")
    except FileExistsError:
        if not path.read_text(encoding="utf-8").strip():
            raise RuntimeError(f"existing secret file is empty: {path}")
        return False
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return True


def initialize_secrets(directory: Path) -> dict[str, bool]:
    directory.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(directory, 0o700)
    except OSError:
        pass
    return {
        "access_password": _create_once(
            directory / "public_access.key", secrets.token_urlsafe(32)
        ),
        "session_secret": _create_once(
            directory / "session.key", secrets.token_urlsafe(64)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create persistent local secrets for the public HTTPS pilot"
    )
    parser.add_argument("--secrets-dir", type=Path, default=Path("local_secrets"))
    args = parser.parse_args()
    created = initialize_secrets(args.secrets_dir.resolve())
    print("PUBLIC ACCESS SECRETS READY")
    print(f"Directory: {args.secrets_dir.resolve()}")
    print(
        "Access password: "
        + ("created" if created["access_password"] else "existing value preserved")
    )
    print(
        "Session secret: "
        + ("created" if created["session_secret"] else "existing value preserved")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
