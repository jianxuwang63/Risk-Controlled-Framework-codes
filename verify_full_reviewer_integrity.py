from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHECKSUM_FILE = ROOT / "SHA256SUMS.txt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checksum_line(line: str) -> tuple[str, Path]:
    digest, separator, relative_name = line.partition("  ")
    if not separator or len(digest) != 64:
        raise ValueError(f"invalid checksum line: {line!r}")
    relative_path = Path(relative_name)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"unsafe checksum path: {relative_name!r}")
    return digest.lower(), ROOT / relative_path


def main() -> None:
    if not CHECKSUM_FILE.is_file():
        raise SystemExit("[ERROR] SHA256SUMS.txt is missing. Re-extract the full ZIP.")

    entries = [
        parse_checksum_line(line.strip())
        for line in CHECKSUM_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(entries) != 6:
        raise SystemExit(
            f"[ERROR] Expected 6 integrity entries but found {len(entries)}."
        )

    failures = []
    for expected, path in entries:
        if not path.is_file():
            failures.append(f"missing: {path.relative_to(ROOT)}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            failures.append(f"hash mismatch: {path.relative_to(ROOT)}")

    if failures:
        print("[ERROR] Full reviewer package integrity verification failed:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print(f"FULL REVIEWER PACKAGE INTEGRITY PASSED ({len(entries)} files)")


if __name__ == "__main__":
    main()
