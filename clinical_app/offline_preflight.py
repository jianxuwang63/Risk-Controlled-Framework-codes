from __future__ import annotations

import argparse
import json
import math
import shutil
import sqlite3
from pathlib import Path
from statistics import fmean

from .policy import sha256_file


EXPECTED_METHOD = (
    "advisor-prespecified arithmetic mean of five fold-specific Case 4 tau values"
)


def validate_offline_pilot(
    policy_path: Path,
    checkpoints: list[Path],
    data_dir: Path,
) -> dict:
    policy_path = policy_path.expanduser().resolve()
    if not policy_path.is_file():
        raise ValueError(f"policy not found: {policy_path}")
    if len(checkpoints) != 5:
        raise ValueError("exactly five checkpoints are required")
    resolved_checkpoints = [path.expanduser().resolve() for path in checkpoints]
    for checkpoint in resolved_checkpoints:
        if not checkpoint.is_file():
            raise ValueError(f"checkpoint not found: {checkpoint}")

    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid policy JSON: {exc}") from exc
    certification = policy.get("certification")
    if not isinstance(certification, dict) or certification.get("method") != EXPECTED_METHOD:
        raise ValueError(
            "the policy is not the required five-fold mean-tau pilot policy; "
            "run make_average_tau_policy.cmd with the five real Case 4 tau values"
        )
    fold_taus = certification.get("fold_taus")
    if not isinstance(fold_taus, list) or len(fold_taus) != 5:
        raise ValueError("the policy must record exactly five fold-specific tau values")
    try:
        numeric_taus = [float(value) for value in fold_taus]
        acceptance_threshold = float(policy["acceptance_threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("the policy contains an invalid tau value") from exc
    if any(not 0.0 <= value <= 1.0 for value in numeric_taus):
        raise ValueError("every fold-specific tau must be in [0, 1]")
    expected_mean = fmean(numeric_taus)
    if not math.isclose(acceptance_threshold, expected_mean, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("acceptance_threshold does not equal the arithmetic mean of fold_taus")
    if certification.get("deployment_rule") != "mean ensemble selection score >= mean_tau":
        raise ValueError("the policy does not record the required ensemble decision rule")
    if policy.get("certified") is not False:
        raise ValueError("the mean-tau pilot policy must remain certified=false")

    expected_hashes = policy.get("checkpoint_sha256")
    actual_hashes = [sha256_file(path) for path in resolved_checkpoints]
    if expected_hashes != actual_hashes:
        raise ValueError("checkpoint hashes do not match the mean-tau policy")

    data_dir = data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    probe = data_dir / ".offline-write-test"
    try:
        probe.write_bytes(b"offline-pilot-write-test")
        probe.unlink()
    except OSError as exc:
        raise ValueError(f"data directory is not writable: {data_dir}") from exc
    database = data_dir / "pilot.db"
    database_integrity = "new"
    if database.exists():
        database_uri = f"file:{database.as_posix()}?mode=ro&immutable=1"
        with sqlite3.connect(database_uri, uri=True) as connection:
            database_integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if database_integrity != "ok":
            raise ValueError(f"existing pilot database integrity check failed: {database_integrity}")
    free_bytes = shutil.disk_usage(data_dir).free
    return {
        "fold_taus": numeric_taus,
        "mean_tau": expected_mean,
        "checkpoint_count": len(resolved_checkpoints),
        "database_integrity": database_integrity,
        "free_bytes": free_bytes,
        "data_dir": str(data_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the Windows offline pilot before startup.")
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, action="append", type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = validate_offline_pilot(args.policy, args.checkpoint, args.data_dir)
    except (OSError, ValueError, sqlite3.DatabaseError) as exc:
        raise SystemExit(f"OFFLINE PREFLIGHT FAILED: {exc}") from exc
    print("OFFLINE PREFLIGHT PASSED")
    print(f"fold taus: {result['fold_taus']}")
    print(f"mean tau: {result['mean_tau']:.10f}")
    print(f"checkpoints: {result['checkpoint_count']}")
    print(f"database integrity: {result['database_integrity']}")
    print(f"data directory: {result['data_dir']}")
    print(f"free space: {result['free_bytes'] / (1024 ** 3):.2f} GiB")


if __name__ == "__main__":
    main()
