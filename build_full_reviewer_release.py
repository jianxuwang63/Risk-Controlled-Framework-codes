from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
APP_VERSION = "0.10.15"
PACKAGE_NAME = f"HistoNexa-MIP-Full-Reviewer-v{APP_VERSION}"
WEIGHT_NAMES = [
    f"cost_5.0_fold_{fold}_LAST.pth" for fold in range(1, 6)
]
POLICY_NAME = "deployment_policy.cost5.validation.json"
MAX_RELEASE_ASSET_BYTES = 2 * 1024**3

ROOT_FILES = [
    "LICENSE",
    "README.md",
    "START_HERE_FULL_REVIEWER.md",
    "RESEARCH_USE_NOTICE.md",
    "THIRD_PARTY_MODEL_NOTICE.md",
    "REVIEWER_QUICKSTART.md",
    "requirements-app.txt",
    "verify_full_reviewer_integrity.py",
    "setup_full_reviewer_macos.command",
    "start_full_reviewer_macos.command",
    "setup_full_reviewer_windows.cmd",
    "start_full_reviewer_windows.cmd",
    "setup_full_reviewer_linux.sh",
    "start_full_reviewer_linux.sh",
]

FORBIDDEN_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "local_secrets",
    "photos",
    "photos_complete",
    "new_data",
    "newedata",
    "runtime",
    "outputs",
    "paper_outputs",
    "yes",
    "no",
}
FORBIDDEN_SUFFIXES = {
    ".db",
    ".sqlite",
    ".sqlite3",
    ".key",
    ".pem",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"required release file is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def load_and_validate_policy(weights_dir: Path) -> dict:
    policy_path = weights_dir / POLICY_NAME
    if not policy_path.is_file():
        raise FileNotFoundError(f"missing policy: {policy_path}")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))

    expected_hashes = policy.get("checkpoint_sha256")
    if not isinstance(expected_hashes, list) or len(expected_hashes) != 5:
        raise ValueError("policy must contain five checkpoint SHA-256 values")
    if policy.get("model_version") != "selectivenet-cost5-ensemble-v1":
        raise ValueError("unexpected model version in deployment policy")
    if policy.get("certified") is not False:
        raise ValueError("the reviewer policy must remain explicitly uncertified")
    if float(policy.get("target_system_fnr")) != 0.05:
        raise ValueError("the reviewer policy must record target_system_fnr=0.05")

    actual_hashes = []
    for name in WEIGHT_NAMES:
        checkpoint = weights_dir / name
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
        if checkpoint.stat().st_size >= MAX_RELEASE_ASSET_BYTES:
            raise ValueError(f"checkpoint exceeds GitHub Release limit: {checkpoint}")
        actual_hashes.append(sha256_file(checkpoint))
    if actual_hashes != expected_hashes:
        raise ValueError("checkpoint hashes do not match the deployment policy")
    return policy


def validate_staging_tree(staging: Path) -> None:
    required = [
        staging / "clinical_app" / "api.py",
        staging / "clinical_app" / "static" / "index.html",
        staging / "clinical_app" / "static" / "app.js",
        staging / "clinical_app" / "static" / "styles.css",
        staging / "weights" / POLICY_NAME,
        *(staging / "weights" / name for name in WEIGHT_NAMES),
        *(staging / name for name in ROOT_FILES),
    ]
    missing = [str(path.relative_to(staging)) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"release staging is incomplete: {missing}")

    for path in staging.rglob("*"):
        relative = path.relative_to(staging)
        lowered_parts = {part.lower() for part in relative.parts}
        if lowered_parts & FORBIDDEN_PARTS:
            raise ValueError(f"forbidden directory or file in release: {relative}")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError(f"credential/database-like file in release: {relative}")
        if path.name in {".DS_Store", ".env"} or path.name.startswith(".env."):
            raise ValueError(f"local environment artifact in release: {relative}")


def write_checksums(staging: Path) -> dict[str, str]:
    checksum_paths = [
        *(staging / "weights" / name for name in WEIGHT_NAMES),
        staging / "weights" / POLICY_NAME,
    ]
    checksums = {
        path.relative_to(staging).as_posix(): sha256_file(path)
        for path in checksum_paths
    }
    lines = [f"{digest}  {name}" for name, digest in checksums.items()]
    (staging / "SHA256SUMS.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return checksums


def write_release_manifest(
    staging: Path,
    policy: dict,
    checksums: dict[str, str],
) -> None:
    manifest = {
        "schema_version": 1,
        "package_name": PACKAGE_NAME,
        "app_version": APP_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "password-free localhost-only full reviewer artifact",
        "contains_real_model_weights": True,
        "contains_hospital_or_pilot_data": False,
        "contains_pathology_images": False,
        "persists_uploaded_image_bytes": False,
        "model_version": policy["model_version"],
        "policy_version": policy["policy_version"],
        "prediction_threshold": policy["prediction_threshold"],
        "acceptance_threshold": policy["acceptance_threshold"],
        "target_system_fnr": policy["target_system_fnr"],
        "certified": policy["certified"],
        "checkpoint_sha256": policy["checkpoint_sha256"],
        "selected_file_sha256": checksums,
    }
    (staging / "FULL_REVIEWER_RELEASE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def make_zip(staging: Path, archive_path: Path) -> None:
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for source in sorted(path for path in staging.rglob("*") if path.is_file()):
            relative = source.relative_to(staging.parent)
            archive.write(source, relative.as_posix())

    archive_size = archive_path.stat().st_size
    if archive_size >= MAX_RELEASE_ASSET_BYTES:
        raise ValueError(
            f"release ZIP is {archive_size / 1024**3:.2f} GiB and exceeds "
            "GitHub's 2 GiB per-release-asset limit"
        )
    archive_hash = sha256_file(archive_path)
    archive_path.with_suffix(archive_path.suffix + ".sha256").write_text(
        f"{archive_hash}  {archive_path.name}\n",
        encoding="utf-8",
    )


def build_release(output_dir: Path) -> tuple[Path, Path]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir == ROOT or ROOT in output_dir.parents and output_dir.name in {"clinical_app", "weights"}:
        raise ValueError("refusing to use a source directory as the release output")
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / PACKAGE_NAME
    archive_path = output_dir / f"{PACKAGE_NAME}.zip"

    policy = load_and_validate_policy(ROOT / "weights")

    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for name in ROOT_FILES:
        copy_required(ROOT / name, staging / name)

    shutil.copytree(
        ROOT / "clinical_app",
        staging / "clinical_app",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )
    copy_required(ROOT / "weights" / "README.md", staging / "weights" / "README.md")
    copy_required(
        ROOT / "weights" / POLICY_NAME,
        staging / "weights" / POLICY_NAME,
    )
    for name in WEIGHT_NAMES:
        destination = staging / "weights" / name
        copy_required(ROOT / "weights" / name, destination)
        destination.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)

    for launcher in [
        "setup_full_reviewer_macos.command",
        "start_full_reviewer_macos.command",
        "setup_full_reviewer_linux.sh",
        "start_full_reviewer_linux.sh",
    ]:
        path = staging / launcher
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    checksums = write_checksums(staging)
    write_release_manifest(staging, policy, checksums)
    validate_staging_tree(staging)
    make_zip(staging, archive_path)
    return staging, archive_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the complete GitHub Release reviewer package."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "release_build",
        help="Directory for the staged package and ZIP archive.",
    )
    args = parser.parse_args()
    try:
        staging, archive = build_release(args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"FULL REVIEWER RELEASE BUILD FAILED: {exc}") from exc

    print("FULL REVIEWER RELEASE BUILD PASSED")
    print(f"staging: {staging}")
    print(f"archive: {archive}")
    print(f"archive size: {archive.stat().st_size / 1024**3:.2f} GiB")
    print(f"archive SHA-256: {sha256_file(archive)}")


if __name__ == "__main__":
    main()
