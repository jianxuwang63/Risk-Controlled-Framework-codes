from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from . import __version__


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_storage_path(root: Path, storage_key: str) -> Path:
    candidate = (root / storage_key).resolve()
    root = root.resolve()
    if candidate == root or root not in candidate.parents:
        raise ValueError(f"unsafe image storage key: {storage_key}")
    return candidate


def create_backup(data_dir: Path, destination: Path, label: str | None) -> Path:
    source = data_dir.expanduser().resolve()
    database_path = source / "pilot.db"
    image_root = source / "images"
    if not database_path.is_file():
        raise ValueError(f"pilot database not found: {database_path}")
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if destination == source or source in destination.parents:
        raise ValueError("backup destination must be outside the live data directory")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    clean_label = "".join(
        character for character in (label or "").strip() if character.isalnum() or character in "-_"
    )[:40]
    folder_name = f"mip_pilot_{timestamp}" + (f"_{clean_label}" if clean_label else "")
    final_dir = destination / folder_name
    if final_dir.exists():
        raise ValueError(f"backup already exists: {final_dir}")
    partial_dir = destination / f".{folder_name}.{uuid4().hex}.partial"
    partial_dir.mkdir(parents=False)

    try:
        backup_db = partial_dir / "pilot.db"
        with sqlite3.connect(database_path) as live, sqlite3.connect(backup_db) as backup:
            live.backup(backup)
        with sqlite3.connect(backup_db) as connection:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise ValueError(f"SQLite integrity check failed: {integrity}")
            case_count = int(connection.execute(
                "SELECT COUNT(DISTINCT submission_id) FROM cases"
            ).fetchone()[0])
            image_count = int(connection.execute("SELECT COUNT(*) FROM cases").fetchone()[0])
            audit_count = int(connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0])
            image_rows = connection.execute(
                """
                SELECT image_storage_key, image_sha256 FROM cases
                WHERE image_storage_key IS NOT NULL AND image_storage_key <> ''
                ORDER BY image_storage_key
                """
            ).fetchall()
            # The live database normally uses WAL mode. A SQLite backup can
            # retain that journal-mode setting, which makes a later read create
            # transient pilot.db-wal/pilot.db-shm files. Convert the standalone
            # snapshot to DELETE mode before checksumming so the manifest only
            # contains stable, portable files.
            connection.execute("PRAGMA journal_mode=DELETE")

        for sidecar_suffix in ("-wal", "-shm"):
            sidecar = Path(f"{backup_db}{sidecar_suffix}")
            if sidecar.exists():
                sidecar.unlink()

        copied_images = 0
        for storage_key, expected_hash in image_rows:
            source_image = _safe_storage_path(image_root, str(storage_key))
            if not source_image.is_file():
                raise ValueError(f"referenced image is missing: {storage_key}")
            actual_hash = sha256_file(source_image)
            if actual_hash != expected_hash:
                raise ValueError(f"image checksum mismatch: {storage_key}")
            destination_image = partial_dir / "images" / str(storage_key)
            destination_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, destination_image)
            copied_images += 1
        if copied_images != len(image_rows):
            raise ValueError("not all referenced images were copied")

        files = []
        total_bytes = 0
        for path in sorted(item for item in partial_dir.rglob("*") if item.is_file()):
            if path.name == "manifest.json":
                continue
            relative = path.relative_to(partial_dir).as_posix()
            size = path.stat().st_size
            total_bytes += size
            files.append({"path": relative, "size_bytes": size, "sha256": sha256_file(path)})
        manifest = {
            "schema_version": 1,
            "app_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "storage_mode": "offline_local_backup",
            "sqlite_integrity": "ok",
            "patient_case_count": case_count,
            "image_record_count": image_count,
            "persisted_image_count": copied_images,
            "audit_event_count": audit_count,
            "file_count": len(files),
            "total_bytes": total_bytes,
            "files": files,
        }
        (partial_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        partial_dir.replace(final_dir)
        return final_dir
    except Exception:
        shutil.rmtree(partial_dir, ignore_errors=True)
        raise


def verify_backup(backup_dir: Path) -> dict:
    root = backup_dir.expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest.get("files", []):
        path = _safe_storage_path(root, str(item["path"]))
        if not path.is_file():
            raise ValueError(f"backup file is missing: {item['path']}")
        if path.stat().st_size != int(item["size_bytes"]):
            raise ValueError(f"backup size mismatch: {item['path']}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"backup checksum mismatch: {item['path']}")
    database_uri = f"file:{(root / 'pilot.db').as_posix()}?mode=ro&immutable=1"
    with sqlite3.connect(database_uri, uri=True) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise ValueError(f"backup SQLite integrity check failed: {integrity}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or verify a complete offline MIP pilot backup."
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    try:
        if args.verify:
            manifest = verify_backup(args.verify)
            print(
                "BACKUP VERIFIED: "
                f"patients={manifest['patient_case_count']}, "
                f"images={manifest['image_record_count']}, "
                f"audit_events={manifest['audit_event_count']}"
            )
            return
        if not args.data_dir or not args.destination:
            raise ValueError("--data-dir and --destination are required for backup")
        backup_dir = create_backup(args.data_dir, args.destination, args.label)
        manifest = verify_backup(backup_dir)
    except (OSError, ValueError, sqlite3.DatabaseError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"BACKUP CREATED AND VERIFIED: {backup_dir}")
    print(
        f"patients={manifest['patient_case_count']}, "
        f"images={manifest['image_record_count']}, "
        f"audit_events={manifest['audit_event_count']}"
    )


if __name__ == "__main__":
    main()
