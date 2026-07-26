from __future__ import annotations

from pathlib import Path
from uuid import uuid4


class ImageStore:
    """Server-side, pseudonymous image storage for the central pilot service."""

    def __init__(self, root: Path, *, enabled: bool):
        self.root = root.resolve()
        self.enabled = enabled
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        submission_id: str,
        case_id: str,
        extension: str,
        content: bytes,
    ) -> str | None:
        if not self.enabled:
            return None
        safe_extension = extension.lower().lstrip(".")
        if safe_extension not in {"png", "jpg", "tif", "bmp"}:
            raise ValueError("unsupported persisted image extension")
        relative = Path(submission_id) / f"{case_id}.{safe_extension}"
        destination = self._resolve(relative.as_posix())
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
        temporary.write_bytes(content)
        temporary.replace(destination)
        return relative.as_posix()

    def path_for(self, storage_key: str) -> Path:
        path = self._resolve(storage_key)
        if not path.is_file():
            raise FileNotFoundError(storage_key)
        return path

    def delete(self, storage_key: str | None) -> None:
        if not storage_key:
            return
        path = self._resolve(storage_key)
        if path.is_file():
            path.unlink()
        parent = path.parent
        if parent != self.root:
            try:
                parent.rmdir()
            except OSError:
                pass

    def _resolve(self, storage_key: str) -> Path:
        candidate = (self.root / storage_key).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError("invalid image storage key")
        return candidate
