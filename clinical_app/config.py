from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_mode: str
    pilot_phase: str
    app_name: str
    data_dir: Path
    database_path: Path
    offline_only: bool
    allow_private_lan: bool
    public_internet_mode: bool
    persist_images: bool
    image_store_dir: Path
    checkpoints: tuple[Path, ...]
    policy_path: Path
    model_version: str
    device: str
    tile_batch_size: int
    max_tiles: int
    max_upload_bytes: int
    max_image_pixels: int
    max_images_per_case: int
    max_case_upload_bytes: int
    api_key: str | None
    session_secret: str | None
    access_session_hours: int
    auth_max_failures: int
    auth_window_seconds: int
    allowed_origins: tuple[str, ...]
    log_level: str

    @property
    def is_demo(self) -> bool:
        return self.app_mode == "demo"

    @property
    def is_validation(self) -> bool:
        return self.app_mode == "validation"

    @classmethod
    def from_env(cls) -> "Settings":
        app_mode = os.getenv("APP_MODE", "clinical").strip().lower()
        if app_mode not in {"clinical", "validation", "demo"}:
            raise ValueError("APP_MODE must be 'clinical', 'validation', or 'demo'")
        pilot_phase = os.getenv("PILOT_PHASE", "silent").strip().lower()
        if pilot_phase not in {"silent", "assisted"}:
            raise ValueError("PILOT_PHASE must be 'silent' or 'assisted'")

        data_dir = Path(os.getenv("DATA_DIR", "runtime")).expanduser().resolve()
        database_path = Path(
            os.getenv("DATABASE_PATH", str(data_dir / "pilot.db"))
        ).expanduser().resolve()
        checkpoint_text = os.getenv("MODEL_CHECKPOINTS", "")
        checkpoints = tuple(
            Path(item.strip()).expanduser().resolve()
            for item in checkpoint_text.split(",")
            if item.strip()
        )
        origins = tuple(
            item.strip()
            for item in os.getenv("ALLOWED_ORIGINS", "").split(",")
            if item.strip()
        )

        settings = cls(
            app_mode=app_mode,
            pilot_phase=pilot_phase,
            app_name=os.getenv("APP_NAME", "MIP Risk-Controlled Pilot"),
            data_dir=data_dir,
            database_path=database_path,
            offline_only=_as_bool(os.getenv("OFFLINE_ONLY"), True),
            allow_private_lan=_as_bool(os.getenv("ALLOW_PRIVATE_LAN"), False),
            public_internet_mode=_as_bool(
                os.getenv("PUBLIC_INTERNET_MODE"), False
            ),
            persist_images=_as_bool(os.getenv("PERSIST_IMAGES"), True),
            image_store_dir=Path(
                os.getenv("IMAGE_STORE_DIR", str(data_dir / "images"))
            ).expanduser().resolve(),
            checkpoints=checkpoints,
            policy_path=Path(
                os.getenv(
                    "POLICY_PATH",
                    "weights/deployment_policy.cost5.validation.json",
                )
            ).expanduser().resolve(),
            model_version=os.getenv(
                "MODEL_VERSION", "selectivenet-cost5-ensemble-v1"
            ).strip(),
            device=os.getenv("MODEL_DEVICE", "auto").strip().lower(),
            tile_batch_size=max(1, int(os.getenv("TILE_BATCH_SIZE", "16"))),
            max_tiles=max(1, int(os.getenv("MAX_TILES", "1024"))),
            max_upload_bytes=max(
                1, int(float(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024)
            ),
            max_image_pixels=max(1, int(os.getenv("MAX_IMAGE_PIXELS", "200000000"))),
            max_images_per_case=max(
                1, int(os.getenv("MAX_IMAGES_PER_CASE", "300"))
            ),
            max_case_upload_bytes=max(
                1, int(float(os.getenv("MAX_CASE_UPLOAD_MB", "1000")) * 1024 * 1024)
            ),
            api_key=os.getenv("APP_API_KEY") or None,
            session_secret=os.getenv("APP_SESSION_SECRET") or None,
            access_session_hours=max(
                1, min(24, int(os.getenv("ACCESS_SESSION_HOURS", "8")))
            ),
            auth_max_failures=max(1, int(os.getenv("AUTH_MAX_FAILURES", "5"))),
            auth_window_seconds=max(
                60, int(os.getenv("AUTH_WINDOW_SECONDS", "900"))
            ),
            allowed_origins=origins,
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )
        if settings.allow_private_lan and not settings.offline_only:
            raise ValueError("ALLOW_PRIVATE_LAN=true requires OFFLINE_ONLY=true")
        if settings.allow_private_lan and not settings.api_key:
            raise ValueError(
                "ALLOW_PRIVATE_LAN=true requires a non-empty APP_API_KEY"
            )
        if settings.public_internet_mode:
            if settings.offline_only:
                raise ValueError(
                    "PUBLIC_INTERNET_MODE=true requires OFFLINE_ONLY=false"
                )
            if settings.allow_private_lan:
                raise ValueError(
                    "PUBLIC_INTERNET_MODE=true cannot use ALLOW_PRIVATE_LAN=true"
                )
            if not settings.api_key:
                raise ValueError(
                    "PUBLIC_INTERNET_MODE=true requires a non-empty APP_API_KEY"
                )
            if not settings.session_secret or len(settings.session_secret) < 32:
                raise ValueError(
                    "PUBLIC_INTERNET_MODE=true requires APP_SESSION_SECRET of at least 32 characters"
                )
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.database_path.parent.mkdir(parents=True, exist_ok=True)
        if settings.persist_images:
            settings.image_store_dir.mkdir(parents=True, exist_ok=True)
        return settings
