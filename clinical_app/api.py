from __future__ import annotations

import hashlib
import io
import ipaddress
import json
import re
import secrets
import uuid
from pathlib import Path
from typing import Annotated, Literal

from fastapi import (
    Depends,
    Cookie,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings
from .database import PilotDatabase, utc_now
from .inference import create_backend
from .metrics import (
    paper_summary_rows,
    publication_validation_rows,
    summarize_postlaunch,
)
from .storage import ImageStore
from .web_auth import (
    SESSION_COOKIE,
    LoginThrottle,
    create_session_token,
    verify_session_token,
)


settings = Settings.from_env()
database = PilotDatabase(settings.database_path)
backend = create_backend(settings)
image_store = ImageStore(settings.image_store_dir, enabled=settings.persist_images)
static_dir = Path(__file__).parent / "static"
MODEL_DISPLAY_NAME = "HistoNexa-MIP"
login_throttle = LoginThrottle(
    max_failures=settings.auth_max_failures,
    window_seconds=settings.auth_window_seconds,
)


def _session_secret() -> str:
    if settings.session_secret:
        return settings.session_secret
    if settings.api_key:
        return hashlib.sha256(
            f"mip-pilot-browser-session:{settings.api_key}".encode("utf-8")
        ).hexdigest()
    return ""

app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description=(
        "Image-level MIP pattern review for pathologist self-review and patient "
        "information. This application is not a standalone diagnostic device."
    ),
    docs_url=None if settings.public_internet_mode else "/docs",
    redoc_url=None if settings.public_internet_mode else "/redoc",
    openapi_url=None if settings.public_internet_mode else "/openapi.json",
)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

if settings.allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
    )


@app.middleware("http")
async def security_headers(request, call_next):
    client_host = request.client.host if request.client else None
    if not _client_host_allowed(
        client_host,
        offline_only=settings.offline_only,
        allow_private_lan=settings.allow_private_lan,
    ):
        return JSONResponse(
            {"detail": "this offline deployment rejects requests from that network"},
            status_code=403,
        )
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=(), payment=(), usb=()"
    )
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; style-src 'self'; script-src 'self'; "
        "img-src 'self' blob: data:; connect-src 'self'"
    )
    response.headers["X-Clinical-Mode"] = backend.mode
    if settings.public_internet_mode:
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
    return response


def _client_host_allowed(
    client_host: str | None,
    *,
    offline_only: bool,
    allow_private_lan: bool,
) -> bool:
    """Allow loopback by default and private/link-local clients only in LAN mode."""
    if not offline_only or client_host in {
        None, "127.0.0.1", "::1", "localhost", "testclient"
    }:
        return True
    if not allow_private_lan:
        return False
    try:
        address = ipaddress.ip_address(client_host.split("%", 1)[0])
    except ValueError:
        return False
    return address.is_private or address.is_link_local


def _is_authenticated(
    *,
    x_api_key: str | None,
    session_cookie: str | None,
) -> bool:
    if not settings.api_key:
        return True
    if x_api_key and secrets.compare_digest(x_api_key, settings.api_key):
        return True
    return verify_session_token(session_cookie, _session_secret())


def require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    session_cookie: Annotated[
        str | None, Cookie(alias=SESSION_COOKIE)
    ] = None,
) -> None:
    del request
    if not _is_authenticated(
        x_api_key=x_api_key,
        session_cookie=session_cookie,
    ):
        raise HTTPException(status_code=401, detail="authentication required")


def _login_client_key(request: Request) -> str:
    direct = request.client.host if request.client else "unknown"
    if settings.public_internet_mode and direct in {"127.0.0.1", "::1", "localhost"}:
        forwarded = request.headers.get("CF-Connecting-IP", "").strip()
        try:
            return str(ipaddress.ip_address(forwarded))
        except ValueError:
            pass
    return direct


def _secure_session_cookie(request: Request) -> bool:
    if not settings.public_internet_mode:
        return False
    forwarded_proto = request.headers.get("X-Forwarded-Proto", "").lower()
    public_hostname = (request.url.hostname or "").lower() not in {
        "",
        "127.0.0.1",
        "::1",
        "localhost",
    }
    return (
        request.url.scheme == "https"
        or forwarded_proto == "https"
        or public_hostname
    )


class LoginRequest(BaseModel):
    password: str = Field(min_length=1, max_length=256)


class ReviewRequest(BaseModel):
    reviewer_id: str = Field(min_length=1, max_length=80)
    ground_truth_label: Literal[0, 1]
    ai_usefulness: Literal["helpful", "neutral", "unhelpful", "not_shown"]
    notes: str | None = Field(default=None, max_length=2000)


class ImageDiagnosis(BaseModel):
    case_id: str = Field(min_length=1, max_length=80)
    ground_truth_label: Literal[0, 1]
    notes: str | None = Field(default=None, max_length=1000)


class DiagnosisRequest(BaseModel):
    reviewer_id: str = Field(min_length=1, max_length=80)
    operator_id: str | None = Field(default=None, min_length=1, max_length=80)
    source_attested: bool = False
    images: list[ImageDiagnosis] = Field(min_length=1)
    patient_notes: str | None = Field(default=None, max_length=2000)
    surgical_procedure: Literal[
        "biopsy_or_other",
        "wedge_resection",
        "segmentectomy",
        "lobectomy",
        "bilobectomy",
        "pneumonectomy",
        "sleeve_resection",
    ] | None = None


class WorkflowStartRequest(BaseModel):
    reviewer_id: str = Field(min_length=1, max_length=80)
    operator_id: str | None = Field(default=None, min_length=1, max_length=80)


class ImageFeedback(BaseModel):
    case_id: str = Field(min_length=1, max_length=80)
    notes: str | None = Field(default=None, max_length=1000)


class FeedbackRequest(BaseModel):
    reviewer_id: str = Field(min_length=1, max_length=80)
    operator_id: str | None = Field(default=None, min_length=1, max_length=80)
    images: list[ImageFeedback] = Field(min_length=1)
    patient_notes: str | None = Field(default=None, max_length=2000)
    diagnosis_changed_after_ai: bool


def _validate_case_ref(case_ref: str) -> str:
    cleaned = case_ref.strip()
    if not 1 <= len(cleaned) <= 80:
        raise HTTPException(status_code=422, detail="case_ref must be 1-80 characters")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", cleaned):
        raise HTTPException(
            status_code=422,
            detail="case_ref must be pseudonymous and use only letters, digits, '.', '_' or '-'",
        )
    return cleaned


def _validate_evaluation_cohort(value: str) -> str:
    cleaned = value.strip()
    if not 1 <= len(cleaned) <= 80:
        raise HTTPException(
            status_code=422,
            detail="evaluation_cohort must be 1-80 characters",
        )
    if not re.fullmatch(r"[A-Za-z0-9._-]+", cleaned):
        raise HTTPException(
            status_code=422,
            detail=(
                "evaluation_cohort must use only letters, digits, '.', '_' or '-'"
            ),
        )
    return cleaned


def _clean_image_name(name: str | None, image_index: int) -> str:
    basename = Path(name or "").name.strip()
    if not basename:
        return f"image_{image_index}"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._")
    return (cleaned or f"image_{image_index}")[:160]


def _validate_image(image_bytes: bytes) -> tuple[int, int, str, str]:
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="image exceeds MAX_UPLOAD_MB")
    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            image_format = (image.format or "").upper()
            image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=415, detail="invalid or unsafe image file") from exc
    if width * height > settings.max_image_pixels:
        raise HTTPException(status_code=413, detail="image exceeds MAX_IMAGE_PIXELS")
    formats = {
        "PNG": ("png", "image/png"),
        "JPEG": ("jpg", "image/jpeg"),
        "TIFF": ("tif", "image/tiff"),
        "BMP": ("bmp", "image/bmp"),
    }
    if image_format not in formats:
        raise HTTPException(status_code=415, detail="unsupported image format")
    extension, content_type = formats[image_format]
    return width, height, extension, content_type


def _clinical_notice(submission: dict) -> str:
    if submission.get("use_context") == "patient_self_check":
        return (
            "Educational image-level information only. This result is not a pathology "
            "diagnosis, cannot assess the whole specimen, and must not be used to make "
            "treatment decisions. Please discuss it with a qualified pathologist."
        )
    if submission["pilot_phase"] == "silent":
        return (
            "Pathologist self-review: image results remain hidden until an independent "
            "assessment has been recorded for every submitted image."
        )
    if backend.mode == "demo":
        return "Demonstration only: current scores have no clinical meaning."
    if backend.mode == "validation":
        return (
            "Research image-review output only. It supplements, but does not replace, "
            "complete slide review and a qualified pathologist's diagnosis."
        )
    return (
        "Decision support only. The final diagnosis and management plan must be made "
        "by qualified clinicians."
    )


def _public_submission(submission: dict) -> dict:
    public = dict(submission)
    public["model_display_name"] = MODEL_DISPLAY_NAME
    public["images"] = [dict(image) for image in submission["images"]]
    visible = submission.get("use_context") == "patient_self_check" or not (
        submission["pilot_phase"] == "silent"
        and submission["review_status"] != "reviewed"
    )
    public["ai_result_visible"] = visible
    public["clinical_notice"] = _clinical_notice(submission)
    for image in public["images"]:
        storage_key = image.pop("image_storage_key", None)
        image["image_available"] = bool(storage_key)
        image["image_url"] = (
            f"/api/v1/cases/{image['case_id']}/image" if storage_key else None
        )
    if visible:
        return public

    hidden_values = {
        "p_mip": None,
        "selection_score": None,
        "predicted_label": None,
        "accepted": None,
        "decision": "silent_ai_hidden",
    }
    public.update(hidden_values)
    public.update(
        {
            "patient_p_mip": None,
            "patient_predicted_label": None,
            "patient_accepted": None,
            "patient_decision": "silent_ai_hidden",
        }
    )
    public["model_hashes"] = []
    for image in public["images"]:
        image.update(hidden_values)
        image["model_hashes"] = []
        image.pop("top_tiles", None)
    return public


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(static_dir / "index.html")


@app.get("/api/v1/public/system", include_in_schema=False)
def public_system_status():
    return {
        "app_version": __version__,
        "model_display_name": MODEL_DISPLAY_NAME,
        "service_available": backend.ready,
        "pilot_phase": settings.pilot_phase,
        "password_required": bool(settings.api_key),
        "public_internet_mode": settings.public_internet_mode,
        "service_scope": (
            "local_reviewer_demonstration"
            if settings.is_demo
            else (
                "public_https_password_protected"
                if settings.public_internet_mode
                else "private_local_or_lan"
            )
        ),
        "public_notice": (
            "Reviewer demonstration mode runs locally without a password. Scores are "
            "deterministic interface placeholders and are not clinical model outputs."
            if settings.is_demo
            else (
                "This public page provides image-level MIP pattern review for "
                "pathologist self-review and patient information. Uploads, results, "
                "records, and exports require an authorized session."
            )
        ),
        "data_notice": (
            "Reviewer demo uploads are processed locally and image bytes are not "
            "persisted when launched with the supplied reviewer scripts."
            if settings.is_demo
            else (
                "Inference and retained records stay on the collaborating clinician's "
                "host computer. In public mode, authorized uploads travel over the "
                "HTTPS connection to that computer; this application does not "
                "implement cloud record storage. Use only pseudonymous, "
                "protocol-approved data."
                if settings.public_internet_mode
                else "The service is restricted to its approved local deployment."
            )
        ),
    }


@app.get("/api/v1/auth/status", include_in_schema=False)
def authentication_status(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    session_cookie: Annotated[
        str | None, Cookie(alias=SESSION_COOKIE)
    ] = None,
):
    return {
        "authenticated": _is_authenticated(
            x_api_key=x_api_key,
            session_cookie=session_cookie,
        ),
        "password_required": bool(settings.api_key),
        "public_internet_mode": settings.public_internet_mode,
        "session_hours": settings.access_session_hours,
    }


@app.post("/api/v1/auth/login", include_in_schema=False)
def login(request: Request, credentials: LoginRequest):
    if not settings.api_key:
        return {"authenticated": True, "password_required": False}
    client_key = _login_client_key(request)
    retry_after = login_throttle.retry_after(client_key)
    if retry_after:
        return JSONResponse(
            {"detail": "too many failed login attempts; try again later"},
            status_code=429,
            headers={"Retry-After": str(retry_after)},
        )
    if not secrets.compare_digest(credentials.password, settings.api_key):
        retry_after = login_throttle.record_failure(client_key)
        if retry_after:
            return JSONResponse(
                {"detail": "too many failed login attempts; try again later"},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
        raise HTTPException(status_code=401, detail="invalid password")
    login_throttle.clear(client_key)
    lifetime = settings.access_session_hours * 60 * 60
    response = JSONResponse(
        {
            "authenticated": True,
            "password_required": True,
            "session_hours": settings.access_session_hours,
        }
    )
    response.set_cookie(
        key=SESSION_COOKIE,
        value=create_session_token(
            _session_secret(), lifetime_seconds=lifetime
        ),
        max_age=lifetime,
        path="/",
        secure=_secure_session_cookie(request),
        httponly=True,
        samesite="strict",
    )
    return response


@app.post("/api/v1/auth/logout", include_in_schema=False)
def logout(request: Request):
    response = JSONResponse({"authenticated": False})
    response.delete_cookie(
        key=SESSION_COOKIE,
        path="/",
        secure=_secure_session_cookie(request),
        httponly=True,
        samesite="strict",
    )
    return response


@app.get("/health/live", tags=["health"])
def liveness():
    return {"status": "live", "version": __version__}


@app.get("/health/ready", tags=["health"])
def readiness():
    payload = {
        "ready": backend.ready,
        "mode": backend.mode,
        "pilot_phase": settings.pilot_phase,
        "reason": backend.reason,
        "model_version": backend.policy.model_version,
        "policy_version": backend.policy.policy_version,
    }
    return JSONResponse(payload, status_code=200 if backend.ready else 503)


@app.get("/api/v1/system", dependencies=[Depends(require_api_key)])
def system_status():
    return {
        "app_version": __version__,
        "model_display_name": MODEL_DISPLAY_NAME,
        "mode": backend.mode,
        "pilot_phase": settings.pilot_phase,
        "ready": backend.ready,
        "reason": backend.reason,
        "clinical_use_allowed": backend.mode == "clinical" and backend.ready,
        "policy": backend.policy.to_public_dict(),
        "api_key_required": bool(settings.api_key),
        "max_images_per_case": settings.max_images_per_case,
        "max_case_upload_mb": round(
            settings.max_case_upload_bytes / 1024 / 1024, 2
        ),
        "aggregation_rule": "none_image_primary",
        "primary_metric_unit": "image",
        "deployment_scope": (
            "public_https_tunnel"
            if settings.public_internet_mode
            else "offline_private_lan"
            if settings.offline_only and settings.allow_private_lan
            else "offline_local" if settings.offline_only else "networked"
        ),
        "offline_only": settings.offline_only,
        "allow_private_lan": settings.allow_private_lan,
        "public_internet_mode": settings.public_internet_mode,
        "image_persistence_enabled": settings.persist_images,
        "privacy": (
            "Pseudonymous images, model outputs and physician reviews are retained on "
            "the collaborating clinician's host computer. Authorized uploads travel "
            "to this host over HTTPS; no cloud record store is implemented."
            if settings.public_internet_mode and settings.persist_images
            else "Pseudonymous images, model outputs and physician reviews are retained only "
            "on this local offline computer. No cloud upload is implemented. Use an "
            "approved encrypted removable drive for offline backups."
            if settings.persist_images
            else "Uploaded image bytes are processed in memory and are not persisted."
        ),
        "warning": "Image-level research support only; not a standalone diagnosis.",
    }


@app.post("/api/v1/cases", dependencies=[Depends(require_api_key)])
async def submit_case(
    case_ref: Annotated[str, Form()],
    images: Annotated[list[UploadFile] | None, File()] = None,
    image: Annotated[UploadFile | None, File()] = None,
    submitted_by: Annotated[str | None, Form()] = None,
    diagnosing_pathologist_id: Annotated[str | None, Form()] = None,
    use_context: Annotated[
        Literal["pathologist_self_review", "patient_self_check"], Form()
    ] = "pathologist_self_review",
    interaction_mode: Annotated[
        Literal["direct_on_device", "researcher_mediated", "self_service"], Form()
    ] = "direct_on_device",
    evaluation_cohort: Annotated[str, Form()] = "UNSPECIFIED",
    evidence_role: Annotated[
        Literal[
            "controlled_post_launch_pilot",
            "held_out_validation",
            "workflow_rehearsal",
            "patient_information",
        ],
        Form(),
    ] = "held_out_validation",
):
    if not backend.ready:
        raise HTTPException(status_code=503, detail=backend.reason or "model unavailable")
    case_ref = _validate_case_ref(case_ref)
    if database.case_ref_exists(case_ref):
        raise HTTPException(status_code=409, detail="case_ref has already been submitted")
    uploads = list(images or [])
    if image is not None:
        uploads.append(image)
    if not uploads:
        raise HTTPException(status_code=422, detail="at least one image is required")
    if len(uploads) > settings.max_images_per_case:
        raise HTTPException(
            status_code=413,
            detail=(
                f"case has {len(uploads)} images; "
                f"MAX_IMAGES_PER_CASE={settings.max_images_per_case}"
            ),
        )

    submission_id = str(uuid.uuid4())
    submitted_by_clean = submitted_by.strip()[:80] if submitted_by else None
    if not submitted_by_clean:
        raise HTTPException(status_code=422, detail="submitted_by cannot be blank")
    if use_context == "patient_self_check":
        interaction_mode = "self_service"
        pathologist_id_clean = None
        evaluation_cohort_clean = "PATIENT_INFORMATION"
        evidence_role = "patient_information"
    else:
        evaluation_cohort_clean = _validate_evaluation_cohort(
            evaluation_cohort
        )
        if evidence_role == "patient_information":
            raise HTTPException(
                status_code=422,
                detail="pathologist review cannot use patient_information evidence role",
            )
        pathologist_id_clean = (
            diagnosing_pathologist_id.strip()[:80]
            if diagnosing_pathologist_id
            else submitted_by_clean
        )
        if not pathologist_id_clean:
            raise HTTPException(
                status_code=422,
                detail="diagnosing_pathologist_id cannot be blank",
            )
        if (
            interaction_mode == "direct_on_device"
            and submitted_by_clean != pathologist_id_clean
        ):
            raise HTTPException(
                status_code=422,
                detail="direct workflow requires operator and pathologist IDs to match",
            )
    submitted_at = utc_now()
    total_bytes = 0
    seen_hashes: set[str] = set()
    records = []
    image_payloads: list[tuple[dict, bytes, str]] = []
    top_tiles_by_case: dict[str, list[dict]] = {}
    for image_index, upload in enumerate(uploads, start=1):
        image_bytes = await upload.read(settings.max_upload_bytes + 1)
        width, height, extension, content_type = _validate_image(image_bytes)
        total_bytes += len(image_bytes)
        if total_bytes > settings.max_case_upload_bytes:
            raise HTTPException(status_code=413, detail="case exceeds MAX_CASE_UPLOAD_MB")
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        if image_sha256 in seen_hashes:
            raise HTTPException(
                status_code=422,
                detail=f"duplicate image detected at position {image_index}",
            )
        seen_hashes.add(image_sha256)
        try:
            result = backend.predict(image_bytes)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"image {image_index}: {exc}",
            ) from exc
        case_id = str(uuid.uuid4())
        records.append(
            {
                "case_id": case_id,
                "case_ref": case_ref,
                "submitted_at": submitted_at,
                "submitted_by": submitted_by_clean,
                "use_context": use_context,
                "interaction_mode": interaction_mode,
                "diagnosing_pathologist_id": pathologist_id_clean,
                "submission_id": submission_id,
                "pilot_phase": settings.pilot_phase,
                "evaluation_cohort": evaluation_cohort_clean,
                "evidence_role": evidence_role,
                "image_index": image_index,
                "image_count": len(uploads),
                "image_name": _clean_image_name(upload.filename, image_index),
                "image_storage_key": None,
                "image_content_type": content_type,
                "image_size_bytes": len(image_bytes),
                "image_sha256": image_sha256,
                "image_width": width,
                "image_height": height,
                "model_version": backend.policy.model_version,
                "model_hashes_json": json.dumps(list(result.model_hashes)),
                "policy_version": backend.policy.policy_version,
                "prediction_threshold": backend.policy.prediction_threshold,
                "acceptance_threshold": backend.policy.acceptance_threshold,
                "target_system_fnr": backend.policy.target_system_fnr,
                "confidence_delta": backend.policy.confidence_delta,
                "backend_mode": backend.mode,
                "p_mip": result.p_mip,
                "selection_score": result.selection_score,
                "predicted_label": result.predicted_label,
                "accepted": int(result.accepted),
                "decision": result.decision,
                "inference_ms": result.inference_ms,
                "tile_count": result.tile_count,
            }
        )
        image_payloads.append((records[-1], image_bytes, extension))
        top_tiles_by_case[case_id] = list(result.top_tiles)

    stored_keys: list[str] = []
    try:
        for record, image_bytes, extension in image_payloads:
            storage_key = image_store.save(
                submission_id=submission_id,
                case_id=record["case_id"],
                extension=extension,
                content=image_bytes,
            )
            record["image_storage_key"] = storage_key
            if storage_key:
                stored_keys.append(storage_key)
        database.insert_submission(records)
    except ValueError as exc:
        for storage_key in stored_keys:
            image_store.delete(storage_key)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OSError as exc:
        for storage_key in stored_keys:
            image_store.delete(storage_key)
        raise HTTPException(status_code=507, detail="central image storage failed") from exc
    stored = database.get_submission(submission_id)
    assert stored is not None
    for stored_image in stored["images"]:
        stored_image["top_tiles"] = top_tiles_by_case.get(
            stored_image["case_id"], []
        )
    return _public_submission(stored)


@app.get("/api/v1/cases", dependencies=[Depends(require_api_key)])
def list_cases(
    status: Literal["pending", "reviewed"] | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
):
    return {
        "items": [
            _public_submission(submission)
            for submission in database.list_submissions(status, limit)
        ]
    }


@app.get("/api/v1/cases/{case_id}", dependencies=[Depends(require_api_key)])
def get_case(case_id: str):
    submission = database.get_submission(case_id)
    if submission is None:
        raise HTTPException(status_code=404, detail="case not found")
    return _public_submission(submission)


@app.get(
    "/api/v1/cases/{case_id}/image",
    dependencies=[Depends(require_api_key)],
    include_in_schema=False,
)
def get_case_image(case_id: str):
    image = database.get_case(case_id)
    if image is None:
        raise HTTPException(status_code=404, detail="image record not found")
    storage_key = image.get("image_storage_key")
    if not storage_key:
        raise HTTPException(status_code=404, detail="image was not persisted")
    try:
        path = image_store.path_for(storage_key)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="stored image unavailable") from exc
    return FileResponse(path, media_type=image.get("image_content_type"))


@app.post(
    "/api/v1/cases/{case_id}/workflow/start-diagnosis",
    dependencies=[Depends(require_api_key)],
)
def start_case_diagnosis(case_id: str, request: WorkflowStartRequest):
    reviewer_id = request.reviewer_id.strip()
    operator_id = (request.operator_id or request.reviewer_id).strip()
    if not reviewer_id:
        raise HTTPException(status_code=422, detail="reviewer_id cannot be blank")
    existing = database.get_submission(case_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="case not found")
    if existing.get("use_context") == "patient_self_check":
        raise HTTPException(
            status_code=409,
            detail="patient information sessions do not enter the pathologist review queue",
        )
    try:
        submission = database.start_diagnosis(
            case_id,
            reviewer_id=reviewer_id,
            operator_id=operator_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if submission is None:
        raise HTTPException(status_code=404, detail="case not found")
    return _public_submission(submission)


@app.post("/api/v1/cases/{case_id}/review", dependencies=[Depends(require_api_key)])
def review_case(case_id: str, review: ReviewRequest):
    reviewer_id = review.reviewer_id.strip()
    if not reviewer_id:
        raise HTTPException(status_code=422, detail="reviewer_id cannot be blank")
    existing = database.get_submission(case_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="case not found")
    if existing["image_count"] != 1:
        raise HTTPException(
            status_code=409,
            detail="multi-image sessions require one diagnosis for every image",
        )
    if existing.get("use_context") == "patient_self_check":
        raise HTTPException(
            status_code=409,
            detail="patient information sessions cannot be used as physician reference data",
        )
    usefulness = (
        "not_shown" if existing["pilot_phase"] == "silent" else review.ai_usefulness
    )
    submission = database.review_submission(
        case_id,
        reviewer_id=reviewer_id,
        ground_truth_label=review.ground_truth_label,
        ai_usefulness=usefulness,
        notes=review.notes.strip() if review.notes else None,
    )
    assert submission is not None
    return _public_submission(submission)


@app.post(
    "/api/v1/cases/{case_id}/diagnosis",
    dependencies=[Depends(require_api_key)],
)
def diagnose_case(case_id: str, diagnosis: DiagnosisRequest):
    reviewer_id = diagnosis.reviewer_id.strip()
    operator_id = (diagnosis.operator_id or diagnosis.reviewer_id).strip()
    if not reviewer_id:
        raise HTTPException(status_code=422, detail="reviewer_id cannot be blank")
    existing = database.get_submission(case_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="case not found")
    if existing.get("use_context") == "patient_self_check":
        raise HTTPException(
            status_code=409,
            detail="patient information sessions cannot be used as physician reference data",
        )
    try:
        submission = database.diagnose_submission(
            case_id,
            reviewer_id=reviewer_id,
            operator_id=operator_id,
            source_attested=diagnosis.source_attested,
            image_reviews=[item.model_dump() for item in diagnosis.images],
            patient_notes=(
                diagnosis.patient_notes.strip() if diagnosis.patient_notes else None
            ),
            surgical_procedure=diagnosis.surgical_procedure,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    assert submission is not None
    return _public_submission(submission)


@app.post(
    "/api/v1/cases/{case_id}/feedback",
    dependencies=[Depends(require_api_key)],
)
def record_case_feedback(case_id: str, feedback: FeedbackRequest):
    reviewer_id = feedback.reviewer_id.strip()
    operator_id = (feedback.operator_id or feedback.reviewer_id).strip()
    if not reviewer_id:
        raise HTTPException(status_code=422, detail="reviewer_id cannot be blank")
    existing = database.get_submission(case_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="case not found")
    if existing.get("use_context") == "patient_self_check":
        raise HTTPException(
            status_code=409,
            detail="patient information sessions do not collect physician feedback",
        )
    try:
        submission = database.record_feedback(
            case_id,
            reviewer_id=reviewer_id,
            operator_id=operator_id,
            image_feedback=[item.model_dump() for item in feedback.images],
            patient_notes=(
                feedback.patient_notes.strip() if feedback.patient_notes else None
            ),
            diagnosis_changed_after_ai=feedback.diagnosis_changed_after_ai,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    assert submission is not None
    return _public_submission(submission)


def _current_metrics_summary() -> dict:
    metrics = summarize_postlaunch(
        database.all_cases_for_metrics(), backend.policy.confidence_delta
    )
    metrics["target_system_fnr"] = backend.policy.target_system_fnr
    metrics["policy_version"] = backend.policy.policy_version
    metrics["certified_prelaunch"] = backend.policy.certified
    metrics["pilot_phase"] = settings.pilot_phase
    return metrics


@app.get("/api/v1/metrics/summary", dependencies=[Depends(require_api_key)])
def metrics_summary():
    return _current_metrics_summary()


@app.get("/api/v1/metrics/export.csv", dependencies=[Depends(require_api_key)])
def export_metrics():
    return PlainTextResponse(
        database.export_reviewed_csv(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=pilot_images.csv"},
    )


@app.get("/api/v1/metrics/export-paper.csv", dependencies=[Depends(require_api_key)])
def export_paper_metrics():
    fields, rows = publication_validation_rows(
        database.all_cases_for_metrics(),
        generated_at_utc=utc_now(),
        app_version=__version__,
        model_display_name=MODEL_DISPLAY_NAME,
        pilot_phase=settings.pilot_phase,
        target_system_fnr=backend.policy.target_system_fnr,
        confidence_delta=backend.policy.confidence_delta,
        certified_prelaunch=backend.policy.certified,
        current_prediction_threshold=backend.policy.prediction_threshold,
        current_acceptance_threshold=backend.policy.acceptance_threshold,
        certification=backend.policy.certification,
    )
    return PlainTextResponse(
        PilotDatabase._write_csv(fields, rows),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": (
                "attachment; filename=histonexa_validation_table.csv"
            )
        },
    )


@app.get(
    "/api/v1/metrics/export-paper-long.csv",
    dependencies=[Depends(require_api_key)],
)
def export_paper_metrics_long():
    fields, rows = paper_summary_rows(
        _current_metrics_summary(),
        generated_at_utc=utc_now(),
        app_version=__version__,
        model_display_name=MODEL_DISPLAY_NAME,
        model_version=backend.policy.model_version,
        policy_version=backend.policy.policy_version,
        pilot_phase=settings.pilot_phase,
        target_system_fnr=backend.policy.target_system_fnr,
        confidence_delta=backend.policy.confidence_delta,
        certified_prelaunch=backend.policy.certified,
    )
    return PlainTextResponse(
        PilotDatabase._write_csv(fields, rows),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": (
                "attachment; filename=histonexa_metric_dictionary.csv"
            )
        },
    )


@app.get("/api/v1/metrics/export-patients.csv", dependencies=[Depends(require_api_key)])
def export_patient_metrics():
    return PlainTextResponse(
        database.export_patients_csv(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=pilot_patients.csv"},
    )


@app.get("/api/v1/metrics/export-audit.csv", dependencies=[Depends(require_api_key)])
def export_audit_metrics():
    return PlainTextResponse(
        database.export_audit_csv(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=pilot_audit.csv"},
    )


@app.get("/api/v1/metrics/export-legacy.csv", dependencies=[Depends(require_api_key)])
def export_legacy_metrics():
    return PlainTextResponse(
        database.export_images_csv(legacy_only=True),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=legacy_engineering_data.csv"},
    )
