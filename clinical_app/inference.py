from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import Settings
from .policy import DeploymentPolicy, PolicyError


@dataclass(frozen=True)
class InferenceResult:
    p_mip: float
    selection_score: float
    predicted_label: int
    accepted: bool
    decision: str
    inference_ms: float
    tile_count: int
    model_hashes: tuple[str, ...]
    top_tiles: tuple[dict[str, Any], ...] = ()


class InferenceBackend(Protocol):
    mode: str
    ready: bool
    reason: str | None
    policy: DeploymentPolicy
    model_hashes: tuple[str, ...]

    def predict(self, image_bytes: bytes) -> InferenceResult: ...


def _decision(p_mip: float, selection_score: float, policy: DeploymentPolicy) -> tuple[int, bool, str]:
    predicted_label = int(p_mip >= policy.prediction_threshold)
    accepted = selection_score >= policy.acceptance_threshold
    if not accepted:
        return predicted_label, False, "defer_to_pathologist"
    return (
        predicted_label,
        True,
        "ai_mip_present" if predicted_label else "ai_mip_absent",
    )


class DemoBackend:
    mode = "demo"
    ready = True
    reason = "DEMO scores are deterministic placeholders and have no clinical meaning."

    def __init__(self) -> None:
        self.policy = DeploymentPolicy.demo()
        self.model_hashes: tuple[str, ...] = ()

    def predict(self, image_bytes: bytes) -> InferenceResult:
        started = time.perf_counter()
        digest = hashlib.sha256(image_bytes).digest()
        p_mip = int.from_bytes(digest[:4], "big") / (2**32 - 1)
        selection_score = int.from_bytes(digest[4:8], "big") / (2**32 - 1)
        predicted, accepted, decision = _decision(
            p_mip, selection_score, self.policy
        )
        return InferenceResult(
            p_mip=p_mip,
            selection_score=selection_score,
            predicted_label=predicted,
            accepted=accepted,
            decision=decision,
            inference_ms=(time.perf_counter() - started) * 1000,
            tile_count=0,
            model_hashes=(),
        )


class UnavailableBackend:
    ready = False

    def __init__(self, reason: str, model_version: str, mode: str = "clinical"):
        self.mode = mode
        self.reason = reason
        self.policy = DeploymentPolicy(
            schema_version=1,
            policy_version="unavailable",
            model_version=model_version,
            checkpoint_sha256=(),
            prediction_threshold=0.5,
            acceptance_threshold=1.0,
            target_system_fnr=0.05,
            confidence_delta=0.05,
            certified=False,
            certification={"method": "none", "reason": reason},
        )
        self.model_hashes: tuple[str, ...] = ()

    def predict(self, image_bytes: bytes) -> InferenceResult:
        raise RuntimeError(self.reason)


def create_backend(settings: Settings) -> InferenceBackend:
    if settings.is_demo:
        return DemoBackend()
    try:
        policy = DeploymentPolicy.load(
            settings.policy_path,
            expected_model_version=settings.model_version,
            checkpoints=settings.checkpoints,
            require_certified=not settings.is_validation,
        )
        from .torch_backend import TorchInferenceBackend

        return TorchInferenceBackend(settings, policy)
    except (PolicyError, ImportError, OSError, RuntimeError, ValueError) as exc:
        return UnavailableBackend(str(exc), settings.model_version, settings.app_mode)
