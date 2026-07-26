from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PolicyError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class DeploymentPolicy:
    schema_version: int
    policy_version: str
    model_version: str
    checkpoint_sha256: tuple[str, ...]
    prediction_threshold: float
    acceptance_threshold: float
    target_system_fnr: float
    confidence_delta: float
    certified: bool
    certification: dict[str, Any]

    @classmethod
    def demo(cls) -> "DeploymentPolicy":
        return cls(
            schema_version=1,
            policy_version="demo-unverified",
            model_version="demo-hash-scorer",
            checkpoint_sha256=(),
            prediction_threshold=0.5,
            acceptance_threshold=0.7,
            target_system_fnr=0.05,
            confidence_delta=0.05,
            certified=False,
            certification={"method": "none", "notice": "DEMO ONLY"},
        )

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_model_version: str,
        checkpoints: tuple[Path, ...],
        require_certified: bool = True,
    ) -> "DeploymentPolicy":
        if not path.is_file():
            raise PolicyError(f"policy file not found: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PolicyError(f"cannot read policy: {exc}") from exc

        required = {
            "schema_version",
            "policy_version",
            "model_version",
            "checkpoint_sha256",
            "prediction_threshold",
            "acceptance_threshold",
            "target_system_fnr",
            "confidence_delta",
            "certified",
            "certification",
        }
        missing = sorted(required - raw.keys())
        if missing:
            raise PolicyError(f"policy is missing fields: {', '.join(missing)}")

        policy = cls(
            schema_version=int(raw["schema_version"]),
            policy_version=str(raw["policy_version"]),
            model_version=str(raw["model_version"]),
            checkpoint_sha256=tuple(str(x).lower() for x in raw["checkpoint_sha256"]),
            prediction_threshold=float(raw["prediction_threshold"]),
            acceptance_threshold=float(raw["acceptance_threshold"]),
            target_system_fnr=float(raw["target_system_fnr"]),
            confidence_delta=float(raw["confidence_delta"]),
            certified=bool(raw["certified"]),
            certification=dict(raw["certification"]),
        )
        policy.validate()
        if policy.model_version != expected_model_version:
            raise PolicyError(
                f"policy model_version={policy.model_version!r} does not match "
                f"MODEL_VERSION={expected_model_version!r}"
            )
        if require_certified and not policy.certified:
            raise PolicyError("clinical mode requires a certified policy")
        if not checkpoints:
            raise PolicyError("clinical mode requires at least one checkpoint")
        for checkpoint in checkpoints:
            if not checkpoint.is_file():
                raise PolicyError(f"checkpoint not found: {checkpoint}")
        actual_hashes = tuple(sha256_file(path) for path in checkpoints)
        if actual_hashes != policy.checkpoint_sha256:
            raise PolicyError(
                "checkpoint hashes do not match the independently certified policy"
            )
        return policy

    def validate(self) -> None:
        if self.schema_version != 1:
            raise PolicyError("unsupported policy schema_version")
        for name, value in (
            ("prediction_threshold", self.prediction_threshold),
            ("acceptance_threshold", self.acceptance_threshold),
            ("target_system_fnr", self.target_system_fnr),
            ("confidence_delta", self.confidence_delta),
        ):
            if not 0.0 <= value <= 1.0:
                raise PolicyError(f"{name} must be in [0, 1]")
        if self.confidence_delta in {0.0, 1.0}:
            raise PolicyError("confidence_delta must be in (0, 1)")

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "model_version": self.model_version,
            "prediction_threshold": self.prediction_threshold,
            "acceptance_threshold": self.acceptance_threshold,
            "target_system_fnr": self.target_system_fnr,
            "confidence": 1.0 - self.confidence_delta,
            "certified": self.certified,
            "certification": self.certification,
        }
