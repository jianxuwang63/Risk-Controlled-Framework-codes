from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

from .policy import sha256_file


def build_average_tau_policy(args: argparse.Namespace) -> dict:
    taus = [float(value) for value in args.tau]
    if len(taus) != 5:
        raise ValueError("exactly five fold-specific --tau values are required")
    if any(not 0.0 <= value <= 1.0 for value in taus):
        raise ValueError("every tau must be in [0, 1]")
    if not 0.0 < args.target_fnr < 1.0:
        raise ValueError("target_fnr must be in (0, 1)")
    checkpoints = tuple(path.resolve() for path in args.checkpoint)
    if len(checkpoints) != 5:
        raise ValueError("exactly five fold checkpoints are required")
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise ValueError(f"checkpoint not found: {checkpoint}")
    average_tau = fmean(taus)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return {
        "schema_version": 1,
        "policy_version": args.policy_version or f"fold-mean-tau-{timestamp}",
        "model_version": args.model_version,
        "checkpoint_sha256": [sha256_file(path) for path in checkpoints],
        "prediction_threshold": args.prediction_threshold,
        "acceptance_threshold": average_tau,
        "target_system_fnr": args.target_fnr,
        "confidence_delta": args.delta,
        "certified": False,
        "certification": {
            "method": "advisor-prespecified arithmetic mean of five fold-specific Case 4 tau values",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "calibration_unit": "legacy image-level five-fold calibration",
            "fold_taus": taus,
            "mean_tau": average_tau,
            "target_fnr": args.target_fnr,
            "deployment_rule": "mean ensemble selection score >= mean_tau",
            "notice": (
                "This is a prespecified empirical pilot operating point, not an "
                "independent patient-level statistical certification."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an offline pilot policy from the arithmetic mean of five Case 4 taus."
    )
    parser.add_argument("--tau", action="append", type=float, required=True)
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--policy-version")
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--target-fnr", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.05)
    args = parser.parse_args()
    if not 0.0 < args.prediction_threshold < 1.0:
        raise SystemExit("--prediction-threshold must be in (0, 1)")
    if not 0.0 < args.delta < 1.0:
        raise SystemExit("--delta must be in (0, 1)")
    try:
        policy = build_average_tau_policy(args)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(f"Policy written to {args.output}")
    print(f"Five taus: {policy['certification']['fold_taus']}")
    print(f"Arithmetic mean tau: {policy['acceptance_threshold']:.10f}")


if __name__ == "__main__":
    main()
