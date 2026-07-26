from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .policy import sha256_file


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a hash-bound, explicitly uncertified policy for real-model "
            "engineering validation. This policy cannot enable clinical mode."
        )
    )
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--policy-version")
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--acceptance-threshold", type=float, default=0.5)
    parser.add_argument("--target-fnr", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.05)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    for name in (
        "prediction_threshold",
        "acceptance_threshold",
        "target_fnr",
        "delta",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.delta in {0.0, 1.0}:
        raise SystemExit("--delta must be in (0, 1)")

    checkpoints = tuple(path.resolve() for path in args.checkpoint)
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise SystemExit(f"checkpoint not found: {', '.join(missing)}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    policy = {
        "schema_version": 1,
        "policy_version": args.policy_version or f"validation-{timestamp}",
        "model_version": args.model_version,
        "checkpoint_sha256": [sha256_file(path) for path in checkpoints],
        "prediction_threshold": args.prediction_threshold,
        "acceptance_threshold": args.acceptance_threshold,
        "target_system_fnr": args.target_fnr,
        "confidence_delta": args.delta,
        "certified": False,
        "certification": {
            "method": "none - engineering validation only",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "notice": (
                "The acceptance threshold is provisional and was not independently "
                "certified. APP_MODE=clinical must reject this policy."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Validation policy written to {args.output}")
    print("UNCERTIFIED: use only with APP_MODE=validation")


if __name__ == "__main__":
    main()
