from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .policy import sha256_file
from .statistics import (
    binomial_cdf,
    clopper_pearson_upper,
    required_positives_for_zero_failures,
)


@dataclass(frozen=True)
class PredictionRow:
    label: int
    p_mip: float
    selection_score: float


@dataclass(frozen=True)
class NPThreshold:
    threshold: float
    order_index: int
    binomial_tail: float
    positive_count: int


def select_np_threshold(
    positive_scores: list[float], alpha: float, delta: float
) -> NPThreshold:
    """Select the least conservative valid order statistic under the NP bound."""
    if not positive_scores:
        raise ValueError("positive_scores cannot be empty")
    if not 0.0 < alpha < 1.0 or not 0.0 < delta < 1.0:
        raise ValueError("alpha and delta must be in (0, 1)")
    scores = sorted(float(score) for score in positive_scores)
    valid: list[tuple[int, float]] = []
    for order_index in range(1, len(scores) + 1):
        tail = binomial_cdf(order_index - 1, len(scores), alpha)
        if tail <= delta:
            valid.append((order_index, tail))
    if not valid:
        minimum = required_positives_for_zero_failures(alpha, delta)
        raise ValueError(
            f"no non-trivial NP threshold is certifiable with {len(scores)} positives; "
            f"at least {minimum} positives are required even for the minimum-score threshold"
        )
    order_index, tail = valid[-1]
    return NPThreshold(
        threshold=scores[order_index - 1],
        order_index=order_index,
        binomial_tail=tail,
        positive_count=len(scores),
    )


def read_predictions(path: Path) -> list[PredictionRow]:
    rows: list[PredictionRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"label", "p_mip", "selection_score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for line_number, item in enumerate(reader, start=2):
            label = int(item["label"])
            p_mip = float(item["p_mip"])
            selection_score = float(item["selection_score"])
            if label not in {0, 1}:
                raise ValueError(f"{path}:{line_number}: label must be 0 or 1")
            if not 0.0 <= p_mip <= 1.0 or not 0.0 <= selection_score <= 1.0:
                raise ValueError(f"{path}:{line_number}: scores must be in [0, 1]")
            rows.append(PredictionRow(label, p_mip, selection_score))
    if not rows:
        raise ValueError(f"{path} contains no predictions")
    return rows


def system_failures(
    rows: list[PredictionRow], prediction_threshold: float, acceptance_threshold: float
) -> tuple[int, int, int]:
    positives = [row for row in rows if row.label == 1]
    failures = sum(
        row.selection_score >= acceptance_threshold
        and row.p_mip < prediction_threshold
        for row in positives
    )
    accepted = sum(row.selection_score >= acceptance_threshold for row in rows)
    return int(failures), len(positives), int(accepted)


def tune_acceptance_threshold(
    rows: list[PredictionRow], prediction_threshold: float, tuning_target: float
) -> float:
    positives = sum(row.label == 1 for row in rows)
    if positives == 0:
        raise ValueError("tuning data must contain positive cases")
    candidates = sorted({row.selection_score for row in rows} | {1.0})
    for threshold in candidates:
        failures, positive_count, _ = system_failures(
            rows, prediction_threshold, threshold
        )
        if failures / positive_count <= tuning_target:
            return threshold
    raise ValueError("no threshold in [0, 1] satisfies the tuning target")


def build_policy(args: argparse.Namespace) -> tuple[dict, bool]:
    tuning_rows = read_predictions(args.tuning_csv)
    certification_rows = read_predictions(args.certification_csv)
    tuning_target = (
        args.tuning_target
        if args.tuning_target is not None
        else args.target_fnr / 2.0
    )
    acceptance_threshold = tune_acceptance_threshold(
        tuning_rows, args.prediction_threshold, tuning_target
    )
    tune_failures, tune_positives, tune_accepted = system_failures(
        tuning_rows, args.prediction_threshold, acceptance_threshold
    )
    cert_failures, cert_positives, cert_accepted = system_failures(
        certification_rows, args.prediction_threshold, acceptance_threshold
    )
    if cert_positives == 0:
        raise ValueError("certification data must contain positive cases")
    upper_bound = clopper_pearson_upper(cert_failures, cert_positives, args.delta)
    certified = upper_bound is not None and upper_bound <= args.target_fnr
    checkpoints = tuple(path.resolve() for path in args.checkpoint)
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise ValueError(f"checkpoint not found: {checkpoint}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    policy = {
        "schema_version": 1,
        "policy_version": args.policy_version or f"pilot-{timestamp}",
        "model_version": args.model_version,
        "checkpoint_sha256": [sha256_file(path) for path in checkpoints],
        "prediction_threshold": args.prediction_threshold,
        "acceptance_threshold": acceptance_threshold,
        "target_system_fnr": args.target_fnr,
        "confidence_delta": args.delta,
        "certified": certified,
        "certification": {
            "method": "independent one-sided exact Clopper-Pearson bound",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tuning_file": args.tuning_csv.name,
            "certification_file": args.certification_csv.name,
            "tuning_target": tuning_target,
            "tuning_positive_count": tune_positives,
            "tuning_failures": tune_failures,
            "tuning_accepted_count": tune_accepted,
            "certification_positive_count": cert_positives,
            "certification_failures": cert_failures,
            "certification_accepted_count": cert_accepted,
            "upper_bound": upper_bound,
            "required_zero_failure_positives": required_positives_for_zero_failures(
                args.target_fnr, args.delta
            ),
            "separation_assertion": (
                "Threshold chosen only on tuning_csv; certification_csv was used only once "
                "for the prespecified operating point."
            ),
        },
    }
    return policy, certified


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune and independently certify the deployment abstention threshold."
    )
    parser.add_argument("--tuning-csv", type=Path, required=True)
    parser.add_argument("--certification-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--policy-version")
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--target-fnr", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--tuning-target", type=float)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.tuning_csv.resolve() == args.certification_csv.resolve():
        raise SystemExit("tuning and certification CSV files must be independent")
    for name in ("prediction_threshold", "target_fnr", "delta"):
        value = getattr(args, name)
        if not 0.0 < value < 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be in (0, 1)")
    if args.tuning_target is not None and not 0.0 <= args.tuning_target <= 1.0:
        raise SystemExit("--tuning-target must be in [0, 1]")
    try:
        policy, certified = build_policy(args)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    certification = policy["certification"]
    print(f"Policy written to {args.output}")
    print(
        f"Certification: failures={certification['certification_failures']}/"
        f"{certification['certification_positive_count']}, "
        f"upper={certification['upper_bound']:.6f}, target={args.target_fnr:.6f}"
    )
    if not certified:
        raise SystemExit(
            "CERTIFICATION FAILED: the file is intentionally marked certified=false and "
            "clinical mode will refuse to load it"
        )
    print("CERTIFICATION PASSED")


if __name__ == "__main__":
    main()
