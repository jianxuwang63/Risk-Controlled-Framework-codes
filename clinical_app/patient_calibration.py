from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .policy import sha256_file
from .statistics import clopper_pearson_upper, required_positives_for_zero_failures


@dataclass(frozen=True)
class PatientImageScore:
    patient_id: str
    patient_label: int
    p_mip: float
    selection_score: float


def read_patient_predictions(path: Path) -> list[PatientImageScore]:
    rows: list[PatientImageScore] = []
    labels: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"patient_id", "patient_label", "p_mip", "selection_score"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for line_number, item in enumerate(reader, start=2):
            patient_id = (item.get("patient_id") or "").strip()
            label = int(item["patient_label"])
            p_mip = float(item["p_mip"])
            score = float(item["selection_score"])
            if not patient_id or label not in {0, 1}:
                raise ValueError(f"{path}:{line_number}: invalid patient identifier or label")
            if not 0 <= p_mip <= 1 or not 0 <= score <= 1:
                raise ValueError(f"{path}:{line_number}: scores must be in [0, 1]")
            if patient_id in labels and labels[patient_id] != label:
                raise ValueError(f"{path}:{line_number}: inconsistent patient_label")
            labels[patient_id] = label
            rows.append(PatientImageScore(patient_id, label, p_mip, score))
    if not rows:
        raise ValueError(f"{path} contains no predictions")
    return rows


def patient_operating_counts(
    rows: list[PatientImageScore],
    prediction_threshold: float,
    acceptance_threshold: float,
) -> tuple[int, int, int, int]:
    grouped: dict[str, list[PatientImageScore]] = defaultdict(list)
    for row in rows:
        grouped[row.patient_id].append(row)
    failures = positives = accepted = 0
    for images in grouped.values():
        label = images[0].patient_label
        positives += label == 1
        accepted_positive = any(
            image.selection_score >= acceptance_threshold
            and image.p_mip >= prediction_threshold
            for image in images
        )
        accepted_negative = all(
            image.selection_score >= acceptance_threshold
            and image.p_mip < prediction_threshold
            for image in images
        )
        is_accepted = accepted_positive or accepted_negative
        accepted += is_accepted
        failures += label == 1 and accepted_negative
    return int(failures), int(positives), int(accepted), len(grouped)


def tune_patient_threshold(
    rows: list[PatientImageScore], prediction_threshold: float, target: float
) -> float:
    candidates = sorted({row.selection_score for row in rows} | {1.0})
    for threshold in candidates:
        failures, positives, _, _ = patient_operating_counts(
            rows, prediction_threshold, threshold
        )
        if positives and failures / positives <= target:
            return threshold
    raise ValueError("no patient-level threshold satisfies the tuning target")


def build_policy(args: argparse.Namespace) -> tuple[dict, bool]:
    tuning = read_patient_predictions(args.tuning_csv)
    certification = read_patient_predictions(args.certification_csv)
    tuning_ids = {row.patient_id for row in tuning}
    certification_ids = {row.patient_id for row in certification}
    overlap = tuning_ids & certification_ids
    if overlap:
        raise ValueError("tuning and certification contain overlapping patient_id values")
    tuning_target = args.tuning_target if args.tuning_target is not None else args.target_fnr / 2
    threshold = tune_patient_threshold(tuning, args.prediction_threshold, tuning_target)
    tune_counts = patient_operating_counts(tuning, args.prediction_threshold, threshold)
    cert_counts = patient_operating_counts(
        certification, args.prediction_threshold, threshold
    )
    cert_failures, cert_positives, cert_accepted, cert_patients = cert_counts
    if cert_positives == 0:
        raise ValueError("certification data must contain positive patients")
    upper = clopper_pearson_upper(cert_failures, cert_positives, args.delta)
    certified = upper is not None and upper <= args.target_fnr
    checkpoints = tuple(path.resolve() for path in args.checkpoint)
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise ValueError(f"checkpoint not found: {checkpoint}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    policy = {
        "schema_version": 1,
        "policy_version": args.policy_version or f"patient-pilot-{timestamp}",
        "model_version": args.model_version,
        "checkpoint_sha256": [sha256_file(path) for path in checkpoints],
        "prediction_threshold": args.prediction_threshold,
        "acceptance_threshold": threshold,
        "target_system_fnr": args.target_fnr,
        "confidence_delta": args.delta,
        "certified": certified,
        "certification": {
            "method": "patient-level independent one-sided exact Clopper-Pearson bound",
            "calibration_unit": "patient",
            "aggregation_rule": "any accepted positive; all accepted negative; otherwise defer",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tuning_file": args.tuning_csv.name,
            "certification_file": args.certification_csv.name,
            "tuning_target": tuning_target,
            "tuning_patient_count": tune_counts[3],
            "tuning_positive_patient_count": tune_counts[1],
            "tuning_failures": tune_counts[0],
            "tuning_accepted_patient_count": tune_counts[2],
            "certification_patient_count": cert_patients,
            "certification_positive_patient_count": cert_positives,
            "certification_failures": cert_failures,
            "certification_accepted_patient_count": cert_accepted,
            "upper_bound": upper,
            "required_zero_failure_positive_patients": required_positives_for_zero_failures(
                args.target_fnr, args.delta
            ),
            "separation_assertion": "Threshold chosen only on tuning patients; certification patients were used once after freezing.",
        },
    }
    return policy, certified


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune and certify the deployed multi-image patient decision rule."
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
    args = parser.parse_args()
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
    args.output.write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Policy written to {args.output}")
    if not certified:
        raise SystemExit("CERTIFICATION FAILED: policy is marked certified=false")
    print("PATIENT-LEVEL CERTIFICATION PASSED")


if __name__ == "__main__":
    main()
