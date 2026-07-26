from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import fmean


def extract_five_fold_taus(path: Path, target: float = 0.05, cost: float = 5.0):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"Fold", "Swap", "Case", "Cost", "Target", "Threshold"}
    if not rows or not required.issubset(rows[0]):
        missing = sorted(required - set(rows[0] if rows else []))
        raise ValueError(f"result CSV is empty or missing columns: {missing}")

    selected = []
    for row in rows:
        if not str(row["Case"]).startswith("Case 4"):
            continue
        if not math.isclose(float(row["Cost"]), cost, abs_tol=1e-12):
            continue
        if not math.isclose(float(row["Target"]), target, abs_tol=1e-12):
            continue
        selected.append(row)

    fold_taus = []
    details = []
    for fold in range(1, 6):
        fold_rows = [row for row in selected if int(row["Fold"]) == fold]
        if len(fold_rows) != 2:
            raise ValueError(
                f"Fold {fold} must have exactly two Case 4 swap rows; found {len(fold_rows)}"
            )
        thresholds = [float(row["Threshold"]) for row in fold_rows]
        fold_tau = fmean(thresholds)
        fold_taus.append(fold_tau)
        details.append((fold, fold_rows[0]["Swap"], thresholds[0], fold_rows[1]["Swap"], thresholds[1], fold_tau))
    return details, fold_taus, fmean(fold_taus)


def main():
    parser = argparse.ArgumentParser(
        description="Extract five fold-level Case 4 taus from an existing result CSV."
    )
    parser.add_argument("result_csv", type=Path)
    parser.add_argument("--target", type=float, default=0.05)
    parser.add_argument("--cost", type=float, default=5.0)
    args = parser.parse_args()
    try:
        details, fold_taus, mean_tau = extract_five_fold_taus(
            args.result_csv, args.target, args.cost
        )
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
    for fold, swap1, tau1, swap2, tau2, fold_tau in details:
        print(
            f"Fold {fold}: {swap1}={tau1:.10f}, {swap2}={tau2:.10f}, "
            f"fold_tau={fold_tau:.10f}"
        )
    print(f"Five fold taus: {[round(value, 10) for value in fold_taus]}")
    print(f"Final mean tau: {mean_tau:.10f}")
    values = " ".join(f"{value:.10f}" for value in fold_taus)
    print(f"make_average_tau_policy.cmd {values}")


if __name__ == "__main__":
    main()
