from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import replace
from pathlib import Path

from .batch_predict import EXTENSIONS
from .config import Settings
from .policy import DeploymentPolicy, sha256_file
from .torch_backend import TorchInferenceBackend


def read_manifest(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    patient_labels: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"patient_id", "image_path", "patient_label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest is missing columns: {', '.join(sorted(missing))}")
        for line_number, item in enumerate(reader, start=2):
            patient_id = (item.get("patient_id") or "").strip()
            if not patient_id:
                raise ValueError(f"manifest:{line_number}: patient_id cannot be blank")
            patient_label = int(item["patient_label"])
            if patient_label not in {0, 1}:
                raise ValueError(f"manifest:{line_number}: patient_label must be 0 or 1")
            if patient_id in patient_labels and patient_labels[patient_id] != patient_label:
                raise ValueError(f"manifest:{line_number}: inconsistent patient_label")
            patient_labels[patient_id] = patient_label
            image_path = Path(item["image_path"])
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            image_path = image_path.resolve()
            if not image_path.is_file() or image_path.suffix.lower() not in EXTENSIONS:
                raise ValueError(f"manifest:{line_number}: unsupported image: {image_path}")
            image_label_text = (item.get("image_label") or "").strip()
            image_label = int(image_label_text) if image_label_text else None
            if image_label not in {None, 0, 1}:
                raise ValueError(f"manifest:{line_number}: image_label must be 0, 1 or blank")
            rows.append(
                {
                    "patient_id": patient_id,
                    "patient_label": patient_label,
                    "image_label": image_label,
                    "image_path": image_path,
                }
            )
    if not rows:
        raise ValueError("manifest contains no images")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen ensemble on a patient-mapped calibration manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tile-batch-size", type=int, default=16)
    parser.add_argument("--max-tiles", type=int, default=1024)
    args = parser.parse_args()

    try:
        manifest = read_manifest(args.manifest)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    checkpoints = tuple(path.resolve() for path in args.checkpoint)
    hashes = tuple(sha256_file(path) for path in checkpoints)
    settings = replace(
        Settings.from_env(),
        app_mode="clinical",
        checkpoints=checkpoints,
        model_version=args.model_version,
        device=args.device,
        tile_batch_size=args.tile_batch_size,
        max_tiles=args.max_tiles,
    )
    policy = DeploymentPolicy(
        schema_version=1,
        policy_version="patient-raw-score-generation",
        model_version=args.model_version,
        checkpoint_sha256=hashes,
        prediction_threshold=0.5,
        acceptance_threshold=0.0,
        target_system_fnr=0.05,
        confidence_delta=0.05,
        certified=False,
        certification={"method": "not applicable"},
    )
    backend = TorchInferenceBackend(settings, policy)
    fields = [
        "patient_id", "patient_label", "image_id", "image_label",
        "p_mip", "selection_score", "tile_count",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(manifest, start=1):
            image_bytes = Path(row["image_path"]).read_bytes()
            result = backend.predict(image_bytes)
            writer.writerow(
                {
                    "patient_id": row["patient_id"],
                    "patient_label": row["patient_label"],
                    "image_id": hashlib.sha256(image_bytes).hexdigest()[:20],
                    "image_label": "" if row["image_label"] is None else row["image_label"],
                    "p_mip": f"{result.p_mip:.10f}",
                    "selection_score": f"{result.selection_score:.10f}",
                    "tile_count": result.tile_count,
                }
            )
            print(f"[{index}/{len(manifest)}] patient={row['patient_id']}")
    print(f"Wrote {len(manifest)} image predictions to {args.output}")


if __name__ == "__main__":
    main()
