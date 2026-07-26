from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import replace
from pathlib import Path

from .config import Settings
from .policy import DeploymentPolicy, sha256_file
from .torch_backend import TorchInferenceBackend


EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in EXTENSIONS
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate raw p_mip and selection_score CSV for an independent labeled set."
    )
    parser.add_argument("--positive-dir", type=Path, required=True)
    parser.add_argument("--negative-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tile-batch-size", type=int, default=16)
    parser.add_argument("--max-tiles", type=int, default=1024)
    args = parser.parse_args()

    checkpoints = tuple(path.resolve() for path in args.checkpoint)
    hashes = tuple(sha256_file(path) for path in checkpoints)
    base = Settings.from_env()
    settings = replace(
        base,
        app_mode="clinical",
        checkpoints=checkpoints,
        model_version=args.model_version,
        device=args.device,
        tile_batch_size=args.tile_batch_size,
        max_tiles=args.max_tiles,
    )
    raw_policy = DeploymentPolicy(
        schema_version=1,
        policy_version="raw-score-generation",
        model_version=args.model_version,
        checkpoint_sha256=hashes,
        prediction_threshold=0.5,
        acceptance_threshold=0.0,
        target_system_fnr=0.05,
        confidence_delta=0.05,
        certified=False,
        certification={"method": "not applicable"},
    )
    backend = TorchInferenceBackend(settings, raw_policy)
    labeled = [(path, 1) for path in image_files(args.positive_dir)] + [
        (path, 0) for path in image_files(args.negative_dir)
    ]
    if not labeled:
        raise SystemExit("no supported image files found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "label", "p_mip", "selection_score", "tile_count"],
        )
        writer.writeheader()
        for index, (path, label) in enumerate(labeled, start=1):
            image_bytes = path.read_bytes()
            result = backend.predict(image_bytes)
            writer.writerow(
                {
                    "image_id": hashlib.sha256(image_bytes).hexdigest()[:20],
                    "label": label,
                    "p_mip": f"{result.p_mip:.10f}",
                    "selection_score": f"{result.selection_score:.10f}",
                    "tile_count": result.tile_count,
                }
            )
            print(f"[{index}/{len(labeled)}] {path.name}")
    print(f"Wrote {len(labeled)} predictions to {args.output}")


if __name__ == "__main__":
    main()
