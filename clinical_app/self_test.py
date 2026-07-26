from __future__ import annotations

import io
import math

from PIL import Image, ImageDraw

from .config import Settings
from .inference import create_backend


def _synthetic_image_bytes() -> bytes:
    """Create a deterministic, non-clinical image for a forward-pass smoke test."""
    image = Image.new("RGB", (512, 512), (238, 220, 226))
    draw = ImageDraw.Draw(image)
    for index in range(12):
        left = 18 + index * 39
        top = 30 + (index % 4) * 104
        color = (118 + index * 7, 58 + index * 5, 108 + index * 6)
        draw.ellipse((left, top, left + 92, top + 68), fill=color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> None:
    settings = Settings.from_env()
    if settings.is_demo:
        raise RuntimeError("the deployment self-test cannot run in demo mode")

    backend = create_backend(settings)
    if not backend.ready:
        raise RuntimeError(f"model backend is unavailable: {backend.reason}")
    if len(backend.model_hashes) != 5:
        raise RuntimeError(
            f"expected five loaded checkpoints, found {len(backend.model_hashes)}"
        )

    result = backend.predict(_synthetic_image_bytes())
    if result.tile_count != 1:
        raise RuntimeError(f"expected one test tile, found {result.tile_count}")
    if not math.isfinite(result.p_mip) or not 0.0 <= result.p_mip <= 1.0:
        raise RuntimeError(f"invalid ensemble probability: {result.p_mip}")
    if not math.isfinite(result.selection_score) or not 0.0 <= result.selection_score <= 1.0:
        raise RuntimeError(f"invalid ensemble selection score: {result.selection_score}")

    expected_label = int(result.p_mip >= backend.policy.prediction_threshold)
    expected_accepted = result.selection_score >= backend.policy.acceptance_threshold
    if result.predicted_label != expected_label or result.accepted != expected_accepted:
        raise RuntimeError("the returned decision does not match the loaded policy")

    print("WINDOWS OFFLINE INFERENCE SELF-TEST PASSED")
    print(f"  policy version: {backend.policy.policy_version}")
    print(f"  checkpoint count: {len(backend.model_hashes)}")
    print(f"  mean tau: {backend.policy.acceptance_threshold:.10f}")
    print(f"  synthetic tiles: {result.tile_count}")
    print(f"  inference time: {result.inference_ms:.0f} ms")
    print("  database writes: none")


if __name__ == "__main__":
    main()
