from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterator

import cv2
import numpy as np
import torch

from .config import Settings
from .inference import InferenceResult, _decision
from .model import SelectiveNetMIL, normalize_checkpoint
from .policy import DeploymentPolicy, sha256_file


@dataclass(frozen=True)
class PreparedImage:
    image_rgb: np.ndarray
    coordinates: tuple[tuple[int, int], ...]


def prepare_image(image_bytes: bytes, max_tiles: int) -> PreparedImage:
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("uploaded file cannot be decoded as an RGB image")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    patch_size = 512
    stride = 256
    original_h, original_w = image.shape[:2]
    pad_h = max(0, patch_size - original_h)
    pad_w = max(0, patch_size - original_w)
    if pad_h or pad_w:
        image = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT
        )
    height, width = image.shape[:2]
    ys = list(range(0, max(1, height - patch_size + 1), stride))
    xs = list(range(0, max(1, width - patch_size + 1), stride))
    if ys[-1] + patch_size < height:
        ys.append(height - patch_size)
    if xs[-1] + patch_size < width:
        xs.append(width - patch_size)
    coordinates = tuple((x, y) for y in ys for x in xs)
    if len(coordinates) > max_tiles:
        raise ValueError(
            f"image creates {len(coordinates)} tiles, exceeding MAX_TILES={max_tiles}; "
            "do not silently sample tiles in a safety-critical workflow"
        )
    return PreparedImage(image_rgb=image, coordinates=coordinates)


def iter_tile_batches(
    prepared: PreparedImage, batch_size: int
) -> Iterator[tuple[torch.Tensor, tuple[tuple[int, int], ...]]]:
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    for start in range(0, len(prepared.coordinates), batch_size):
        coords = prepared.coordinates[start : start + batch_size]
        batch = []
        for x, y in coords:
            patch = prepared.image_rgb[y : y + 512, x : x + 512]
            resized = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_LINEAR)
            normalized = (resized.astype(np.float32) / 255.0 - mean) / std
            batch.append(np.transpose(normalized, (2, 0, 1)))
        array = np.ascontiguousarray(np.stack(batch, axis=0))
        yield torch.from_numpy(array), coords


def _load_raw_checkpoint(path, device: torch.device) -> object:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


class TorchInferenceBackend:
    mode = "clinical"
    ready = True
    reason = None

    def __init__(self, settings: Settings, policy: DeploymentPolicy):
        self.settings = settings
        self.mode = settings.app_mode
        self.policy = policy
        self.device = self._resolve_device(settings.device)
        self.model_hashes = tuple(sha256_file(path) for path in settings.checkpoints)
        self.models: list[SelectiveNetMIL] = []
        for checkpoint in settings.checkpoints:
            model = SelectiveNetMIL()
            raw = _load_raw_checkpoint(checkpoint, torch.device("cpu"))
            state = normalize_checkpoint(raw)
            model.load_state_dict(state, strict=True)
            model.eval().to(self.device)
            self.models.append(model)
        self._lock = threading.Lock()

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("MODEL_DEVICE requests CUDA, but CUDA is unavailable")
        return device

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    def _predict_one(
        self, model: SelectiveNetMIL, prepared: PreparedImage
    ) -> tuple[float, float, np.ndarray]:
        features = []
        with torch.inference_mode():
            for tiles, _ in iter_tile_batches(
                prepared, self.settings.tile_batch_size
            ):
                tiles = tiles.to(self.device, non_blocking=True)
                with self._autocast():
                    encoded = model.encode_tiles(tiles)
                features.append(encoded.float())
            feature_tensor = torch.cat(features, dim=0)
            with self._autocast():
                logits, selection_logit, attention = model.predict_from_features(
                    feature_tensor
                )
                p_mip = torch.softmax(logits.float(), dim=1)[0, 1].item()
                selection_score = torch.sigmoid(selection_logit.float())[0, 0].item()
        return p_mip, selection_score, attention.float().cpu().numpy()

    def predict(self, image_bytes: bytes) -> InferenceResult:
        started = time.perf_counter()
        prepared = prepare_image(image_bytes, self.settings.max_tiles)
        probabilities: list[float] = []
        selection_scores: list[float] = []
        attention_vectors: list[np.ndarray] = []
        with self._lock:
            for model in self.models:
                probability, selection, attention = self._predict_one(model, prepared)
                probabilities.append(probability)
                selection_scores.append(selection)
                attention_vectors.append(attention)
        p_mip = float(np.mean(probabilities))
        selection_score = float(np.mean(selection_scores))
        mean_attention = np.mean(np.stack(attention_vectors, axis=0), axis=0)
        ranked = np.argsort(mean_attention)[::-1][:5]
        top_tiles: list[dict[str, Any]] = []
        for index in ranked:
            x, y = prepared.coordinates[int(index)]
            top_tiles.append(
                {
                    "x": int(x),
                    "y": int(y),
                    "size": 512,
                    "attention": float(mean_attention[int(index)]),
                }
            )
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
            tile_count=len(prepared.coordinates),
            model_hashes=self.model_hashes,
            top_tiles=tuple(top_tiles),
        )
