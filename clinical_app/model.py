from __future__ import annotations

import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel


class SelectiveNetMIL(nn.Module):
    """Deployment-compatible copy of the paper's cost-sensitive SelectiveNet."""

    def __init__(self, dropout_rate: float = 0.2):
        super().__init__()
        config = ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=224,
            patch_size=16,
            num_channels=3,
        )
        self.vit = ViTModel(config)
        self.feature_dim = 768
        self.attention_V = nn.Sequential(nn.Linear(768, 256), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(768, 256), nn.Sigmoid())
        self.attention_weights = nn.Linear(256, 1)
        self.prediction_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 2),
        )
        self.aux_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 2),
        )
        self.selection_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def encode_tiles(self, tiles: torch.Tensor) -> torch.Tensor:
        output = self.vit(pixel_values=tiles)
        return output.last_hidden_state[:, 0, :]

    def predict_from_features(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attention = self.attention_weights(
            self.attention_V(features) * self.attention_U(features)
        ).squeeze(-1)
        attention = torch.softmax(attention, dim=0)
        pooled = torch.sum(attention.unsqueeze(-1) * features, dim=0, keepdim=True)
        prediction_logits = self.prediction_head(pooled)
        selection_logit = self.selection_head(pooled)
        return prediction_logits, selection_logit, attention


def normalize_checkpoint(raw: object) -> dict[str, torch.Tensor]:
    if not isinstance(raw, dict):
        raise ValueError("checkpoint must contain a state dictionary")
    state = raw
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    elif "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        clean_key = str(key)
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module.") :]
        normalized[clean_key] = value
    return normalized
