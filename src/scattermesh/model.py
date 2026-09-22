"""Budget-conditioned residual U-Net for separable mesh-density prediction."""

import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .distillation import (
    conditioning_features,
    conditioning_features_v2,
    rasterize_circle,
    rasterize_scene,
    resample_axis_profiles,
)


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        groups = min(8, output_channels)
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1, bias=False)
        )

    def forward(self, inputs):
        return F.silu(self.body(inputs) + self.skip(inputs))


class AxisDensityUNet(nn.Module):
    """Predict positive x/y probability profiles from a realizable 2D importance map."""

    def __init__(self, input_channels=5, conditioning_size=10, base_channels=16):
        super().__init__()
        base = int(base_channels)
        self.encoder_0 = ResidualBlock(input_channels, base)
        self.encoder_1 = ResidualBlock(base, 2 * base)
        self.encoder_2 = ResidualBlock(2 * base, 4 * base)
        self.bottleneck = ResidualBlock(4 * base, 8 * base)
        self.conditioning = nn.Sequential(
            nn.Linear(conditioning_size, 4 * base),
            nn.SiLU(),
            nn.Linear(4 * base, 16 * base),
        )
        self.decoder_2 = ResidualBlock(12 * base, 4 * base)
        self.decoder_1 = ResidualBlock(6 * base, 2 * base)
        self.decoder_0 = ResidualBlock(3 * base, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, raster, conditioning):
        level_0 = self.encoder_0(raster)
        level_1 = self.encoder_1(F.avg_pool2d(level_0, 2))
        level_2 = self.encoder_2(F.avg_pool2d(level_1, 2))
        latent = self.bottleneck(F.avg_pool2d(level_2, 2))
        scale, shift = self.conditioning(conditioning).chunk(2, dim=1)
        latent = latent * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        decoded_2 = self.decoder_2(
            torch.cat([F.interpolate(latent, size=level_2.shape[-2:], mode="nearest"), level_2], 1)
        )
        decoded_1 = self.decoder_1(
            torch.cat(
                [F.interpolate(decoded_2, size=level_1.shape[-2:], mode="nearest"), level_1],
                1,
            )
        )
        decoded_0 = self.decoder_0(
            torch.cat(
                [F.interpolate(decoded_1, size=level_0.shape[-2:], mode="nearest"), level_0],
                1,
            )
        )
        importance = self.head(decoded_0)
        x_logits = torch.logsumexp(importance[:, 0], dim=1) - math.log(importance.shape[2])
        y_logits = torch.logsumexp(importance[:, 0], dim=2) - math.log(importance.shape[3])
        return torch.stack([F.softmax(x_logits, dim=-1), F.softmax(y_logits, dim=-1)], dim=1)


class MeshDistillationDataset(Dataset):
    """Lazy geometry rasterization with all physics-evaluated candidate targets."""

    def __init__(
        self,
        dataset_path,
        split,
        *,
        raster_resolution=None,
        input_schema="circle_v1",
    ):
        metadata_path = Path(dataset_path)
        metadata = json.loads(metadata_path.read_text())
        arrays = np.load(metadata_path.parent / metadata["arrays"])
        indices = [
            index for index, example in enumerate(metadata["examples"]) if example["split"] == split
        ]
        if not indices:
            raise ValueError(f"No examples for split {split}")
        self.metadata = metadata
        self.examples = [metadata["examples"][index] for index in indices]
        self.profiles = arrays["profiles"][indices].copy()
        self.scores = arrays["scores"][indices].copy()
        self.candidate_mask = (
            arrays["candidate_mask"][indices].copy()
            if "candidate_mask" in arrays
            else np.ones_like(self.scores, dtype=bool)
        )
        arrays.close()
        self.resolution = int(raster_resolution or metadata["profile_bins"])
        if self.resolution != metadata["profile_bins"]:
            self.profiles = resample_axis_profiles(self.profiles, self.resolution)
        if input_schema not in {"circle_v1", "sparse_v2"}:
            raise ValueError(f"Unsupported model input schema: {input_schema}")
        self.input_schema = input_schema
        self._raster_cache = OrderedDict()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        key = example["geometry_id"]
        raster = self._raster_cache.get(key)
        if raster is None:
            if self.input_schema == "circle_v1":
                raster = rasterize_circle(example, self.resolution)
            else:
                raster = rasterize_scene(example, self.resolution)
            self._raster_cache[key] = raster
            if len(self._raster_cache) > 96:
                self._raster_cache.popitem(last=False)
        else:
            self._raster_cache.move_to_end(key)
        conditioning = (
            conditioning_features(example)
            if self.input_schema == "circle_v1"
            else conditioning_features_v2(example)
        )
        return {
            "raster": torch.from_numpy(raster),
            "conditioning": torch.from_numpy(conditioning),
            "profiles": torch.from_numpy(self.profiles[index]),
            "scores": torch.from_numpy(self.scores[index].astype(np.float32)),
            "candidate_mask": torch.from_numpy(self.candidate_mask[index]),
            "best_index": torch.tensor(example["best_candidate_index"], dtype=torch.long),
            "sample_index": torch.tensor(index, dtype=torch.long),
        }


def set_valued_profile_loss(
    prediction,
    targets,
    scores,
    best_index,
    candidate_mask=None,
    *,
    score_temperature=4.0,
    reduction="mean",
):
    """Combine best-candidate and physics-weighted set supervision in CDF space."""
    if prediction.ndim != 3 or targets.ndim != 4 or scores.ndim != 2:
        raise ValueError("Expected prediction [B,2,R], targets [B,K,2,R], scores [B,K]")
    if candidate_mask is None:
        candidate_mask = torch.ones_like(scores, dtype=torch.bool)
    if candidate_mask.shape != scores.shape or not candidate_mask.any(dim=1).all():
        raise ValueError("Each example requires at least one valid candidate")
    if not candidate_mask.gather(1, best_index[:, None]).all():
        raise ValueError("Selected best candidate must be valid")
    cumulative_prediction = prediction.cumsum(dim=-1)[:, None]
    cumulative_targets = targets.cumsum(dim=-1)
    cdf_loss = (cumulative_prediction - cumulative_targets).square().mean(dim=(-1, -2))
    density_loss = (prediction[:, None] - targets).abs().mean(dim=(-1, -2))
    candidate_loss = cdf_loss + 0.05 * density_loss
    masked_scores = scores.masked_fill(~candidate_mask, torch.inf)
    relative_log_score = torch.log(
        masked_scores / masked_scores.min(dim=1, keepdim=True).values
    )
    weights = F.softmax(-score_temperature * relative_log_score, dim=1)
    weighted = (weights * candidate_loss).sum(dim=1)
    selected = candidate_loss.gather(1, best_index[:, None]).squeeze(1)
    per_example = 0.5 * selected + 0.5 * weighted
    if reduction not in {"mean", "none"}:
        raise ValueError("Reduction must be mean or none")
    return (per_example.mean() if reduction == "mean" else per_example), {
        "selected_cdf_l1": selected.detach().mean(),
        "weighted_set_loss": weighted.detach().mean(),
    }
