"""Capacity-controlled residual U-Net for the second mesh curriculum."""

import math

import torch
from torch import nn
from torch.nn import functional as F


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
    relative_log_score = torch.log(masked_scores / masked_scores.min(dim=1, keepdim=True).values)
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


class AxisDensityUNetV2(nn.Module):
    """Map continuous-scene rasters to realizable x/y density profiles."""

    def __init__(self, input_channels=9, conditioning_size=15, base_channels=64):
        super().__init__()
        if base_channels < 16 or base_channels % 16:
            raise ValueError("base_channels must be a multiple of 16 and at least 16")
        a, b, c, d = (base_channels * i for i in (1, 2, 4, 8))
        self.encoder = nn.ModuleList(
            (
                ResidualBlock(input_channels, a),
                ResidualBlock(a, b),
                ResidualBlock(b, c),
                ResidualBlock(c, d),
            )
        )
        self.bottleneck = nn.Sequential(ResidualBlock(d, d), ResidualBlock(d, d))
        self.conditioning = nn.Sequential(
            nn.Linear(conditioning_size, d),
            nn.SiLU(),
            nn.Linear(d, 2 * d),
        )
        self.attention = nn.MultiheadAttention(d, 8, batch_first=True)
        self.attention_norm = nn.LayerNorm(d)
        self.decoder = nn.ModuleList(
            (
                ResidualBlock(2 * d, d),
                ResidualBlock(d + c, c),
                ResidualBlock(c + b, b),
                ResidualBlock(b + a, a),
            )
        )
        self.head = nn.Conv2d(a, 1, 1)

    def forward(self, raster, conditioning):
        if raster.ndim != 4 or raster.shape[1] != self.encoder[0].body[0].in_channels:
            raise ValueError("Unexpected raster shape")
        if raster.shape[-1] != raster.shape[-2] or raster.shape[-1] % 16:
            raise ValueError("Raster must be square with resolution divisible by 16")
        features = []
        hidden = raster
        for block in self.encoder:
            hidden = block(hidden)
            features.append(hidden)
            hidden = F.avg_pool2d(hidden, 2)
        hidden = self.bottleneck(hidden)
        scale, shift = self.conditioning(conditioning).chunk(2, dim=-1)
        hidden = hidden * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        batch, channels, height, width = hidden.shape
        tokens = hidden.flatten(2).transpose(1, 2)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        hidden = (
            self.attention_norm(tokens + attended)
            .transpose(1, 2)
            .reshape(batch, channels, height, width)
        )
        for block, skip in zip(self.decoder, reversed(features)):
            hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
            hidden = block(torch.cat((hidden, skip), dim=1))
        importance = self.head(hidden)[:, 0]
        # Width is x and height is y in the physical raster convention.
        x_logits = torch.logsumexp(importance, dim=1) - math.log(importance.shape[1])
        y_logits = torch.logsumexp(importance, dim=2) - math.log(importance.shape[2])
        probabilities = torch.stack(
            (F.softmax(x_logits.float(), -1), F.softmax(y_logits.float(), -1)), dim=1
        )
        probabilities = probabilities.clamp_min(1e-8)
        return probabilities / probabilities.sum(dim=-1, keepdim=True)
