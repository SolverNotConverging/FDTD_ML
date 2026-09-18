"""FiLM-conditioned ResU-Net, raw scene rasterization, and versioned checkpoints."""

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .constants import C0, EPS0
from .mesh import MESH_POLICY, AxisCollar, cell_count, density_mesh, projected_density
from .scene import raster_index

CHANNELS = [
    "epsilon_r",
    "sigma_over_omega_eps0",
    "PEC",
    "source",
    "receiver",
    "x_anchor",
    "y_anchor",
    "mu_r",
    "PML",
]
CONDITIONING = [
    "log_Lx_over_lambda0",
    "log_Ly_over_lambda0",
    "f_min_over_f_max",
    "log_Nx",
    "log_Ny",
]
NORMALIZATION = {
    "epsilon_r": "identity",
    "sigma_over_omega_eps0": "sigma/(2*pi*f_max*eps0)",
    "PEC": "binary",
    "source": "binary",
    "receiver": "binary",
    "x_anchor": "binary",
    "y_anchor": "binary",
    "mu_r": "identity",
    "PML": "binary",
}


class FiLMBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.norm1 = nn.GroupNorm(1, cout)
        self.norm2 = nn.GroupNorm(1, cout)
        self.film = nn.Linear(5, 2 * cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, condition):
        scale, bias = self.film(condition).chunk(2, dim=1)
        h = self.norm1(self.conv1(x))
        h = F.silu(h * (1 + scale[:, :, None, None]) + bias[:, :, None, None])
        return F.silu(self.norm2(self.conv2(h)) + self.skip(x))


class ResUNet(nn.Module):
    """Produces logits (B,2,H,W), never mesh coordinates. Supports odd raster sizes."""

    def __init__(self, width=16):
        super().__init__()
        if isinstance(width, bool) or int(width) != width or width < 2:
            raise ValueError("width must be an integer >= 2")
        width = self.width = int(width)
        self.enc1 = FiLMBlock(len(CHANNELS), width)
        self.enc2 = FiLMBlock(width, 2 * width)
        self.bottom = FiLMBlock(2 * width, 4 * width)
        self.dec2 = FiLMBlock(6 * width, 2 * width)
        self.dec1 = FiLMBlock(3 * width, width)
        self.head = nn.Conv2d(width, 2, 1)

    def forward(self, raster, condition):
        if (
            raster.ndim != 4
            or raster.shape[1] != len(CHANNELS)
            or min(raster.shape[2:]) < 4
            or condition.shape != (raster.shape[0], 5)
        ):
            raise ValueError("Expected raster (B,9,H,W), H/W>=4, and conditioning (B,5)")
        a = self.enc1(raster, condition)
        b = self.enc2(F.avg_pool2d(a, 2), condition)
        h = self.bottom(F.avg_pool2d(b, 2), condition)
        h = self.dec2(
            torch.cat(
                [F.interpolate(h, size=b.shape[-2:], mode="bilinear", align_corners=False), b], 1
            ),
            condition,
        )
        h = self.dec1(
            torch.cat(
                [F.interpolate(h, size=a.shape[-2:], mode="bilinear", align_corners=False), a], 1
            ),
            condition,
        )
        return self.head(h)


def pool_axes(logits, alpha=4.0):
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Pooling alpha must be finite and positive")
    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError("Expected logits (B,2,H,W)")
    # Log-mean-exp removes raster-size dependence for constant logits.
    qx = (torch.logsumexp(alpha * logits[:, 0], dim=1) - math.log(logits.shape[2])) / alpha
    qy = (torch.logsumexp(alpha * logits[:, 1], dim=2) - math.log(logits.shape[3])) / alpha
    return F.softplus(qx) + 1e-6, F.softplus(qy) + 1e-6


def rasterize(scene, shape, f_max):
    H, W = shape
    H, W = cell_count(H), cell_count(W)
    if H < 4 or W < 4:
        raise ValueError("Raster dimensions must be >= 4")
    if not np.isfinite(f_max) or f_max <= 0:
        raise ValueError("f_max must be finite and positive")
    x = (np.arange(W) + 0.5) * scene.Lx / W
    y = (np.arange(H) + 0.5) * scene.Ly / H
    eps, mu, sigma, pec = scene.sample(x, y, raster=True)
    raster = np.zeros((len(CHANNELS), H, W), dtype=np.float32)
    raster[0], raster[1], raster[2], raster[7] = (
        eps.T,
        sigma.T / (2 * np.pi * f_max * EPS0),
        pec.T,
        mu.T,
    )
    if scene.pml is not None:
        x0, x1, y0, y1 = scene.pml.interfaces(scene)
        raster[8] = (x[None, :] < x0) | (x[None, :] > x1) | (y[:, None] < y0) | (y[:, None] > y1)
    for channel, probes in ((3, [s[0] for s in scene.sources]), (4, scene.receivers)):
        for p in probes:
            if p.kind == "point":
                raster[channel, raster_index(p.y, scene.Ly, H), raster_index(p.x, scene.Lx, W)] = 1
            elif np.ndim(p.x) == 0:
                start, stop = [raster_index(v, scene.Ly, H) for v in p.y]
                raster[channel, start : stop + 1, raster_index(p.x, scene.Lx, W)] = 1
            else:
                start, stop = [raster_index(v, scene.Lx, W) for v in p.x]
                raster[channel, raster_index(p.y, scene.Ly, H), start : stop + 1] = 1
    for a in scene.x_anchors:
        raster[5, :, raster_index(a, scene.Lx, W)] = 1
    for a in scene.y_anchors:
        raster[6, raster_index(a, scene.Ly, H), :] = 1
    if not np.isfinite(raster).all():
        raise ValueError("Scene raster overflows float32")
    return raster


def conditioning(scene, Nx, Ny, f_max, f_min=0):
    Nx, Ny = cell_count(Nx), cell_count(Ny)
    if not np.isfinite([f_max, f_min]).all() or f_max <= 0 or not 0 <= f_min <= f_max:
        raise ValueError("Require 0 <= f_min <= f_max and f_max > 0")
    return np.array(
        [
            np.log(scene.Lx * f_max / C0),
            np.log(scene.Ly * f_max / C0),
            f_min / f_max,
            np.log(Nx),
            np.log(Ny),
        ],
        dtype=np.float32,
    )


def save_model(
    path,
    model,
    *,
    raster_shape=(128, 128),
    alpha=4.0,
    training_commit="untrained",
    dataset_version="untrained",
):
    metadata = dict(
        format_version=2,
        mesh_policy=MESH_POLICY,
        architecture={"name": "ResUNet", "width": model.width},
        input_channels=CHANNELS,
        normalization=NORMALIZATION,
        raster_shape=list(raster_shape),
        conditioning=CONDITIONING,
        pooling={"kind": "log_mean_exp_logits", "alpha": alpha},
        output="positive_axis_density_softplus_eps_1e-6",
        training_commit=training_commit,
        dataset_version=dataset_version,
    )
    _validate_metadata(metadata)
    torch.save({**metadata, "state_dict": model.state_dict()}, Path(path))


def _validate_metadata(data):
    if not isinstance(data, dict):
        raise ValueError("Mesh checkpoint must contain a metadata dictionary")
    expected = {
        "format_version": 2,
        "mesh_policy": MESH_POLICY,
        "input_channels": CHANNELS,
        "normalization": NORMALIZATION,
        "conditioning": CONDITIONING,
        "output": "positive_axis_density_softplus_eps_1e-6",
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise ValueError(f"Incompatible mesh checkpoint {key}; expected {value!r}")
    try:
        if data["architecture"]["name"] != "ResUNet":
            raise ValueError("Unsupported mesh architecture")
        H, W = data["raster_shape"]
        if any(isinstance(v, bool) or int(v) != v or v < 4 for v in (H, W)):
            raise ValueError("Invalid checkpoint raster shape")
        pool = data["pooling"]
        if (
            pool["kind"] != "log_mean_exp_logits"
            or not math.isfinite(pool["alpha"])
            or pool["alpha"] <= 0
        ):
            raise ValueError("Invalid checkpoint pooling")
        for key in ("training_commit", "dataset_version"):
            if not isinstance(data[key], str):
                raise ValueError(f"Invalid checkpoint {key}")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Incomplete mesh checkpoint metadata: {exc}") from exc


def load_model(path, device="cpu"):
    data = torch.load(Path(path), map_location=device, weights_only=True)
    _validate_metadata(data)
    model = ResUNet(data["architecture"]["width"]).to(device)
    model.load_state_dict(data["state_dict"], strict=True)
    model.eval()
    return model, {key: value for key, value in data.items() if key != "state_dict"}


def infer_mesh(scene, model, metadata, Nx, Ny, f_max, f_min=0, **constraints):
    _validate_metadata(metadata)
    device = next(model.parameters()).device
    condition = conditioning(scene, Nx, Ny, f_max, f_min)
    if scene.pml is not None:
        scene.pml.validate_scene(scene)
        if "x_collar" in constraints or "y_collar" in constraints:
            raise ValueError("PML collars are controlled by the scene")
        constraints.update(x_collar=scene.pml.x, y_collar=scene.pml.y)
    inputs = rasterize(scene, metadata["raster_shape"], f_max)
    with torch.inference_mode():
        logits = model(
            torch.from_numpy(inputs[None]).to(device), torch.from_numpy(condition[None]).to(device)
        )
        rx, ry = pool_axes(logits, metadata["pooling"]["alpha"])
    return density_mesh(
        scene.Lx,
        scene.Ly,
        Nx,
        Ny,
        rx[0].cpu().numpy(),
        ry[0].cpu().numpy(),
        x_anchors=sorted(scene.x_anchors),
        y_anchors=sorted(scene.y_anchors),
        **constraints,
    )


def repair_loss(rho_x, rho_y, meshes, *, x_collars=None, y_collars=None):
    """Detached repaired-density CDF targets; an auxiliary loss, not FDTD backprop.

    Ignore fixed collar pixels by normalizing predicted mass over the learned
    physical interval. Multiplying a predicted density by a constant changes no loss.
    """
    if rho_x.ndim != 2 or rho_y.ndim != 2 or len(meshes) != len(rho_x) or len(meshes) != len(rho_y):
        raise ValueError("Expected batched axis densities and one repaired mesh per sample")
    losses = []
    for rho, axis, collars in ((rho_x, "x", x_collars), (rho_y, "y", y_collars)):
        collars = [AxisCollar()] * len(meshes) if collars is None else collars
        if len(collars) != len(meshes):
            raise ValueError("One collar definition is required per sample")
        if not torch.isfinite(rho).all() or not (rho > 0).all():
            raise ValueError("Predicted densities must be positive and finite")
        targets, masks = [], []
        for mesh, collar in zip(meshes, collars):
            lines = getattr(mesh, axis)
            targets.append(projected_density(lines, rho.shape[1], collar=collar))
            edges = np.linspace(0, lines[-1], rho.shape[1] + 1)
            masks.append(
                np.maximum(
                    0,
                    np.minimum(edges[1:], lines[-1] - collar.thickness)
                    - np.maximum(edges[:-1], collar.thickness),
                )
            )
        target = torch.as_tensor(np.array(targets), device=rho.device, dtype=rho.dtype).detach()
        mask = torch.as_tensor(np.array(masks), device=rho.device, dtype=rho.dtype)
        probability = rho * mask
        probability = probability / probability.sum(dim=1, keepdim=True)
        losses.append((probability.cumsum(1) - target.cumsum(1)).square().mean())
    return sum(losses) / 2
