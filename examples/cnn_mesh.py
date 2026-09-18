"""Use --checkpoint for a trained compatible model; otherwise smoke-test random weights."""

import argparse
from pathlib import Path

import torch

from fdtdmesh import FDTD_2D_Ez
from fdtdmesh.ml import ResUNet, save_model

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path)
args = parser.parse_args()
if args.checkpoint is None:
    torch.manual_seed(42)
    args.checkpoint = Path("artifacts/untrained_demo.pt")
    args.checkpoint.parent.mkdir(exist_ok=True)
    save_model(args.checkpoint, ResUNet(), raster_shape=(64, 64))
    print("Smoke test only: random weights do not imply learned mesh quality.")
sim = FDTD_2D_Ez(0.02, 0.015, 80, 60, 20e9, t_end=1e-9)
glass = sim.add_material("glass", epsilon_r=4)
sim.add_circle(glass, center=(0.011, 0.0075), radius=0.002)
sim.add_source("point", x=0.003, y=0.0075, width=2e-11, delay=8e-11)
sim.add_receiver("point", x=0.017, y=0.0075)
sim.mesh_with_model(args.checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
print(sim.run().diagnostics)
