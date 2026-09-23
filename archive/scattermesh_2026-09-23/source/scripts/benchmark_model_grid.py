#!/usr/bin/env python3
"""Measure FP32 training-step memory and throughput for the 3x3 model grid."""

import argparse
import json
import time

import torch

from scattermesh.model import AxisDensityUNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    device_index = 0 if device.index is None else device.index
    torch.cuda.set_device(device_index)
    batches = {128: 32, 256: 12, 384: 6}
    records = []
    for resolution in (128, 256, 384):
        for base_channels in (16, 24, 32):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = AxisDensityUNet(base_channels=base_channels).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            batch = batches[resolution]
            raster = torch.randn(batch, 5, resolution, resolution, device=device)
            conditioning = torch.randn(batch, 10, device=device)
            target = torch.softmax(torch.randn(batch, 2, resolution, device=device), dim=-1)
            elapsed = []
            for step in range(args.steps + 1):
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                prediction = model(raster, conditioning)
                loss = (prediction.cumsum(-1) - target.cumsum(-1)).square().mean()
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize(device)
                if step:
                    elapsed.append(time.perf_counter() - started)
            records.append(
                {
                    "resolution": resolution,
                    "base_channels": base_channels,
                    "batch_size": batch,
                    "parameter_count": sum(value.numel() for value in model.parameters()),
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "mean_step_seconds": sum(elapsed) / len(elapsed),
                    "examples_per_second": batch * len(elapsed) / sum(elapsed),
                }
            )
            del model, optimizer, raster, conditioning, target, prediction, loss
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(device),
                "total_memory_gib": torch.cuda.get_device_properties(device).total_memory / 2**30,
                "precision": "float32",
                "records": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
