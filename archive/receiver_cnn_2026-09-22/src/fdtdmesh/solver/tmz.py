"""Backend discovery: CUDA is explicit, and never falls back to CPU stepping."""

import os
import shutil
from pathlib import Path

_dll_handles = []
if os.name == "nt":
    cuda = os.environ.get("CUDA_PATH")
    if not cuda and shutil.which("nvcc"):
        cuda = str(Path(shutil.which("nvcc")).parent.parent)
    if cuda:
        for folder in (Path(cuda) / "bin", Path(cuda) / "bin" / "x64"):
            if folder.is_dir():
                _dll_handles.append(os.add_dll_directory(str(folder)))


def cuda_backend():
    try:
        from . import cuda_runtime
    except ImportError as exc:
        raise RuntimeError(
            "CUDA extension unavailable. Run scripts/build_cuda.ps1 "
            "(see README for build requirements)."
        ) from exc
    if cuda_runtime.device_count() == 0:
        raise RuntimeError("No usable CUDA device/driver found")
    return cuda_runtime


def cuda_available():
    try:
        cuda_backend()
        return True
    except RuntimeError:
        return False
