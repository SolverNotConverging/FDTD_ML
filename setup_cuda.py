"""Build the optional Cython binding and nvcc TMz device kernel in place."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent
CUDA_HOME = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
NVCC = CUDA_HOME / "bin" / "nvcc"
SOURCE = ROOT / "src" / "scattermesh"


class BuildCuda(build_ext):
    def build_extensions(self):
        if not NVCC.is_file():
            raise RuntimeError(f"nvcc is required at {NVCC}")
        self.build_temp = str(Path(self.build_temp).resolve())
        Path(self.build_temp).mkdir(parents=True, exist_ok=True)
        cuda_object = Path(self.build_temp) / "cuda_fdtd_kernels.o"
        subprocess.check_call(
            [
                str(NVCC),
                "-std=c++17",
                "-O3",
                "-Xcompiler",
                "-fPIC",
                "-gencode=arch=compute_75,code=sm_75",
                "-gencode=arch=compute_75,code=compute_75",
                "-I",
                str(SOURCE),
                "-c",
                str(SOURCE / "cuda_fdtd_kernels.cu"),
                "-o",
                str(cuda_object),
            ]
        )
        for extension in self.extensions:
            extension.extra_objects.append(str(cuda_object))
        self.force = True
        super().build_extensions()
        sources = ("_cuda_fdtd.pyx", "cuda_fdtd_kernels.h", "cuda_fdtd_kernels.cu")
        manifest = {
            "schema_version": 1,
            "architecture": "sm_75",
            "source_sha256": {
                name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in sources
            },
        }
        (SOURCE / "_cuda_fdtd_build.json").write_text(json.dumps(manifest, indent=2) + "\n")


extension = Extension(
    "scattermesh._cuda_fdtd",
    [str(SOURCE / "_cuda_fdtd.pyx")],
    include_dirs=[str(SOURCE), numpy.get_include(), str(CUDA_HOME / "include")],
    library_dirs=[str(CUDA_HOME / "lib64")],
    runtime_library_dirs=[str(CUDA_HOME / "lib64")],
    libraries=["cudart"],
    language="c++",
    extra_compile_args=["-O3", "-std=c++17"],
)

setup(
    name="scattermesh-cuda-kernel",
    ext_modules=cythonize([extension], compiler_directives={"language_level": 3}),
    cmdclass={"build_ext": BuildCuda},
)
