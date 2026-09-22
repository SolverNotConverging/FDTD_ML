"""Set FDTDMESH_BUILD_CUDA=1 to compile the optional native production backend."""

import os
import shutil
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CudaBuild(build_ext):
    def build_extensions(self):
        nvcc = shutil.which("nvcc")
        if not nvcc:
            raise RuntimeError("nvcc is required; add your CUDA toolkit bin directory to PATH")
        cuda = Path(nvcc).resolve().parent.parent
        build = Path(self.build_temp).resolve()
        build.mkdir(parents=True, exist_ok=True)
        obj = build / ("fdtd_runtime.obj" if os.name == "nt" else "fdtd_runtime.o")
        command = [
            nvcc,
            "-c",
            "src/fdtdmesh/solver/cuda/fdtd_runtime.cu",
            "-o",
            str(obj),
            "-std=c++17",
            "-O3",
            "--fmad=false",
            "-gencode=arch=compute_75,code=sm_75",
            "-gencode=arch=compute_89,code=sm_89",
            "-gencode=arch=compute_89,code=compute_89",
        ]
        command += ["-Xcompiler", "/MD"] if os.name == "nt" else ["-Xcompiler", "-fPIC"]
        # setuptools initializes MSVC's environment when it starts compilation;
        # nvcc needs it earlier to locate cl.exe.
        if os.name == "nt":
            self.compiler.initialize()
            cl = shutil.which(self.compiler.cc, path=self.compiler._paths)
            if not cl and os.environ.get("VCToolsInstallDir"):
                cl_path = Path(os.environ["VCToolsInstallDir"]) / "bin/Hostx64/x64/cl.exe"
                if cl_path.is_file():
                    cl = str(cl_path)
                    self.compiler._paths = str(cl_path.parent) + os.pathsep + self.compiler._paths
                    self.compiler.cc = cl
                    self.compiler.linker = str(cl_path.with_name("link.exe"))
                    os.environ["PATH"] = self.compiler._paths
            sdk_bin = os.environ.get("WindowsSdkVerBinPath")
            if sdk_bin:
                self.compiler._paths = (
                    str(Path(sdk_bin) / "x64") + os.pathsep + self.compiler._paths
                )
                os.environ["PATH"] = self.compiler._paths
            if not cl:
                raise RuntimeError(
                    "MSVC is not in PATH. Run scripts/build_cuda.ps1 to load vcvars64."
                )
            command += ["-ccbin", str(Path(cl).parent)]
        subprocess.run(command, check=True)
        for ext in self.extensions:
            ext.extra_objects = [str(obj)]
            ext.library_dirs = [str(cuda / ("lib/x64" if os.name == "nt" else "lib64"))]
        super().build_extensions()


extensions = []
if os.environ.get("FDTDMESH_BUILD_CUDA") == "1":
    import numpy
    from Cython.Build import cythonize

    extensions = cythonize(
        [
            Extension(
                "fdtdmesh.solver.cuda_runtime",
                ["src/fdtdmesh/solver/cuda_runtime.pyx"],
                language="c++",
                include_dirs=[numpy.get_include(), "src/fdtdmesh/solver/cuda"],
                libraries=["cudart"],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            )
        ]
    )

setup(ext_modules=extensions, cmdclass={"build_ext": CudaBuild})
