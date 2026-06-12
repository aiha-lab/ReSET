"""Build script for the `nvfp4r` C++/CUDA extension.

We compile a single shared object `nvfp4r._C` that registers all ops
under the `torch.ops.nvfp4r` namespace via TORCH_LIBRARY. The Python layer in
`python/nvfp4r/ops.py` only forwards calls so that vLLM / TRT-LLM can adopt
the same `torch.ops.nvfp4r.*` entry points without going through Python.
"""

from __future__ import annotations

import glob
import os
import re

from setuptools import setup
import torch.utils.cpp_extension as _torch_cpp_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _discover_cuda_home() -> str | None:
    """Pick a CUDA toolkit: the canonical ``/usr/local/cuda`` symlink, else the
    newest ``/usr/local/cuda-<ver>`` install."""
    if os.path.isdir("/usr/local/cuda"):
        return "/usr/local/cuda"
    versioned = [d for d in glob.glob("/usr/local/cuda-*") if os.path.isdir(d)]
    versioned.sort(key=lambda d: tuple(int(n) for n in re.findall(r"\d+", d)),
                   reverse=True)
    return versioned[0] if versioned else None


# Respect a user-provided CUDA_HOME / CUDA_PATH; otherwise auto-discover one.
# (PyTorch can also infer this from ``nvcc`` on PATH, so this is a fallback.)
if not (os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")):
    _cuda_home = _discover_cuda_home()
    if _cuda_home:
        os.environ["CUDA_HOME"] = _cuda_home
        os.environ["CUDA_PATH"] = _cuda_home

if os.environ.get("NVFP4R_SKIP_CUDA_VERSION_CHECK", "1") == "1":
    _torch_cpp_ext._check_cuda_version = lambda *a, **k: None  # type: ignore[attr-defined]

ROOT = os.path.dirname(os.path.abspath(__file__))
CSRC_DIR = os.path.join(ROOT, "csrc")
CUDA_DIR = os.path.join(ROOT, "cuda")


GEMV_ONLY = os.environ.get("NVFP4R_GEMV_ONLY", "0") == "1"

# Kernels that use tcgen05.mma.block16 (GEMM) require a ptxas that
# understands those modifiers. On hosts where that kernel fails to
# assemble, set NVFP4R_GEMV_ONLY=1 to build only the GEMV kernel and
# link a stub implementation for gemm instead.
_GEMM_CU_PATTERNS = ("cuda/gemm/",)


def _gather_sources() -> list[str]:
    cpp = sorted(glob.glob(os.path.join(CSRC_DIR, "**", "*.cpp"), recursive=True))
    cu  = sorted(glob.glob(os.path.join(CUDA_DIR, "**", "*.cu"),  recursive=True))

    if GEMV_ONLY:
        cu = [p for p in cu if not any(pat in p for pat in _GEMM_CU_PATTERNS)]
    else:
        # Real gemm CUDA kernel is present -> drop the stub that
        # would otherwise multiply-define its symbol.
        cpp = [p for p in cpp if not p.endswith("gemm_stubs.cpp")]

    srcs = cpp + cu
    return [os.path.relpath(p, ROOT) for p in srcs]


NVCC_FLAGS = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "-gencode=arch=compute_100a,code=sm_100a",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-Xcompiler=-fno-strict-aliasing",
    "-Xcompiler=-fvisibility=hidden",
    "-DNVFP4R_TARGET_SM=100",
]

CXX_FLAGS = [
    "-O3",
    "-std=c++17",
    "-fvisibility=hidden",
]


ext_modules = [
    CUDAExtension(
        name="nvfp4r._C",
        sources=_gather_sources(),
        include_dirs=[CSRC_DIR, CUDA_DIR],
        libraries=["cuda"],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
        },
    ),
]


setup(
    name="nvfp4r",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
