# Installation

libgguf uses `scikit-build-core` with CMake >=3.18 and C++17. The core Python package depends on NumPy.

Editable core install:

```bash
python -m pip install -e .
```

Install Python conversion helper dependencies:

```bash
python -m pip install -e ".[quantize]"
```

Install Torch for the optional CUDA extension path:

```bash
python -m pip install -e ".[cuda]"
```

The optional extras are:

| Extra | Dependencies | Purpose |
| --- | --- | --- |
| `cuda` | `torch` | Build/use the optional Torch CUDA extension when CUDA tooling is present. |
| `quantize` | `gguf`, `safetensors`, `torch`, `tqdm` | Experimental/internal Python conversion helper and safetensors/ckpt loading. |
| `test` | `gguf`, `huggingface_hub`, `pytest`, `requests`, `safetensors`, `torch`, `tqdm` | Test runner and test-only script dependencies. |

## Native Build

CMake options exposed by the project:

| Option | Default | Meaning |
| --- | --- | --- |
| `LIBGGUF_BUILD_SHARED` | `ON` | Build the standalone shared library. |
| `LIBGGUF_BUILD_PYTHON` | `ON` | Build the Python native extension `_libgguf`. |
| `LIBGGUF_BUILD_TOOLS` | `ON` | Build native command-line tools such as `libgguf_quantize_gguf`. |
| `LIBGGUF_BUILD_BENCHMARKS` | `OFF` | Build native benchmark binaries. |
| `LIBGGUF_BUILD_CUDA_KERNELS` | `AUTO` | Build the optional `libgguf_cuda` Torch CUDA extension. Values: `AUTO`, `ON`, `OFF`. |
| `LIBGGUF_CPU_BACKEND` | `REF` | Native CPU row backend to compile. Values: `REF`, `SSE2`, `SSE4_1`, `AVX2`. |

Example explicit native build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLIBGGUF_BUILD_TOOLS=ON
cmake --build build --config Release
```

Optimized CPU backends are build-time choices, not runtime auto-dispatch. Use
`LIBGGUF_CPU_BACKEND=REF` for portable/reference builds, or choose an x86 SIMD
backend explicitly for wheels or local builds targeting machines with that
instruction set.

## CUDA Requirements

The CUDA extension builds only when:

- `torch` imports in the build Python environment.
- Torch CMake metadata is discoverable through `torch.utils.cmake_prefix_path`.
- `nvcc` is available.

With `LIBGGUF_BUILD_CUDA_KERNELS=AUTO`, the build skips CUDA when those requirements are missing. With `LIBGGUF_BUILD_CUDA_KERNELS=ON`, missing CUDA requirements are a build error.

The CUDA target is a Torch extension installed as `libgguf.libgguf_cuda._C_gguf`.

When installing through pip, build isolation can hide an already installed Torch from CMake. For an editable CUDA extension build, install the build backend and Torch first, then build without isolation:

```bash
python -m pip install scikit-build-core cmake ninja torch
python -m pip install -e ".[cuda]" --no-build-isolation \
  --config-settings=cmake.define.LIBGGUF_BUILD_CUDA_KERNELS=ON
```
