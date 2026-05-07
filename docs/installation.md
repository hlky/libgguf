# Installation

libgguf uses `scikit-build-core` with CMake >=3.18 and C++17. The core Python package depends on NumPy.

Editable core install:

```bash
python -m pip install -e .
```

Install conversion dependencies:

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
| `quantize` | `gguf`, `safetensors`, `torch` | Python conversion CLIs and safetensors/ckpt loading. |
| `test` | `pytest` | Test runner. |

## Native Build

CMake options exposed by the project:

| Option | Default | Meaning |
| --- | --- | --- |
| `LIBGGUF_BUILD_SHARED` | `ON` | Build the standalone shared library. |
| `LIBGGUF_BUILD_PYTHON` | `ON` | Build the Python native extension `_libgguf`. |
| `LIBGGUF_BUILD_TOOLS` | `ON` | Build native command-line tools such as `libgguf_quantize_gguf`. |
| `LIBGGUF_BUILD_BENCHMARKS` | `OFF` | Build native benchmark binaries. |
| `LIBGGUF_BUILD_CUDA_KERNELS` | `AUTO` | Build the optional `libgguf_cuda` Torch CUDA extension. Values: `AUTO`, `ON`, `OFF`. |

Example explicit native build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLIBGGUF_BUILD_TOOLS=ON
cmake --build build --config Release
```

## CUDA Requirements

The CUDA extension builds only when:

- `torch` imports in the build Python environment.
- Torch CMake metadata is discoverable through `torch.utils.cmake_prefix_path`.
- `nvcc` is available.

With `LIBGGUF_BUILD_CUDA_KERNELS=AUTO`, the build skips CUDA when those requirements are missing. With `LIBGGUF_BUILD_CUDA_KERNELS=ON`, missing CUDA requirements are a build error.

The CUDA target is a Torch extension installed as `libgguf.libgguf_cuda._C_gguf`.
