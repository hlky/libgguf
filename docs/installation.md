# Installation

libgguf is a Python package built with scikit-build-core and CMake. A normal
installation builds the native CPU extension and installs the Python package.

```bash
python -m pip install .
```

For development, install it editable so the Python sources remain connected to
the working tree:

```bash
python -m pip install --editable .
```

## GPU build

The current GPU build uses HIP/ROCm. The caller must provide the ROCm SDK tools,
a HIP-capable compiler, and an appropriate GPU architecture. On Windows the
ROCm SDK `rocm-sdk` command and Ninja must be available on `PATH`.

The Torch wrapper additionally requires a ROCm-enabled Torch installation in
the build environment. Disable build isolation so CMake can discover that
installation:

```powershell
$env:CMAKE_ARGS='-DLIBGGUF_BUILD_GPU=ON -DLIBGGUF_ENABLE_ROCM=ON -DLIBGGUF_BUILD_TORCH=ON'
python -m pip install --editable ".[gpu]" --no-build-isolation
```

The installed wrapper is `libgguf.libgguf_gpu`. PyTorch uses its `cuda` device
namespace for ROCm tensors, so tensors are still created with `device="cuda"`.

## CMake build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

| Option | Default | Meaning |
| --- | --- | --- |
| `LIBGGUF_BUILD_SHARED` | `ON` | Build the standalone shared library. |
| `LIBGGUF_BUILD_PYTHON` | `ON` | Build the native Python extension. |
| `LIBGGUF_BUILD_TOOLS` | `ON` | Build the native converter. |
| `LIBGGUF_BUILD_BENCHMARKS` | `OFF` | Build the native CPU benchmark. |
| `LIBGGUF_BUILD_GPU` | `OFF` | Build generic GPU kernels and conversion support. |
| `LIBGGUF_ENABLE_ROCM` | `OFF` | Compile GPU targets with HIP/ROCm. |
| `LIBGGUF_ENABLE_CUDA` | `OFF` | Compile GPU targets with CUDA. |
| `LIBGGUF_BUILD_TORCH` | `OFF` | Build the Torch GPU extension; requires `LIBGGUF_BUILD_GPU=ON`. |
| `LIBGGUF_CPU_BACKEND` | `REF` | Select `REF`, `SSE2`, `SSE4_1`, or `AVX2` at compile time. |

CPU backend selection is compile-time. There is no runtime CPU feature
dispatch. GPU builds must select exactly one GPU backend.
