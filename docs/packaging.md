# Packaging Notes

libgguf is packaged with `scikit-build-core`. Python installs drive the CMake
build, then install the Python package, native `_libgguf` extension, native
`libgguf_quantize_gguf` executable, and, when available, the optional CUDA Torch
extension.

## Editable And Wheel Builds

Editable installs and local wheels use the same CMake options documented in
[Installation](installation.md):

```bash
python -m pip install -e .
python -m pip wheel .
```

The default build is CPU-first. It builds the core Python extension and native
tooling without requiring Torch or CUDA. Native benchmarks are disabled by
default and are intended for explicit local builds.

Backend tests live under `tests/backends/` in the source tree and are not
packaged into wheels.

## Optional CUDA Builds

The CUDA extension is optional and controlled by
`LIBGGUF_BUILD_CUDA_KERNELS=AUTO|ON|OFF`. With the default `AUTO`, CMake builds
`libgguf.libgguf_cuda._C_gguf` only when Torch, Torch CMake metadata, and `nvcc`
are visible in the build environment. Use `ON` when the extension is required;
missing CUDA requirements then fail the build instead of silently producing a
CPU-only install.

PEP 517 build isolation can hide an already installed Torch package from CMake.
For local editable CUDA builds, install the build backend and Torch first, then
disable isolation:

```bash
python -m pip install scikit-build-core cmake ninja torch
python -m pip install -e ".[cuda]" --no-build-isolation \
  --config-settings=cmake.define.LIBGGUF_BUILD_CUDA_KERNELS=ON
```

## Current Limitations

- Wheels are local/source-built today; there is no committed multi-platform
  wheel publishing pipeline.
- CPU builds are the packaging baseline. CUDA-enabled wheels depend on the local
  Torch/CUDA toolchain and are not yet produced as release artifacts.
- The low-cost CI path is intended to validate source installs, generated docs,
  exactness smoke checks, and the CPU test suite. It does not exercise GPU
  runners, publish artifacts, or build a platform wheel matrix.
