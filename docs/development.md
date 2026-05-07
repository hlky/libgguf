# Development

## Build

Editable Python build:

```bash
python -m pip install -e ".[test,quantize]"
```

Native CMake build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Build native benchmarks:

```bash
cmake -S . -B build -DLIBGGUF_BUILD_BENCHMARKS=ON
cmake --build build --config Release
```

Control CUDA extension builds:

```bash
cmake -S . -B build -DLIBGGUF_BUILD_CUDA_KERNELS=ON
```

## Repository Layout

- `include/`: public C API.
- `csrc/`: native C++ implementation.
- `csrc/common/`: shared tables, CPU feature probing, storage helpers, and quantization helpers.
- `csrc/quant/`: scalar row quantizers plus x86 SIMD variants in `sse2/`, `sse4_1/`, and `avx2/`.
- `csrc/dequant/`: scalar row dequantizers, dispatch glue, and the same x86 SIMD variant layout.
- `csrc/quantize_gguf.cpp`: native GGUF quantization/conversion executable.
- `src/libgguf/`: public Python package, row APIs, conversion helpers, and inspection/validation CLIs.
- `src/libgguf/libgguf_numpy/`, `src/libgguf/libgguf_torch/`, `src/libgguf/libgguf_cuda/`: optional backend packages.
- `tests/`, `bench/`, `scripts/`, `docs/`: parity tests, benchmark drivers, maintenance scripts, and documentation.

Installed Python inspection CLIs are `gguf-inspect` and `gguf-validate`.

## CPU SIMD and Backend Selection

On x86 builds, CMake includes the SIMD source variants and applies per-source flags for `sse2`, `sse4_1`, and `avx2` files. Runtime code probes CPU features before using those implementations, so unsupported instruction sets fall back to a supported backend or `ref`. Backend-specific private hooks exist so tests can compare `ref` with supported SIMD implementations for byte/value parity.

Use runtime auto dispatch as the normal policy. Current defaults are hard-coded preferences by operation/qtype, not benchmark results measured on the user's CPU, so do not treat one benchmark machine as a universal compile-time choice.

If reproducibility or benchmarking needs backend pinning later, add an explicit documented override and keep auto dispatch as the default user path. If portable non-x86 or reference-only builds become a release target, add a CMake option to control SIMD source inclusion instead of replacing runtime dispatch.

## Tests

```bash
pytest
```

CUDA tests are skipped when CUDA/Torch extension support is unavailable.

Check generated docs:

```bash
python scripts/generate_support_matrix.py --check
```

## CI

The v1 GitHub Actions workflow is intentionally low-cost: Linux standard
CPU runners only, with no GPU jobs, larger runners, schedules, artifacts,
or caches by default.

For public repositories, standard GitHub-hosted runners are free. If the
repository is private, workflow minutes and storage draw from the repository
owner's quota and may bill overages to that owner.

## Benchmarks

Torch backend:

```bash
python bench/torch_bench.py --qtypes default --rows 1 --blocks-per-row 1
```

CUDA quantization:

```bash
python bench/cuda_quant_bench.py \
  --qtypes Q4_0,Q8_0,Q4_K \
  --shapes 64x4096,4096x4096,11008x4096 \
  --csv bench/results/local_cuda_quant.csv
```

Native benchmark binary is available only when `LIBGGUF_BUILD_BENCHMARKS=ON`.

## CUDA Resource Tracking

Track these for CUDA kernels:

- registers per thread;
- local stack bytes;
- shared memory;
- occupancy-relevant launch shape;
- traffic throughput;
- exactness against native CPU reference bytes.

Register and stack data usually comes from `ptxas` output or CUDA binary inspection tools, not from the Python benchmark directly.

## Optimization Principles

- Keep dequantization kernels stack-free.
- Preserve scalar accumulation order for byte-sensitive quantization searches.
- Use lane-subgroup patterns for exact K quant optimization where independent subgroup searches can fill lanes without changing search order.
- Use lookup tables for exact IQ neighbor search when they reduce repeated scalar work without changing chosen codes.
- Do not use warp reductions where reduction order changes encoded bytes unless exactness is proven.
- Treat byte equality as the primary signal for quantization optimization correctness.
