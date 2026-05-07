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
