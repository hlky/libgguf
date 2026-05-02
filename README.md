# libgguf

Standalone CPython bindings and command-line helpers for minimized GGUF reference quantizers derived from llama.cpp.

The native extension builds as one portable artifact. On x86/x64 it includes runtime-dispatched scalar, SSE2, SSE4.1, and AVX2 dequantization paths for the supported GGML qtypes, plus SIMD quantization paths for Q4_0 and Q8_0 where the compiler and CPU support them. SSE and AVX2 are compiled only for isolated translation units, not as global build modes.

Supported quantization types currently include Q1_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, TQ1_0, TQ2_0, MXFP4, and NVFP4.

## Install

Use Python 3.10+ with a C++17 compiler.

```bash
python -m pip install -e . --no-build-isolation
```

The base package depends on NumPy. Direct safetensors-to-GGUF conversion also needs optional runtime packages. Install them through the project extra when developing locally:

```bash
python -m pip install -e ".[quantize]" --no-build-isolation
```

Install test dependencies with:

```bash
python -m pip install -e ".[test]" --no-build-isolation
```

## Commands

Build the Python extension in place:

```bash
python setup.py build_ext --inplace
```

Build the standalone C ABI shared library:

```bash
python scripts/build_libgguf.py
```

Run tests:

```bash
python -m pytest
```

Benchmark public dequantization API throughput:

```bash
python tools/bench_dequant.py --qtypes Q4_0,Q8_0,Q4_K --rows 4096 --cols 4096
```

Benchmark individual private dequantization backends:

```bash
python scripts/bench_dequant.py --qtype Q4_0 --backends ref,sse2,sse4_1,avx2 --rows 4096
```

See `docs/performance.md` for performance guidance, `docs/simd.md` for SIMD implementation notes, and `docs/benchmark.md` for the latest local backend sweep.

Run the conversion CLI through the module entry point or installed script:

```bash
python -m libgguf.quantize_gguf --src model.safetensors --dst model.gguf --qtype Q8_0
quantize-gguf --src model.safetensors --dst model.gguf --qtype Q8_0
```

## Python API

Quantize rows into a newly allocated bytes object:

```python
import libgguf

Q8_0 = 8
raw = libgguf.quantize_rows_raw(Q8_0, rows, rows.shape[0], rows.shape[1])
```

Or reuse a preallocated writable buffer:

```python
Q8_0 = 8
dst = bytearray(libgguf.row_size(Q8_0, rows.shape[1]) * rows.shape[0])
written = libgguf.quantize_rows_into_raw(Q8_0, rows, dst, rows.shape[0], rows.shape[1])
```

Dequantize encoded rows back to float32:

```python
decoded = libgguf.dequantize_rows(encoded, Q8_0)
```

## Layout

- `src/libgguf/` exposes the Python package and CLI.
- `include/libgguf.h` is the public C ABI header.
- `csrc/` contains the CPython extension shim and native quantizer implementation.
- `scripts/build_libgguf.py` builds the standalone shared library for callers that use the C ABI directly.
- `scripts/bench_dequant.py` benchmarks private dequantization backend hooks for row-kernel comparisons.
- `tools/bench_dequant.py` benchmarks the public Python dequantization APIs.
- `tests/` contains standalone package, C ABI, imatrix, and conversion tests.
