# AGENTS.md

## Project Overview

`libgguf` is a Python 3.10+ package that provides CPython bindings and command-line helpers for GGUF reference quantizers derived from llama.cpp. The package includes a native C++17 extension, Python conversion utilities, a standalone C ABI build, dequantization benchmark helpers, and tests for the extension, C ABI, SIMD backend parity, imatrix loading, and safetensors-to-GGUF conversion.

## Repository Layout

- `src/libgguf/` contains the Python package and CLI entry points.
- `include/libgguf.h` is the public C ABI header.
- `csrc/` contains the CPython extension shim and native quantizer implementation.
- `scripts/build_libgguf.py` builds the standalone shared library used by C ABI callers and tests.
- `scripts/bench_dequant.py` benchmarks private dequantization backend hooks for row-kernel comparisons.
- `tools/bench_dequant.py` benchmarks public Python dequantization API throughput.
- `docs/performance.md` records durable performance guidance and review criteria.
- `docs/simd.md` records SIMD implementation and review guidance.
- `docs/benchmark.md` records local benchmark runs and measured backend sweeps.
- `tests/` contains pytest coverage and local GGML type metadata used by the reference checks.
- `pyproject.toml`, `setup.py`, and `MANIFEST.in` define packaging and build behavior.
- `third_party/ComfyUI-GGUF` and `third_party/llama.cpp` are submodules used as upstream references; avoid editing vendored sources unless the task explicitly requires it.

## Environment Setup

Use Python 3.10 or newer with a working C++17 compiler.

Install the package for local development:

```bash
python -m pip install -e . --no-build-isolation
```

Install test dependencies:

```bash
python -m pip install -e ".[test]" --no-build-isolation
```

Install optional direct quantization dependencies when working on `quantize.py` or `quantize_gguf.py`:

```bash
python -m pip install -e ".[quantize,test]" --no-build-isolation
```

## Build Commands

Build the CPython extension in place:

```bash
python setup.py build_ext --inplace
```

Build the standalone shared library:

```bash
python scripts/build_libgguf.py
```

On Windows this creates `libgguf.dll`; on Linux `libgguf.so`; on macOS `libgguf.dylib`.

Native builds produce one portable artifact. On x86/x64, dequantization baseline, SSE2, SSE4.1, and AVX2 implementations are compiled as separate translation units where supported for the supported GGML qtypes. Q4_0 and Q8_0 quantization also have isolated SIMD translation units. SSE and AVX2 flags must be applied only to their source files. Do not add global SSE/AVX2 flags to the extension or shared-library builds.

## Test Commands

Run the full test suite:

```bash
python -m pytest
```

Run the native binding and C ABI tests:

```bash
python -m pytest tests/test_libgguf_reference.py
```

Run the direct GGUF conversion tests:

```bash
python -m pytest tests/test_quantize_gguf.py
```

`tests/test_quantize_gguf.py` skips unless `gguf`, `safetensors`, and `torch` are installed.

Run the benchmark smoke path for public dequantization APIs:

```bash
python tools/bench_dequant.py --qtypes Q4_0,Q8_0,Q4_K --rows 512 --cols 1024 --repeats 3 --warmup 1
```

Run a private backend comparison for one qtype:

```bash
python scripts/bench_dequant.py --qtype Q4_0 --backends ref,sse2,sse4_1,avx2 --rows 1024 --iterations 25 --repetitions 3
```

## Development Notes

- Keep native changes compatible with C++17 and the existing setuptools build flow.
- Preserve the `src/libgguf` package layout and keep native implementation code under `csrc/`.
- Prefer NumPy arrays that are contiguous and explicitly typed when crossing the Python/native boundary.
- Keep runtime SIMD dispatch portable: feature detection belongs in `csrc/common/libgguf_cpu.*`, SIMD kernels should live in isolated translation units, and dispatch should fall back cleanly to scalar code on unsupported CPUs and non-x86 platforms.
- Follow `docs/simd.md` when adding or changing SIMD kernels, including per-source flags, exact scalar parity, and backend benchmark expectations.
- Keep supported qtype metadata synchronized across `include/libgguf.h`, `src/libgguf/__init__.py`, `tests/ggml_types.py`, `scripts/bench_dequant.py`, and `tools/bench_dequant.py` when adding or removing a qtype.
- Keep AVX2 code behind per-source compiler flags (`/arch:AVX2` or `-mavx2`) rather than global build flags. The normal Python and shared-library builds should not require `LIBGGUF_AVX2`.
- Use private test/debug hooks for backend selection and per-backend parity checks instead of expanding the public C ABI unless a public API change is intentional.
- Preserve deterministic backend parity with the scalar reference; SIMD changes should be covered by `tests/test_libgguf_reference.py`.
- Keep quantization policy changes covered by focused pytest cases in `tests/test_quantize_gguf.py`.
- Keep extension or ABI changes covered by `tests/test_libgguf_reference.py`.
- Do not commit generated build outputs such as `build/`, `dist/`, extension binaries, shared libraries, egg-info, or pytest caches.

## CLI Smoke Test

After changing the conversion CLI, run a small conversion through the module entry point when optional dependencies are available:

```bash
python -m libgguf.quantize_gguf --src model.safetensors --dst model.gguf --qtype Q8_0
```

## .venv

Prefer `.venv` if it exists.
