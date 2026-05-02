# SIMD Notes

## Build Model

`libgguf` keeps SIMD code in backend-specific translation units and builds one portable extension/shared library. Runtime dispatch chooses the best available backend after CPU feature checks.

- Scalar/reference code must remain available for every supported qtype.
- SSE2, SSE4.1, and AVX2 kernels live in files named like `csrc/quant/dequant_<qtype>_<backend>.cpp`.
- AVX2 files are compiled with `/arch:AVX2` on MSVC or `-mavx2` elsewhere.
- SSE2 and SSE4.1 files get per-source flags on non-MSVC builds. Do not add global `-msse*`, `-mavx2`, or `/arch:AVX2` flags.
- Keep source lists in `scripts/native_sources.py`; both `setup.py` and `scripts/build_libgguf.py` consume that list.

## Correctness

SIMD kernels are performance implementations of the scalar format rules, not alternate numeric policies.

- Preserve byte-for-byte parity with the scalar reference for dequantization output and quantized bytes where tests assert exact output.
- Match llama.cpp tie behavior and rounding order for quantizers. Do not "simplify" rounding unless tests and reference behavior are updated intentionally.
- Handle zero scales, sign extraction, nibble order, and high/low lane ordering explicitly.
- Keep `k` block-aligned assertions local to kernels and validate public inputs at the API boundary.
- Use unaligned loads/stores unless the caller contract guarantees alignment. NumPy buffers and byte views should not be assumed aligned.

## Performance Hints

The fastest backend is not always the widest backend.

- For simple, memory-bound formats, SSE2 or SSE4.1 can match or beat AVX2 because the extra lane width does not remove the memory bottleneck.
- AVX2 tends to pay off for lookup-heavy formats such as the IQ2/IQ3 family, where more work can be amortized per loaded block.
- Avoid per-element branches inside block loops. Prefer vector masks, table lookups, unpack/shuffle stages, and straight-line stores.
- Keep lookup tables hot and avoid large temporary arrays in inner loops. If a helper needs a tiny stack table, verify it does not dominate the loop.
- Hoist block-invariant scales, masks, signs, and constants out of sub-loops.
- Benchmark row-kernel speed separately from allocation and Python wrapper overhead. `scripts/bench_dequant.py` measures private backend hooks; `tools/bench_dequant.py` measures public API behavior.
- Benchmark both `LIBGGUF_NUM_THREADS=1` and default threading. Threading can hide or amplify row-kernel differences.

## Dispatch

- Feature detection belongs in `csrc/common/libgguf_cpu.*`.
- Dequant backend selection lives with the dequant dispatch code, and forced backend environment variables are for tests/debugging.
- Unsupported CPUs must fall back to scalar code without requiring rebuilds.
- Prefer measured defaults for each qtype. Do not assume `avx2` is best just because it is available.

## Review Checklist

Before merging SIMD work:

- Build the extension and standalone shared library.
- Run `python -m pytest tests/test_libgguf_reference.py`.
- Run a focused backend benchmark for the changed qtype with `scripts/bench_dequant.py`.
- Run `tools/bench_dequant.py` when public API throughput may change.
- Update `docs/benchmark.md` only when recording a new measured run; keep durable guidance here and in `docs/performance.md`.
