# SIMD Notes

## Build Model

`libgguf` keeps SIMD code in backend-specific translation units and builds one portable extension/shared library. Runtime dispatch chooses the best available backend after CPU feature checks.

- Scalar/reference code must remain available for every supported qtype.
- SSE2, SSE4.1, and AVX2 kernels live in backend-specific directories named like `csrc/dequant/<backend>/<qtype>.cpp` or `csrc/quant/<backend>/<qtype>.cpp`.
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

## Implementation Shape

Use the smallest structure that keeps the backend code reviewable without hiding CPU-specific details.

- Tiny qtype-specific kernels may keep the full implementation in each backend translation unit when the body is short and the ISA-specific code is easier to read directly.
- Use a private shared `*_simd.h` helper when SSE2, SSE4.1, and AVX2 would otherwise duplicate a substantial scalar-order algorithm. The backend `.cpp` files should only define the ISA macro or pass the backend primitive.
- Use common helper dispatch only for helpers shared by multiple quantizers, such as `make_qx_quants`, `make_q3_quants`, `make_qkx2/3_quants`, and `make_qp_quants`.
- Keep SIMD intrinsics out of generic scalar files. Runtime selection can live beside the qtype dispatcher, but ISA code stays in backend-specific translation units.

## Dequant-to-Quant Replacement Map

The dequant SIMD kernels are the safest guide for quant SIMD work because forced SSE2, SSE4.1, and AVX2 dequant backends are compared byte-for-byte against `ref` output for every supported qtype in `tests/test_libgguf_reference.py`.

Safe patterns to reuse in quantizers:

- Pure unpack, high-bit merge, sign extension, nibble extraction, and final byte packing are safe to SIMD-vectorize when the scalar byte order is preserved.
- Per-lane affine math after scales are already chosen is safe: `q * d`, `q * d - m`, and equivalent independent lane transforms.
- Fixed small table lookups are safe when the selected index is already known; the IQ and FP4 dequant kernels use scalar index extraction plus SIMD widen/multiply/store.
- Ternary bit/base-3 expansion is safe when the packed byte formula stays identical. The classification step can be vectorized, but final packed bytes should keep the scalar trit order.
- K-quant post-helper packing is safe to SIMD-vectorize after `make_q*` helper decisions have produced the same intermediate `L`, `scales`, and `mins` arrays.

Patterns that should stay scalar or preserve scalar ordering:

- Reductions that feed encoded bytes, especially max/min tracking where signed zero can reach half encoding. Use scalar branch form when parity depends on `+0` versus `-0`.
- Search, tie-breaking, and weighted grid/table selection in IQ1/IQ2/IQ3. Dequant only expands selected grid ids; it does not prove a reordered SIMD search is safe.
- Non-finite quant inputs. Dequant has no equivalent search/classification behavior, so SIMD quantizers should fall back to ref or keep scalar handling where NaN/Inf can affect decisions.
- Alternate rounding policies. `nearest_int`, `roundf`, and `lroundf` call sites must match the exact scalar operation and tie behavior used by the reference.

## Quantization Support Boundaries

Quantization SIMD is considered supported only when the forced backend produces byte-identical blocks to `ref` on deterministic parity rows and is faster in focused benchmarks.

- Q1_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, MXFP4, NVFP4, TQ1_0, TQ2_0, and the K-quants have real SIMD row/helper paths where measured backends are selected by default.
- IQ4_NL and IQ4_XS have SIMD subpaths for known-safe table/index work, with scalar fallbacks for quality/search behavior that remains scalar-order-sensitive.
- IQ1, IQ2, and IQ3 quantizers intentionally keep the search and packing decisions scalar. Their dequant kernels validate unpack/table expansion patterns, but that does not make the weighted quantization search safe to vectorize without a separate byte-parity proof.

## Performance Hints

The fastest backend is not always the widest backend.

- For simple, memory-bound formats, SSE2 or SSE4.1 can match or beat AVX2 because the extra lane width does not remove the memory bottleneck.
- AVX2 tends to pay off for lookup-heavy formats such as the IQ2/IQ3 family, where more work can be amortized per loaded block.
- Avoid per-element branches inside block loops. Prefer vector masks, table lookups, unpack/shuffle stages, and straight-line stores.
- Keep lookup tables hot and avoid large temporary arrays in inner loops. If a helper needs a tiny stack table, verify it does not dominate the loop.
- Hoist block-invariant scales, masks, signs, and constants out of sub-loops.
- Benchmark row-kernel speed separately from allocation and Python wrapper overhead. `scripts/bench_dequant.py` and `scripts/bench_quant.py` measure private backend hooks; `tools/bench_dequant.py` measures public API behavior.
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
- Run a focused backend benchmark for the changed qtype with `scripts/bench_dequant.py` or `scripts/bench_quant.py`.
- Run `tools/bench_dequant.py` when public API throughput may change.
- Update `docs/benchmark.md` only when recording a new measured run; keep durable guidance here and in `docs/performance.md`.
