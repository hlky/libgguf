# Performance Notes

## Dequantization

Use `tools/bench_dequant.py` for public API throughput. It benchmarks both `dequantize_rows_into_raw` with a preallocated output and `dequantize_rows`, which allocates the output array on each call.

```powershell
.\.venv\Scripts\python.exe tools\bench_dequant.py --qtypes Q4_0,Q8_0,Q4_K --rows 4096 --cols 4096
```

Use `scripts/bench_dequant.py` for private backend comparison. It calls `_dequantize_for_backend` directly, so its results are row-kernel measurements rather than public API throughput.

```powershell
.\.venv\Scripts\python.exe setup.py build_ext --inplace
.\.venv\Scripts\python.exe scripts\bench_dequant.py --qtype Q4_0 --backends ref,sse2,sse4_1,avx2 --rows 4096 --iterations 25 --repetitions 3
```

The latest checked-in local backend sweep is in `docs/benchmark.md`. SIMD implementation and review guidance lives in `docs/simd.md`.

## Current Findings

- SIMD dequantization is generally faster than `ref`, but the best backend is workload- and qtype-dependent.
- SSE2 or SSE4.1 often wins for memory-bound/simple dequant kernels at larger row counts.
- AVX2 tends to be strongest for lookup-heavy IQ2/IQ3 kernels and some small-row workloads.
- Do not select defaults from CPU feature availability alone; use measured backend behavior and keep scalar fallback correctness as the baseline.
- Benchmark allocation cost separately from row-kernel cost. Public Python API numbers can move for reasons unrelated to SIMD kernel throughput.

## Review Checklist

When adding or changing SIMD quantizers:

- Keep AVX2 and SSE4.1 code in isolated translation units with per-source compiler flags.
- Keep SSE2 code in isolated translation units with per-source compiler flags on non-MSVC builds.
- Keep runtime feature detection in `csrc/common/libgguf_cpu.*`.
- Preserve byte-for-byte parity with the scalar reference, including tie behavior.
- Benchmark preallocated and allocating APIs separately.
- Benchmark `LIBGGUF_NUM_THREADS=1` and default threaded behavior separately.
- Prefer measured backend defaults over CPU-feature-only assumptions.
- Update `scripts/native_sources.py`, package qtype metadata, test qtype metadata, and benchmark qtype maps together when adding a backend or qtype.
- Follow `docs/simd.md` for SIMD-specific implementation hints.
