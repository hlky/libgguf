# Public API Backend Benchmark

## Summary

Run date: 2026-05-02 21:17:09

Environment:

- OS: Windows-10-10.0.19044-SP0
- Python: 3.12.10
- Logical processors: 36
- `LIBGGUF_NUM_THREADS`: unset, so libgguf used its default hardware thread count
- Repeats: 5 timed, 1 warmup
- Total benchmark wall time: 988.2s

Method:

- Dequantization uses public `dequantize_rows_into_raw` and `dequantize_rows` with each backend forced by `LIBGGUF_DEQUANT_<QTYPE>_BACKEND` before importing `libgguf`.
- Quantization backend forcing exists only for public `Q4_0` and `Q8_0`, using `LIBGGUF_Q4_0_BACKEND` and `LIBGGUF_Q8_0_BACKEND`.
- Imatrix values required by IQ qtypes are precomputed outside the timed region and passed explicitly.
- Throughput is based on decoded float32 bytes. `*_into` modes use preallocated buffers; `*_alloc` modes allocate output arrays on each call.
- Each qtype/backend run executes in a fresh Python process because native backend selection is cached after first use.

## Dequantization By Backend

### Small Tensors

| qtype | shape | ref into GiB/s | sse2 into GiB/s | sse4_1 into GiB/s | avx2 into GiB/s | best into | ref alloc GiB/s | sse2 alloc GiB/s | sse4_1 alloc GiB/s | avx2 alloc GiB/s | best alloc |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Q4_0 | 256x256 | 0.04 | 0.05 | 0.04 | 0.05 | avx2 | 0.04 | 0.05 | 0.04 | 0.05 | avx2 |
| Q4_1 | 256x256 | 0.05 | 0.05 | 0.05 | 0.05 | sse2 | 0.05 | 0.05 | 0.05 | 0.05 | ref |
| Q5_0 | 256x256 | 0.04 | 0.05 | 0.05 | 0.05 | sse2 | 0.05 | 0.05 | 0.05 | 0.05 | avx2 |
| Q5_1 | 256x256 | 0.04 | 0.05 | 0.04 | 0.05 | sse2 | 0.04 | 0.06 | 0.04 | 0.06 | avx2 |
| Q8_0 | 256x256 | 0.05 | 0.05 | 0.04 | 0.04 | sse2 | 0.05 | 0.06 | 0.05 | 0.05 | sse2 |
| Q2_K | 256x2048 | 0.38 | 0.47 | 0.44 | 0.46 | sse2 | 0.40 | 0.44 | 0.42 | 0.43 | sse2 |
| Q3_K | 256x2048 | 0.34 | 0.43 | 0.39 | 0.37 | sse2 | 0.38 | 0.42 | 0.40 | 0.40 | sse2 |
| Q4_K | 256x2048 | 0.45 | 0.49 | 0.50 | 0.47 | sse4_1 | 0.41 | 0.43 | 0.39 | 0.41 | sse2 |
| Q5_K | 256x2048 | 0.46 | 0.43 | 0.42 | 0.50 | avx2 | 0.40 | 0.44 | 0.40 | 0.43 | sse2 |
| Q6_K | 256x2048 | 0.43 | 0.49 | 0.42 | 0.49 | sse2 | 0.43 | 0.42 | 0.40 | 0.42 | ref |
| IQ2_XXS | 256x2048 | 0.50 | 0.46 | 0.44 | 0.42 | ref | 0.47 | 0.47 | 0.43 | 0.41 | sse2 |
| IQ2_XS | 256x2048 | 0.52 | 0.52 | 0.53 | 0.51 | sse4_1 | 0.50 | 0.49 | 0.53 | 0.48 | sse4_1 |
| IQ3_XXS | 256x2048 | 0.52 | 0.48 | 0.51 | 0.46 | ref | 0.47 | 0.44 | 0.50 | 0.41 | sse4_1 |
| IQ1_S | 256x2048 | 0.51 | 0.49 | 0.51 | 0.48 | ref | 0.45 | 0.44 | 0.46 | 0.42 | sse4_1 |
| IQ4_NL | 256x256 | 0.06 | 0.06 | 0.06 | 0.05 | sse4_1 | 0.06 | 0.06 | 0.06 | 0.05 | sse2 |
| IQ3_S | 256x2048 | 0.48 | 0.47 | 0.46 | 0.45 | ref | 0.44 | 0.41 | 0.45 | 0.41 | sse4_1 |
| IQ2_S | 256x2048 | 0.51 | 0.43 | 0.46 | 0.44 | ref | 0.46 | 0.40 | 0.44 | 0.43 | ref |
| IQ4_XS | 256x2048 | 0.53 | 0.53 | 0.51 | 0.47 | sse2 | 0.44 | 0.46 | 0.45 | 0.43 | sse2 |
| IQ1_M | 256x2048 | 0.52 | 0.55 | 0.53 | 0.51 | sse2 | 0.47 | 0.50 | 0.48 | 0.47 | sse2 |
| TQ1_0 | 256x2048 | 0.36 | 0.44 | 0.32 | 0.35 | sse2 | 0.41 | 0.44 | 0.34 | 0.35 | sse2 |
| TQ2_0 | 256x2048 | 0.31 | 0.34 | 0.32 | 0.34 | sse2 | 0.32 | 0.37 | 0.31 | 0.31 | sse2 |
| MXFP4 | 256x256 | 0.04 | 0.04 | 0.05 | 0.06 | avx2 | 0.04 | 0.04 | 0.06 | 0.05 | sse4_1 |
| NVFP4 | 256x512 | 0.12 | 0.12 | 0.10 | 0.11 | ref | 0.11 | 0.12 | 0.11 | 0.11 | sse2 |
| Q1_0 | 256x1024 | 0.18 | 0.24 | 0.17 | 0.17 | sse2 | 0.22 | 0.23 | 0.17 | 0.17 | sse2 |

### Medium Tensors

| qtype | shape | ref into GiB/s | sse2 into GiB/s | sse4_1 into GiB/s | avx2 into GiB/s | best into | ref alloc GiB/s | sse2 alloc GiB/s | sse4_1 alloc GiB/s | avx2 alloc GiB/s | best alloc |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Q4_0 | 1024x1024 | 0.73 | 0.89 | 0.94 | 0.78 | sse4_1 | 0.77 | 0.79 | 0.83 | 0.67 | sse4_1 |
| Q4_1 | 1024x1024 | 0.67 | 0.78 | 0.64 | 0.51 | sse2 | 0.72 | 0.81 | 0.65 | 0.50 | sse2 |
| Q5_0 | 1024x1024 | 0.73 | 0.82 | 0.73 | 0.84 | avx2 | 0.77 | 0.76 | 0.80 | 0.77 | sse4_1 |
| Q5_1 | 1024x1024 | 0.72 | 0.87 | 0.65 | 0.83 | sse2 | 0.78 | 0.81 | 0.75 | 0.73 | sse2 |
| Q8_0 | 1024x1024 | 0.88 | 0.62 | 0.84 | 0.60 | ref | 0.77 | 0.57 | 0.77 | 0.53 | ref |
| Q2_K | 1024x1024 | 0.80 | 0.85 | 0.91 | 0.81 | sse4_1 | 0.75 | 0.77 | 0.75 | 0.71 | sse2 |
| Q3_K | 1024x1024 | 0.83 | 0.58 | 0.72 | 0.57 | ref | 0.81 | 0.51 | 0.67 | 0.62 | ref |
| Q4_K | 1024x1024 | 0.84 | 0.87 | 0.87 | 0.86 | sse2 | 0.78 | 0.75 | 0.79 | 0.75 | sse4_1 |
| Q5_K | 1024x1024 | 0.84 | 0.93 | 0.92 | 0.85 | sse2 | 0.81 | 0.86 | 0.82 | 0.77 | sse2 |
| Q6_K | 1024x1024 | 0.90 | 0.86 | 0.82 | 0.82 | ref | 0.80 | 0.74 | 0.72 | 0.78 | ref |
| IQ2_XXS | 1024x1024 | 0.98 | 1.02 | 1.04 | 0.99 | sse4_1 | 0.88 | 0.91 | 0.93 | 0.86 | sse4_1 |
| IQ2_XS | 1024x1024 | 0.99 | 1.01 | 1.07 | 1.00 | sse4_1 | 0.81 | 0.90 | 0.88 | 0.85 | sse2 |
| IQ3_XXS | 1024x1024 | 0.93 | 0.97 | 1.01 | 0.94 | sse4_1 | 0.79 | 0.85 | 0.92 | 0.83 | sse4_1 |
| IQ1_S | 1024x1024 | 0.90 | 0.91 | 0.90 | 0.83 | sse2 | 0.80 | 0.80 | 0.77 | 0.78 | sse2 |
| IQ4_NL | 1024x1024 | 0.87 | 0.92 | 0.98 | 0.88 | sse4_1 | 0.79 | 0.80 | 0.88 | 0.79 | sse4_1 |
| IQ3_S | 1024x1024 | 0.88 | 0.92 | 0.93 | 0.93 | avx2 | 0.84 | 0.85 | 0.79 | 0.83 | sse2 |
| IQ2_S | 1024x1024 | 0.84 | 0.90 | 0.87 | 0.81 | sse2 | 0.78 | 0.82 | 0.79 | 0.74 | sse2 |
| IQ4_XS | 1024x1024 | 0.85 | 0.88 | 0.86 | 0.79 | sse2 | 0.78 | 0.76 | 0.78 | 0.78 | ref |
| IQ1_M | 1024x1024 | 0.96 | 0.99 | 0.99 | 0.92 | sse2 | 0.84 | 0.85 | 0.85 | 0.81 | sse4_1 |
| TQ1_0 | 1024x1024 | 0.66 | 0.57 | 0.87 | 0.70 | sse4_1 | 0.70 | 0.63 | 0.77 | 0.72 | sse4_1 |
| TQ2_0 | 1024x1024 | 0.80 | 0.85 | 0.80 | 0.68 | sse2 | 0.76 | 0.75 | 0.78 | 0.71 | sse4_1 |
| MXFP4 | 1024x1024 | 0.87 | 0.59 | 0.89 | 0.77 | sse4_1 | 0.79 | 0.61 | 0.79 | 0.72 | sse4_1 |
| NVFP4 | 1024x1024 | 0.87 | 0.80 | 0.84 | 0.89 | avx2 | 0.78 | 0.79 | 0.77 | 0.78 | sse2 |
| Q1_0 | 1024x1024 | 0.92 | 0.82 | 0.71 | 0.61 | ref | 0.79 | 0.76 | 0.65 | 0.70 | ref |

### Large Tensors

| qtype | shape | ref into GiB/s | sse2 into GiB/s | sse4_1 into GiB/s | avx2 into GiB/s | best into | ref alloc GiB/s | sse2 alloc GiB/s | sse4_1 alloc GiB/s | avx2 alloc GiB/s | best alloc |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Q4_0 | 4096x4096 | 13.07 | 13.65 | 13.39 | 12.97 | sse2 | 4.58 | 4.48 | 4.43 | 4.27 | ref |
| Q4_1 | 4096x4096 | 12.31 | 13.35 | 11.36 | 13.31 | sse2 | 4.37 | 4.41 | 3.81 | 4.34 | sse2 |
| Q5_0 | 4096x4096 | 13.06 | 13.18 | 14.10 | 12.67 | sse4_1 | 4.44 | 4.52 | 4.45 | 4.26 | sse2 |
| Q5_1 | 4096x4096 | 13.48 | 12.26 | 13.32 | 13.19 | ref | 4.34 | 4.48 | 4.54 | 3.93 | sse4_1 |
| Q8_0 | 4096x4096 | 13.43 | 13.19 | 12.90 | 12.58 | ref | 4.45 | 4.38 | 4.39 | 4.28 | ref |
| Q2_K | 4096x4096 | 14.04 | 15.23 | 14.79 | 13.99 | sse2 | 3.88 | 4.43 | 4.37 | 4.24 | sse2 |
| Q3_K | 4096x4096 | 13.09 | 14.19 | 13.99 | 13.42 | sse2 | 4.36 | 4.33 | 3.82 | 4.10 | ref |
| Q4_K | 4096x4096 | 14.80 | 15.35 | 14.25 | 14.76 | sse2 | 4.44 | 4.47 | 4.50 | 4.35 | sse4_1 |
| Q5_K | 4096x4096 | 14.55 | 14.26 | 14.91 | 14.83 | sse4_1 | 4.46 | 3.82 | 3.98 | 4.36 | ref |
| Q6_K | 4096x4096 | 14.18 | 15.45 | 15.06 | 13.85 | sse2 | 4.44 | 4.49 | 4.46 | 4.36 | sse2 |
| IQ2_XXS | 4096x4096 | 10.32 | 13.15 | 13.09 | 13.50 | avx2 | 4.29 | 4.37 | 4.43 | 4.36 | sse4_1 |
| IQ2_XS | 4096x4096 | 9.97 | 13.07 | 13.35 | 12.70 | sse4_1 | 4.26 | 4.40 | 4.47 | 4.41 | sse4_1 |
| IQ3_XXS | 4096x4096 | 10.01 | 12.81 | 12.89 | 12.98 | avx2 | 4.20 | 4.36 | 4.38 | 4.23 | sse4_1 |
| IQ1_S | 4096x4096 | 15.24 | 14.53 | 15.03 | 14.02 | ref | 4.44 | 4.43 | 4.04 | 4.33 | ref |
| IQ4_NL | 4096x4096 | 14.39 | 14.83 | 14.55 | 12.97 | sse2 | 4.46 | 4.39 | 4.46 | 4.26 | ref |
| IQ3_S | 4096x4096 | 9.86 | 13.01 | 12.63 | 13.71 | avx2 | 4.29 | 4.47 | 4.39 | 4.31 | sse2 |
| IQ2_S | 4096x4096 | 10.06 | 13.79 | 13.54 | 12.85 | sse2 | 4.22 | 4.48 | 4.41 | 4.25 | sse2 |
| IQ4_XS | 4096x4096 | 14.54 | 13.92 | 14.23 | 13.76 | ref | 4.36 | 4.02 | 4.49 | 3.86 | sse4_1 |
| IQ1_M | 4096x4096 | 13.79 | 13.40 | 13.74 | 13.14 | ref | 4.52 | 3.87 | 4.45 | 4.28 | ref |
| TQ1_0 | 4096x4096 | 13.82 | 13.85 | 12.88 | 12.83 | sse2 | 4.47 | 4.45 | 4.45 | 4.31 | ref |
| TQ2_0 | 4096x4096 | 14.26 | 13.75 | 14.33 | 13.11 | sse4_1 | 4.47 | 4.45 | 4.47 | 4.42 | ref |
| MXFP4 | 4096x4096 | 14.13 | 13.95 | 14.24 | 13.17 | sse4_1 | 4.43 | 4.37 | 4.45 | 4.32 | sse4_1 |
| NVFP4 | 4096x4096 | 12.48 | 12.24 | 12.93 | 11.35 | sse4_1 | 4.36 | 4.43 | 4.40 | 4.40 | sse2 |
| Q1_0 | 4096x4096 | 8.15 | 13.56 | 13.63 | 11.92 | sse4_1 | 4.18 | 4.52 | 4.41 | 4.38 | sse2 |

## Quantization Backend Controls

Only `Q4_0` and `Q8_0` expose public quantization backend forcing in this build.

### Small Tensors

| qtype | shape | ref into GiB/s | sse2 into GiB/s | avx2 into GiB/s | best into | ref alloc GiB/s | sse2 alloc GiB/s | avx2 alloc GiB/s | best alloc |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| Q4_0 | 256x256 | 0.06 | 0.05 | 0.04 | ref | 0.06 | 0.05 | 0.05 | ref |
| Q8_0 | 256x256 | 0.05 | 0.05 | 0.04 | sse2 | 0.05 | 0.06 | 0.05 | sse2 |

### Medium Tensors

| qtype | shape | ref into GiB/s | sse2 into GiB/s | avx2 into GiB/s | best into | ref alloc GiB/s | sse2 alloc GiB/s | avx2 alloc GiB/s | best alloc |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| Q4_0 | 1024x1024 | 0.76 | 0.93 | 0.73 | sse2 | 0.83 | 0.89 | 0.80 | sse2 |
| Q8_0 | 1024x1024 | 0.82 | 0.64 | 0.53 | ref | 0.82 | 0.75 | 0.72 | ref |

### Large Tensors

| qtype | shape | ref into GiB/s | sse2 into GiB/s | avx2 into GiB/s | best into | ref alloc GiB/s | sse2 alloc GiB/s | avx2 alloc GiB/s | best alloc |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| Q4_0 | 4096x4096 | 12.30 | 12.73 | 12.62 | sse2 | 9.57 | 10.45 | 10.62 | avx2 |
| Q8_0 | 4096x4096 | 3.15 | 13.49 | 10.79 | sse2 | 2.21 | 9.83 | 9.35 | sse2 |
