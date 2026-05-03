# Benchmark

## Quantization

### Non-IQ Quant Default Selection Sweep

Local sweep on 2026-05-03 after adding non-IQ explicit SIMD quant kernels and standardized benchmark output. This timed private row-kernel hooks where available and private environment-selected `quantize_rows_raw` paths for qtypes that do not expose a direct hook. IQ qtypes were intentionally omitted.

Command shape:

```powershell
.\.venv\Scripts\python.exe setup.py build_ext --inplace --force
.\.venv\Scripts\python.exe scripts\bench_quant.py --qtypes all --backends ref,sse2,sse4_1,avx2 --rows 1024 --iterations 20 --repetitions 3
.\.venv\Scripts\python.exe scripts\bench_quant.py --qtypes Q5_0,Q3_K,Q4_K,Q5_K --backends ref,sse2,sse4_1,avx2 --rows 2048 --iterations 20 --repetitions 2
.\.venv\Scripts\python.exe scripts\bench_common_quant.py --qtypes all --backends ref,sse2,sse4_1,avx2 --modes unweighted,weighted --rows 512 --iterations 20 --repetitions 3
```

Rows `1024`, mean throughput in GiB/s:

```text
qtype  ref  sse2 sse4_1 avx2 default
Q1_0   0.62 2.95 2.86   3.06 avx2
Q4_0   1.40 1.87 2.44   2.29 sse4_1
Q4_1   1.46 2.31 2.25   1.70 sse2
Q5_0   0.92 1.10 1.33   1.34 sse4_1
Q5_1   1.17 1.32 1.28   1.27 sse2
Q8_0   0.13 3.60 3.67   4.33 avx2
Q2_K   0.55 0.68 0.70   0.67 sse4_1
Q3_K   1.54 1.56 1.57   1.50 avx2
Q4_K   0.03 0.04 0.04   0.04 sse4_1
Q5_K   0.48 0.73 0.74   0.71 sse4_1
Q6_K   0.63 0.63 0.77   0.80 avx2
TQ1_0  1.70 1.74 1.56   1.51 sse2
TQ2_0  0.21 3.13 3.17   1.68 sse4_1
MXFP4  0.14 0.22 0.27   0.43 avx2
NVFP4  0.13 0.21 0.25   0.37 avx2
```

Confirmation rows `2048` were used for close decisions. `Q5_0` confirmed `sse4_1`; `Q4_K` and `Q5_K` confirmed `sse4_1`; `Q3_K` favored `avx2` at larger row count (`2.66 GiB/s` vs `2.62` for `sse2` and `2.60` for `sse4_1`), so its default follows the larger-row result.

Common helper dispatch stays `ref` by default. In the common-helper sweep, aggregate mean throughput was `ref=0.0639`, `sse2=0.0626`, `sse4_1=0.0621`, and `avx2=0.0605` GiB/s; normalized per-case scoring also favored `ref`.

### Simple and FP4 Backend Sweep

Local focused sweep on 2026-05-03 after wiring runtime dispatch for simple and FP4 quantizers. This times the private quantize backend hooks directly, not public API conversion overhead.

Command shape:

```powershell
.\.venv\Scripts\python.exe scripts\bench_quant.py --qtype <QTYPE> --rows 1024 --iterations 50 --repetitions 2 --backends ref,sse2,sse4_1,avx2
.\.venv\Scripts\python.exe scripts\bench_quant.py --qtype <QTYPE> --rows 4096 --iterations 100 --repetitions 3 --backends ref,sse2,sse4_1,avx2
```

Rows `1024`, mean throughput in GiB/s:

```text
qtype  ref  sse2 sse4_1 avx2 best
Q1_0   0.59 2.82 2.87   3.10 avx2
Q4_1   1.39 2.18 2.15   1.65 sse2
Q5_0   0.97 1.04 1.28   1.28 sse4_1
Q5_1   1.12 1.24 1.21   1.25 avx2
MXFP4  0.13 0.21 0.21   0.40 avx2
NVFP4  0.12 0.20 0.20   0.33 avx2
```

Rows `4096` confirmation sweep for backend selection:

```text
qtype ref  sse2 sse4_1 avx2 best
Q4_1  1.41 2.23 2.22   1.68 sse2
Q5_0  0.90 0.97 1.13   1.11 sse4_1
```

### K-Quant Backend Sweep

Focused subprocess sweep on 2026-05-03 using each qtype's private environment backend override. Rows `1024`, `n_per_row=1024`, `iterations=30`, `repetitions=2`; values are mean throughput in GiB/s.

```text
qtype ref  sse2 sse4_1 avx2 best
Q2_K  0.48 0.54 0.53   0.51 sse2
Q3_K  0.89 0.83 0.81   0.78 ref
Q4_K  0.44 0.49 0.45   0.45 sse2
Q5_K  0.46 0.54 0.52   0.52 sse2
Q6_K  0.54 0.57 0.55   0.54 sse2
```

### Existing Quant Dispatch Sweep

Focused private-hook sweep on 2026-05-03 for previously dispatched qtypes. Rows `512`, default `n_per_row`, `iterations=20`, `repetitions=2`; values are mean throughput in GiB/s.

```text
qtype  ref  sse2 sse4_1 avx2 best
Q4_0   1.36 1.73 2.28   2.14 sse4_1
Q8_0   0.13 3.53 3.65   4.40 avx2
TQ2_0  0.21 3.13 3.14   1.67 sse4_1
```

## Dequantization

### All Supported Dequant Backend Sweep

This sweep times the private backend hook directly for every qtype in `SUPPORTED_LIBGGUF_QTYPES`. It is intended for row-kernel comparison, not public API throughput. CPU backend support on this run was `ref=True`, `sse2=True`, `sse4_1=True`, and `avx2=True`.

Command shape:

```powershell
.\.venv\Scripts\python.exe setup.py build_ext --inplace
.\.venv\Scripts\python.exe scripts\bench_dequant.py --qtype <QTYPE> --backends ref,sse2,sse4_1,avx2 --rows <256|1024|4096> --iterations 25 --repetitions 3
```

Total wall-clock time for the full 24-qtype, 3-row-size matrix was `370.0s` (`6.2 min`). Values are mean decoded throughput in GiB/s.

Rows `256`:

```text
qtype     ref   sse2  sse4_1 avx2  best    speedup
Q1_0      0.47  2.32  2.39   2.35  sse4_1  5.09x
Q4_0      3.75 12.95 12.14  13.76  avx2    3.67x
Q4_1      3.57 11.16  9.81  12.48  avx2    3.49x
Q5_0      2.38  3.33  3.42   3.28  sse4_1  1.44x
Q5_1      2.18  3.02  3.23   3.26  avx2    1.49x
Q8_0      5.90 12.79 12.83  13.24  avx2    2.25x
Q2_K      1.77  2.57  2.64   2.52  sse4_1  1.49x
Q3_K      1.59  2.27  2.33   2.35  avx2    1.48x
Q4_K      2.36  2.64  2.68   2.57  sse4_1  1.14x
Q5_K      1.57  2.62  2.56   2.54  sse2    1.66x
Q6_K      1.29  2.45  2.49   2.41  sse4_1  1.92x
IQ2_XXS   0.74  1.12  1.21   1.36  avx2    1.84x
IQ2_XS    0.72  1.11  1.17   1.35  avx2    1.88x
IQ2_S     0.69  1.28  1.23   1.38  avx2    1.99x
IQ3_XXS   0.73  1.12  1.25   1.49  avx2    2.04x
IQ3_S     0.67  1.08  1.12   1.41  avx2    2.09x
IQ1_S     2.34  2.45  2.51   2.57  avx2    1.10x
IQ1_M     1.86  2.04  2.04   2.38  avx2    1.28x
IQ4_NL    3.90  4.90  4.87   4.12  sse2    1.25x
IQ4_XS    1.93  2.14  2.14   2.00  sse4_1  1.11x
TQ1_0     1.67  2.38  2.42   2.52  avx2    1.51x
TQ2_0     1.93  2.66  2.64   2.57  sse2    1.38x
MXFP4     3.47  4.88  4.92   4.27  sse4_1  1.42x
NVFP4     1.36  1.38  1.38   1.26  sse2    1.02x
```

Rows `1024`:

```text
qtype     ref  sse2 sse4_1 avx2 best    speedup
Q1_0      0.48 2.68 2.78   2.70 sse4_1  5.76x
Q4_0      1.87 2.41 2.42   2.31 sse4_1  1.29x
Q4_1      1.81 2.36 2.36   2.28 sse2    1.31x
Q5_0      1.43 1.69 1.72   1.64 sse4_1  1.21x
Q5_1      1.36 1.69 1.68   1.67 sse2    1.24x
Q8_0      2.39 2.58 2.58   2.47 sse2    1.08x
Q2_K      1.82 2.85 2.86   2.79 sse4_1  1.57x
Q3_K      1.70 2.86 2.84   2.76 sse2    1.68x
Q4_K      2.49 2.84 2.86   2.75 sse4_1  1.15x
Q5_K      1.66 2.82 2.82   2.72 sse4_1  1.69x
Q6_K      1.39 2.86 2.86   2.78 sse4_1  2.05x
IQ2_XXS   0.74 1.15 1.32   1.56 avx2    2.10x
IQ2_XS    0.73 1.15 1.23   1.40 avx2    1.91x
IQ2_S     0.71 1.33 1.40   1.43 avx2    2.02x
IQ3_XXS   0.73 1.15 1.31   1.55 avx2    2.13x
IQ3_S     0.68 1.13 1.16   1.45 avx2    2.12x
IQ1_S     2.46 2.65 2.75   2.72 sse4_1  1.12x
IQ1_M     1.97 2.16 2.23   2.64 avx2    1.34x
IQ4_NL    1.79 1.98 1.99   1.86 sse4_1  1.11x
IQ4_XS    2.08 2.31 2.29   1.81 sse2    1.11x
TQ1_0     1.73 2.56 2.63   2.72 avx2    1.57x
TQ2_0     2.03 2.87 3.00   2.94 sse4_1  1.48x
MXFP4     1.81 2.01 2.01   1.91 sse4_1  1.11x
NVFP4     1.05 1.07 1.07   0.97 sse2    1.03x
```

Rows `4096`:

```text
qtype     ref  sse2 sse4_1 avx2 best    speedup
Q1_0      0.49 2.85 2.88   2.81 sse4_1  5.92x
Q4_0      2.06 2.78 2.79   2.64 sse4_1  1.35x
Q4_1      1.99 2.70 2.77   2.70 sse4_1  1.39x
Q5_0      1.56 1.92 1.94   1.86 sse4_1  1.24x
Q5_1      1.47 1.89 1.90   1.85 sse4_1  1.29x
Q8_0      2.58 2.76 2.83   2.74 sse4_1  1.10x
Q2_K      1.78 2.75 2.77   2.70 sse4_1  1.56x
Q3_K      1.64 2.76 2.73   2.64 sse2    1.68x
Q4_K      2.42 2.62 2.66   2.49 sse4_1  1.10x
Q5_K      1.57 2.57 2.55   2.42 sse2    1.63x
Q6_K      1.35 2.58 2.58   2.52 sse4_1  1.91x
IQ2_XXS   0.68 1.01 1.20   1.37 avx2    2.01x
IQ2_XS    0.72 1.12 1.27   1.49 avx2    2.08x
IQ2_S     0.69 1.28 1.22   1.34 avx2    1.93x
IQ3_XXS   0.70 1.18 1.25   1.33 avx2    1.89x
IQ3_S     0.66 1.09 1.11   1.33 avx2    2.00x
IQ1_S     2.39 2.57 2.68   2.63 sse4_1  1.12x
IQ1_M     1.85 2.02 2.07   2.45 avx2    1.32x
IQ4_NL    2.00 2.19 2.17   2.05 sse2    1.10x
IQ4_XS    1.93 2.15 2.14   2.03 sse2    1.11x
TQ1_0     1.67 2.39 2.43   2.52 avx2    1.51x
TQ2_0     1.95 2.68 2.66   2.61 sse2    1.37x
MXFP4     1.93 2.18 2.18   2.06 sse2    1.13x
NVFP4     1.05 1.10 1.10   0.97 sse2    1.04x
```

SIMD was generally faster than `ref`; the notable slower individual backends were `NVFP4/avx2` at all row sizes and `IQ4_XS/avx2` at rows `1024`. The fastest backend is often SSE2 or SSE4.1 for memory-bound/simple kernels, while AVX2 is strongest for the lookup-heavy IQ2/IQ3 family.
