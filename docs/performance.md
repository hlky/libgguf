# Performance Notes

This document records local benchmark findings for native quantization backends. Results are machine-dependent and should be treated as regression guidance rather than portable guarantees.

## Q4_0 SSE2 and AVX2

Q4_0 quantization now has scalar, SSE2, and AVX2 row kernels behind runtime dispatch. AVX2 is compiled only for `quant_q4_0_avx2.cpp`; it is not enabled as a global build flag.

The first SIMD implementation only vectorized nibble quantization. The dominant cost remained a scalar, branchy signed-absmax scan over each 32-float block:

```cpp
float amax = 0.0f;
float max = 0.0f;
for (int j = 0; j < QK4_0; ++j) {
  const float v = xb[j];
  if (amax < fabsf(v)) {
    amax = fabsf(v);
    max = v;
  }
}
```

After review, both SIMD paths were reshaped around the block-level bottleneck:

- Vectorized signed max-by-absolute-value selection.
- Preserved reference tie behavior by tracking the source index for each SIMD lane, so equal absolute values still pick the earliest element.
- Changed SSE2 from four output bytes per loop to eight output bytes per loop.
- Changed AVX2 to combine low and high nibbles while still in 32-bit lanes, then narrow and emit one 8-byte store without stack spill/manual copy.

## Current Local Results

Benchmark input:

- qtype: `Q4_0`
- shape: `4096 x 4096`
- input size: `64.0 MiB` float32
- data: deterministic standard-normal random values, seed `12345`
- build: Windows x64, MSVC, extension built with `python setup.py build_ext --inplace`
- timing: median and best after warmup, measured separately for preallocated and allocating APIs

Results:

```text
requested_backend,selected_backend,mode,threads,median_ms,best_ms,input_gib_s
ref,ref,preallocated,1,47.296,46.801,1.321
ref,ref,preallocated,max,5.465,5.064,11.437
ref,ref,allocating,1,50.643,50.429,1.234
ref,ref,allocating,max,6.991,6.362,8.939
sse2,sse2,preallocated,1,35.078,34.963,1.782
sse2,sse2,preallocated,max,4.748,4.351,13.163
sse2,sse2,allocating,1,38.453,38.125,1.625
sse2,sse2,allocating,max,5.938,5.408,10.525
avx2,avx2,preallocated,1,28.096,27.926,2.225
avx2,avx2,preallocated,max,4.771,4.576,13.101
avx2,avx2,allocating,1,31.373,31.142,1.992
avx2,avx2,allocating,max,5.602,5.523,11.156
```

Single-thread preallocated speedups versus scalar:

```text
SSE2: 1.35x
AVX2: 1.68x
```

The threaded preallocated path is close between SSE2 and AVX2 on this machine (`4.748 ms` versus `4.771 ms`). The allocating threaded path favors AVX2 (`5.602 ms` versus `5.938 ms`). Based on these results, the default runtime preference is AVX2, then SSE2, then scalar when the CPU supports those backends.

Use `LIBGGUF_Q4_0_BACKEND=ref|sse2|avx2` to force a backend for local comparisons.

## Dequantization

The dequantization benchmark times the public Python APIs after one untimed quantization setup pass. It measures:

- `preallocated`: `libgguf.dequantize_rows_into_raw(...)`
- `allocating`: `libgguf.dequantize_rows(...)`

Current local command:

```powershell
$env:PYTHONPATH='src'
$py = if (Test-Path .venv\Scripts\python.exe) { (Resolve-Path .venv\Scripts\python.exe).Path } else { 'python' }
& $py tools\bench_dequant.py --rows 4096 --cols 4096 --repeats 11 --warmup 3 --qtypes all
```

Benchmark input:

- shape: `4096 x 4096`
- decoded size: `64.0 MiB` float32
- data: deterministic standard-normal random values, seed `12345`
- build: Windows x64, MSVC, extension built with `python setup.py build_ext --inplace`
- backend: default threaded dequantization through `libgguf_dequantize_chunk`

Q4_0 and Q8_0 dequantization have scalar, SSE2, SSE4.1, and AVX2 row kernels behind runtime dispatch. On this machine, AVX2 was fastest for Q4_0 and SSE4.1 was slightly ahead of AVX2 for Q8_0, so the default runtime preference is AVX2, then SSE4.1, then SSE2, then scalar when the CPU supports those backends. Use `LIBGGUF_DEQUANT_Q4_0_BACKEND=ref|sse2|sse4_1|avx2` or `LIBGGUF_DEQUANT_Q8_0_BACKEND=ref|sse2|sse4_1|avx2` to force a backend for local comparisons.

Focused dequant kernel timing on this same machine used:

- `python scripts/bench_dequant_q4_0_avx2.py --rows 2048 --n-per-row 2048 --iterations 200 --repetitions 5 --backends sse2,sse4_1,avx2`
- `python scripts/bench_dequant_q8_0_avx2.py --rows 2048 --n-per-row 2048 --iterations 200 --repetitions 5 --backends sse2,sse4_1,avx2`

Observed means:

```text
Q4_0: avx2 1.1039 ms, sse4_1 1.2081 ms, sse2 1.1803 ms
Q8_0: avx2 1.1720 ms, sse4_1 1.1601 ms, sse2 1.2165 ms
```

Results:

```text
qtype,mode,median_ms,best_ms,encoded_gib_s,decoded_gib_s
Q1_0,preallocated,8.759,8.486,0.251,7.135
Q1_0,allocating,16.983,16.439,0.129,3.680
Q4_0,preallocated,4.393,3.915,2.000,14.226
Q4_0,allocating,13.856,13.663,0.634,4.511
Q4_1,preallocated,4.398,4.142,2.220,14.211
Q4_1,allocating,14.279,13.823,0.684,4.377
Q5_0,preallocated,4.457,4.326,2.410,14.021
Q5_0,allocating,14.167,13.761,0.758,4.412
Q5_1,preallocated,4.920,4.609,2.382,12.704
Q5_1,allocating,14.243,14.005,0.823,4.388
Q8_0,preallocated,4.336,4.098,3.829,14.415
Q8_0,allocating,14.071,13.705,1.180,4.442
Q2_K,preallocated,4.322,4.097,1.186,14.461
Q2_K,allocating,13.855,13.521,0.370,4.511
Q3_K,preallocated,4.425,4.134,1.517,14.124
Q3_K,allocating,13.724,13.476,0.489,4.554
Q4_K,preallocated,4.194,4.123,2.096,14.903
Q4_K,allocating,13.730,13.602,0.640,4.552
Q5_K,preallocated,4.443,4.125,2.418,14.068
Q5_K,allocating,15.918,13.684,0.675,3.926
Q6_K,preallocated,4.560,4.212,2.811,13.707
Q6_K,allocating,14.044,13.677,0.913,4.450
IQ2_XXS,preallocated,6.356,5.960,0.634,9.833
IQ2_XXS,allocating,14.675,14.373,0.274,4.259
IQ2_XS,preallocated,6.674,5.894,0.677,9.364
IQ2_XS,allocating,14.751,14.507,0.306,4.237
IQ2_S,preallocated,6.524,6.138,0.767,9.580
IQ2_S,allocating,15.218,14.599,0.329,4.107
IQ3_XXS,preallocated,6.363,6.079,0.940,9.822
IQ3_XXS,allocating,14.820,14.316,0.404,4.217
IQ3_S,preallocated,7.073,6.405,0.949,8.836
IQ3_S,allocating,15.008,14.447,0.447,4.165
IQ1_S,preallocated,4.154,3.923,0.735,15.045
IQ1_S,allocating,13.800,13.360,0.221,4.529
IQ1_M,preallocated,4.272,3.961,0.800,14.631
IQ1_M,allocating,13.852,13.484,0.247,4.512
IQ4_NL,preallocated,4.242,3.954,2.072,14.735
IQ4_NL,allocating,15.629,13.520,0.562,3.999
IQ4_XS,preallocated,4.324,4.119,1.920,14.454
IQ4_XS,allocating,13.761,13.412,0.603,4.542
TQ1_0,preallocated,4.296,4.201,0.767,14.549
TQ1_0,allocating,13.966,13.528,0.236,4.475
TQ2_0,preallocated,4.339,3.942,0.929,14.406
TQ2_0,allocating,13.642,13.218,0.295,4.582
MXFP4,preallocated,4.100,3.853,2.025,15.245
MXFP4,allocating,13.522,13.280,0.614,4.622
NVFP4,preallocated,5.123,4.785,1.715,12.199
NVFP4,allocating,14.259,13.697,0.616,4.383
```

The allocating API is dominated by output allocation and initialization at this size. Use the preallocated API for throughput-sensitive paths.

## Benchmark Method

The benchmark should measure preallocated and allocating APIs separately, and it should separate one-thread from default threaded behavior. The private backend test hooks are useful for direct backend parity checks, but timing the public raw APIs with `LIBGGUF_Q4_0_BACKEND` exercises the normal dispatch path.

Minimal PowerShell benchmark:

```powershell
$env:PYTHONPATH='src'
$py = if (Test-Path .venv\Scripts\python.exe) { (Resolve-Path .venv\Scripts\python.exe).Path } else { 'python' }
@'
from __future__ import annotations

import os
import subprocess
import sys

backends = ["ref", "sse2", "sse4_1", "avx2"]
rows = int(os.environ.get("BENCH_Q4_ROWS", "4096"))
cols = int(os.environ.get("BENCH_Q4_COLS", "4096"))

child = r'''
import gc
import os
import statistics
import time
import numpy as np
import libgguf
from libgguf import _libgguf

Q4_0 = 2
rows = int(os.environ["BENCH_Q4_ROWS"])
cols = int(os.environ["BENCH_Q4_COLS"])
repeats = int(os.environ.get("BENCH_Q4_REPEATS", "11"))
src = np.random.default_rng(12345).standard_normal((rows, cols), dtype=np.float32)
backend = _libgguf._q4_0_backend()
out_size = rows * libgguf.row_size(Q4_0, cols)
dst = bytearray(out_size)

def time_call(fn):
    for _ in range(3):
        fn()
    samples = []
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - start)
    finally:
        gc.enable()
    return statistics.median(samples), min(samples)

for mode, fn in [
    ("preallocated", lambda: libgguf.quantize_rows_into_raw(Q4_0, src, dst, rows, cols)),
    ("allocating", lambda: libgguf.quantize_rows_raw(Q4_0, src, rows, cols)),
]:
    for threads in ["1", "max"]:
        if threads == "max":
            os.environ.pop("LIBGGUF_NUM_THREADS", None)
        else:
            os.environ["LIBGGUF_NUM_THREADS"] = threads
        median, best = time_call(fn)
        gib_s = src.nbytes / (1024 ** 3) / median
        print(f"{os.environ['LIBGGUF_Q4_0_BACKEND']},{backend},{mode},{threads},{median * 1000:.3f},{best * 1000:.3f},{gib_s:.3f}")
'''

print(f"Q4_0 benchmark: rows={rows} cols={cols} input={rows * cols * 4 / 1024 / 1024:.1f} MiB", flush=True)
print("requested_backend,selected_backend,mode,threads,median_ms,best_ms,input_gib_s", flush=True)
for backend in backends:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")
    env["LIBGGUF_Q4_0_BACKEND"] = backend
    env["BENCH_Q4_ROWS"] = str(rows)
    env["BENCH_Q4_COLS"] = str(cols)
    env["BENCH_Q4_REPEATS"] = "11"
    subprocess.run([sys.executable, "-c", child], cwd=os.getcwd(), env=env, check=True)
'@ | & $py -
```

## Review Checklist

When adding or changing SIMD quantizers:

- Keep AVX2 and SSE4.1 code in isolated translation units with per-source compiler flags.
- Keep runtime feature detection in `csrc/common/libgguf_cpu.*`.
- Preserve byte-for-byte parity with the scalar reference, including tie behavior.
- Benchmark preallocated and allocating APIs separately.
- Benchmark `LIBGGUF_NUM_THREADS=1` and default threaded behavior separately.
- Prefer measured backend defaults over CPU-feature-only assumptions.
