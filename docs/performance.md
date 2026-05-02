# Performance Notes

This document records local benchmark findings for native quantization backends. Results are machine-dependent and should be treated as regression guidance rather than portable guarantees.

## Q4_0 SSE2 and AVX2

Q4_0 now has scalar, SSE2, and AVX2 row kernels behind runtime dispatch. AVX2 is compiled only for `quant_q4_0_avx2.cpp`; it is not enabled as a global build flag.

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

backends = ["ref", "sse2", "avx2"]
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

- Keep AVX2 code in isolated translation units with per-source compiler flags.
- Keep runtime feature detection in `csrc/common/libgguf_cpu.*`.
- Preserve byte-for-byte parity with the scalar reference, including tie behavior.
- Benchmark preallocated and allocating APIs separately.
- Benchmark `LIBGGUF_NUM_THREADS=1` and default threaded behavior separately.
- Prefer measured backend defaults over CPU-feature-only assumptions.
