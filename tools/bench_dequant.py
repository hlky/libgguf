from __future__ import annotations

import argparse
import gc
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

import libgguf


QTYPES: dict[str, int] = {
    "Q1_0": 41,
    "Q4_0": 2,
    "Q4_1": 3,
    "Q5_0": 6,
    "Q5_1": 7,
    "Q8_0": 8,
    "Q2_K": 10,
    "Q3_K": 11,
    "Q4_K": 12,
    "Q5_K": 13,
    "Q6_K": 14,
    "IQ2_XXS": 16,
    "IQ2_XS": 17,
    "IQ2_S": 22,
    "IQ3_XXS": 18,
    "IQ3_S": 21,
    "IQ1_S": 19,
    "IQ1_M": 29,
    "IQ4_NL": 20,
    "IQ4_XS": 23,
    "TQ1_0": 34,
    "TQ2_0": 35,
    "MXFP4": 39,
    "NVFP4": 40,
}


@dataclass(frozen=True)
class Result:
    qtype: str
    mode: str
    median_s: float
    best_s: float
    encoded_bytes: int
    decoded_bytes: int


def time_call(fn: Callable[[], object], repeats: int, warmup: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()

    samples: list[float] = []
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - start)
    finally:
        gc.enable()

    return statistics.median(samples), min(samples)


def benchmark_qtype(name: str, qtype: int, rows: int, cols: int, repeats: int, warmup: int) -> list[Result]:
    src = np.random.default_rng(12345).standard_normal((rows, cols), dtype=np.float32)

    if libgguf.quantize_requires_imatrix(qtype):
        imatrix = np.sum(src * src, axis=0, dtype=np.float32)
    else:
        imatrix = None

    quantized = libgguf.quantize_rows(src, qtype, imatrix=imatrix)
    dst = np.empty_like(src)

    results: list[Result] = []
    encoded_bytes = int(quantized.nbytes)
    decoded_bytes = int(dst.nbytes)

    preallocated = lambda: libgguf.dequantize_rows_into_raw(qtype, quantized, dst, rows, cols)
    median, best = time_call(preallocated, repeats=repeats, warmup=warmup)
    results.append(Result(name, "preallocated", median, best, encoded_bytes, decoded_bytes))

    allocating = lambda: libgguf.dequantize_rows(quantized, qtype, n_per_row=cols)
    median, best = time_call(allocating, repeats=repeats, warmup=warmup)
    results.append(Result(name, "allocating", median, best, encoded_bytes, decoded_bytes))

    return results


def parse_qtypes(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(QTYPES)
    names = [part.strip().upper() for part in value.split(",") if part.strip()]
    unknown = [name for name in names if name not in QTYPES]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown qtype(s): {', '.join(unknown)}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark libgguf dequantization APIs.")
    parser.add_argument("--qtypes", type=parse_qtypes, default=parse_qtypes("Q4_0,Q8_0,Q4_K"), help="Comma-separated GGML qtypes.")
    parser.add_argument("--rows", type=int, default=4096, help="Number of rows to benchmark.")
    parser.add_argument("--cols", type=int, default=4096, help="Float values per row.")
    parser.add_argument("--repeats", type=int, default=11, help="Timed repetitions.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup repetitions before timing.")
    args = parser.parse_args()

    print(f"dequant benchmark: rows={args.rows} cols={args.cols} decoded={args.rows * args.cols * 4 / 1024 / 1024:.1f} MiB")
    print("qtype,mode,median_ms,best_ms,encoded_gib_s,decoded_gib_s")

    for name in args.qtypes:
        for result in benchmark_qtype(name, QTYPES[name], args.rows, args.cols, args.repeats, args.warmup):
            encoded_gib_s = result.encoded_bytes / (1024**3) / result.median_s
            decoded_gib_s = result.decoded_bytes / (1024**3) / result.median_s
            print(
                f"{result.qtype},{result.mode},{result.median_s * 1000:.3f},{result.best_s * 1000:.3f},"
                f"{encoded_gib_s:.3f},{decoded_gib_s:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
