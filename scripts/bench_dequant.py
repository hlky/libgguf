#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import libgguf
from libgguf import _libgguf


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
    "IQ3_XXS": 18,
    "IQ1_S": 19,
    "IQ4_NL": 20,
    "IQ3_S": 21,
    "IQ2_S": 22,
    "IQ4_XS": 23,
    "IQ1_M": 29,
    "TQ1_0": 34,
    "TQ2_0": 35,
    "MXFP4": 39,
    "NVFP4": 40,
}

BLOCK_SIZES: dict[int, int] = {
    2: 32,
    3: 32,
    6: 32,
    7: 32,
    8: 32,
    10: 256,
    11: 256,
    12: 256,
    13: 256,
    14: 256,
    16: 256,
    17: 256,
    18: 256,
    19: 256,
    20: 32,
    21: 256,
    22: 256,
    23: 256,
    29: 256,
    34: 256,
    35: 256,
    39: 32,
    40: 64,
    41: 128,
}


def parse_qtype(value: str) -> int:
    key = value.upper()
    if key not in QTYPES:
        supported = ", ".join(sorted(QTYPES))
        raise argparse.ArgumentTypeError(f"unsupported qtype {value!r}; supported: {supported}")
    return QTYPES[key]


def build_input(qtype: int, rows: int, n_per_row: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    source = rng.standard_normal((rows, n_per_row), dtype=np.float32).astype(np.float32)
    if libgguf.quantize_requires_imatrix(qtype):
        imatrix = np.sum(source * source, axis=0, dtype=np.float32)
    else:
        imatrix = None
    return libgguf.quantize_rows(source, qtype, imatrix=imatrix)


def benchmark_backend(
    qtype: int,
    backend: str,
    quantized: np.ndarray,
    rows: int,
    n_per_row: int,
    iterations: int,
) -> dict[str, float | str]:
    _libgguf._dequantize_for_backend(qtype, backend, quantized, rows, n_per_row)
    start = time.perf_counter()
    for _ in range(iterations):
        result = _libgguf._dequantize_for_backend(qtype, backend, quantized, rows, n_per_row)
    elapsed = time.perf_counter() - start
    bytes_per_iter = rows * n_per_row * 4
    return {
        "backend": backend,
        "elapsed_s": elapsed / iterations,
        "throughput_gib_s": bytes_per_iter * iterations / elapsed / (1024**3),
        "bytes_per_iter": float(bytes_per_iter),
        "checksum": float(np.frombuffer(result, dtype=np.float32, count=1)[0]),
    }


def summarize(samples: list[dict[str, float | str]], iterations: int) -> dict[str, float | str]:
    elapsed = [float(sample["elapsed_s"]) for sample in samples]
    throughput = [float(sample["throughput_gib_s"]) for sample in samples]
    return {
        "backend": str(samples[0]["backend"]),
        "elapsed_ms_mean": statistics.mean(elapsed) * 1e3,
        "elapsed_ms_median": statistics.median(elapsed) * 1e3,
        "throughput_gib_s_mean": statistics.mean(throughput),
        "throughput_gib_s_median": statistics.median(throughput),
        "repetitions": float(len(samples)),
        "iterations_per_run": float(iterations),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark libgguf dequant backends.")
    parser.add_argument("--qtype", type=parse_qtype, required=True)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--n-per-row", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--backends", type=str, default="ref,sse2,sse4_1,avx2")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    block_size = BLOCK_SIZES[args.qtype]
    n_per_row = args.n_per_row or block_size * 8
    if n_per_row % block_size != 0:
        raise SystemExit(f"--n-per-row must be a multiple of {block_size} for this qtype")

    quantized = build_input(args.qtype, args.rows, n_per_row, args.seed)
    backends = [backend.strip() for backend in args.backends.split(",") if backend.strip()]

    summaries = {}
    for backend in backends:
        if not _libgguf._dequant_cpu_supports_backend(backend):
            summaries[backend] = {"backend": backend, "skipped": "unsupported CPU backend"}
            continue
        runs = [
            benchmark_backend(args.qtype, backend, quantized, args.rows, n_per_row, args.iterations)
            for _ in range(args.repetitions)
        ]
        summaries[backend] = summarize(runs, args.iterations)

    print(json.dumps({"benchmarks": summaries}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

