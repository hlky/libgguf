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


BACKENDS = ("ref", "sse2", "sse4_1", "avx2")

# Non-IQ private dequant coverage only. IQ qtypes are intentionally omitted
# from this standardized sweep.
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
    34: 256,
    35: 256,
    39: 32,
    40: 64,
    41: 128,
}


def parse_csv(value: str, choices: tuple[str, ...] | list[str], label: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(choices)
    selected = []
    valid = set(choices)
    for item in value.split(","):
        key = item.strip().lower() if label == "backend" else item.strip().upper()
        if not key:
            continue
        if key not in valid:
            supported = ", ".join(choices)
            raise argparse.ArgumentTypeError(f"unsupported {label} {item!r}; supported: all,{supported}")
        selected.append(key)
    if not selected:
        raise argparse.ArgumentTypeError(f"empty {label} list")
    return selected


def build_input(qtype: int, rows: int, n_per_row: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    source = rng.standard_normal((rows, n_per_row), dtype=np.float32).astype(np.float32)
    if libgguf.quantize_requires_imatrix(qtype):
        imatrix = np.sum(source * source, axis=0, dtype=np.float32)
    else:
        imatrix = None
    return libgguf.quantize_rows(source, qtype, imatrix=imatrix)


def summarize(samples: list[dict[str, float | str]], iterations: int) -> dict[str, float | str]:
    elapsed = [float(sample["elapsed_s"]) for sample in samples]
    throughput = [float(sample["throughput_gib_s"]) for sample in samples]
    return {
        "elapsed_ms_mean": statistics.mean(elapsed) * 1e3,
        "elapsed_ms_median": statistics.median(elapsed) * 1e3,
        "throughput_gib_s_mean": statistics.mean(throughput),
        "throughput_gib_s_median": statistics.median(throughput),
        "repetitions": float(len(samples)),
        "iterations_per_run": float(iterations),
    }


def benchmark_backend(
    qtype_name: str,
    backend: str,
    quantized: np.ndarray,
    rows: int,
    n_per_row: int,
    iterations: int,
    repetitions: int,
) -> dict[str, float | str]:
    qtype = QTYPES[qtype_name]
    samples = []
    checksum = 0.0
    for _ in range(repetitions):
        _libgguf._dequantize_for_backend(qtype, backend, quantized, rows, n_per_row)
        start = time.perf_counter()
        for _ in range(iterations):
            result = _libgguf._dequantize_for_backend(qtype, backend, quantized, rows, n_per_row)
        elapsed = time.perf_counter() - start
        output = np.frombuffer(result, dtype=np.float32)
        checksum = float(output[: min(16, output.size)].sum(dtype=np.float64))
        samples.append(
            {
                "elapsed_s": elapsed / iterations,
                "throughput_gib_s": rows * n_per_row * 4 * iterations / elapsed / (1024**3),
            }
        )
    summary = summarize(samples, iterations)
    summary.update({"qtype": qtype_name, "backend": backend, "method": "private_hook", "checksum": checksum})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark non-IQ libgguf dequant backends.")
    parser.add_argument("--qtypes", default="all", help="Comma-separated qtypes or all. IQ qtypes are not included.")
    parser.add_argument("--qtype", default=None, help="Deprecated alias for --qtypes.")
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--n-per-row", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--backends", type=str, default="all")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    qtype_arg = args.qtype if args.qtype is not None else args.qtypes
    qtypes = parse_csv(qtype_arg, list(QTYPES), "qtype")
    backends = parse_csv(args.backends, list(BACKENDS), "backend")

    benchmarks: list[dict[str, float | str]] = []
    for qtype_name in qtypes:
        block_size = BLOCK_SIZES[QTYPES[qtype_name]]
        n_per_row = args.n_per_row or block_size * 8
        if n_per_row % block_size != 0:
            raise SystemExit(f"--n-per-row must be a multiple of {block_size} for {qtype_name}")
        quantized = build_input(QTYPES[qtype_name], args.rows, n_per_row, args.seed)
        for backend in backends:
            if not _libgguf._dequant_cpu_supports_backend(backend):
                benchmarks.append(
                    {
                        "qtype": qtype_name,
                        "backend": backend,
                        "method": "private_hook",
                        "skipped": "unsupported CPU backend",
                    }
                )
                continue
            benchmarks.append(
                benchmark_backend(
                    qtype_name,
                    backend,
                    quantized,
                    args.rows,
                    n_per_row,
                    args.iterations,
                    args.repetitions,
                )
            )

    print(
        json.dumps(
            {
                "config": {
                    "qtypes": qtypes,
                    "backends": backends,
                    "rows": args.rows,
                    "n_per_row": args.n_per_row,
                    "iterations": args.iterations,
                    "repetitions": args.repetitions,
                    "seed": args.seed,
                    "iq_qtypes": "ignored",
                },
                "benchmarks": benchmarks,
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
