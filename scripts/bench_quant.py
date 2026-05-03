#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from libgguf import _libgguf


BACKENDS = ("ref", "sse2", "sse4_1", "avx2")

# Non-IQ quantization coverage only. IQ quantizers are intentionally omitted from
# this benchmark matrix because their benchmark coverage is tracked separately.
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
    41: 128,
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


def qtype_name(qtype: int) -> str:
    for name, value in QTYPES.items():
        if value == qtype:
            return name
    raise ValueError(f"unsupported qtype {qtype}")


def direct_backend_support_fn(qtype: int) -> Callable[[str], bool] | None:
    if qtype == QTYPES["Q1_0"]:
        return _libgguf._q1_0_cpu_supports_backend
    if qtype == QTYPES["Q4_0"]:
        return _libgguf._q4_0_cpu_supports_backend
    if qtype == QTYPES["Q4_1"]:
        return _libgguf._q4_1_cpu_supports_backend
    if qtype == QTYPES["Q5_0"]:
        return _libgguf._q5_0_cpu_supports_backend
    if qtype == QTYPES["Q5_1"]:
        return _libgguf._q5_1_cpu_supports_backend
    if qtype == QTYPES["Q8_0"]:
        return _libgguf._q8_0_cpu_supports_backend
    if qtype == QTYPES["Q2_K"]:
        return _libgguf._q2_k_cpu_supports_backend
    if qtype == QTYPES["Q3_K"]:
        return _libgguf._q3_k_cpu_supports_backend
    if qtype == QTYPES["Q4_K"]:
        return _libgguf._q4_k_cpu_supports_backend
    if qtype == QTYPES["Q5_K"]:
        return _libgguf._q5_k_cpu_supports_backend
    if qtype == QTYPES["Q6_K"]:
        return _libgguf._q6_k_cpu_supports_backend
    if qtype == QTYPES["TQ1_0"]:
        return _libgguf._tq1_0_cpu_supports_backend
    if qtype == QTYPES["TQ2_0"]:
        return _libgguf._tq2_0_cpu_supports_backend
    if qtype == QTYPES["MXFP4"]:
        return _libgguf._mxfp4_cpu_supports_backend
    if qtype == QTYPES["NVFP4"]:
        return _libgguf._nvfp4_cpu_supports_backend
    return None


def direct_quantize_fn(qtype: int) -> Callable[[str, np.ndarray, int, int], bytes] | None:
    if qtype == QTYPES["Q1_0"]:
        return _libgguf._quantize_q1_0_for_backend
    if qtype == QTYPES["Q4_0"]:
        return _libgguf._quantize_q4_0_for_backend
    if qtype == QTYPES["Q4_1"]:
        return _libgguf._quantize_q4_1_for_backend
    if qtype == QTYPES["Q5_0"]:
        return _libgguf._quantize_q5_0_for_backend
    if qtype == QTYPES["Q5_1"]:
        return _libgguf._quantize_q5_1_for_backend
    if qtype == QTYPES["Q8_0"]:
        return _libgguf._quantize_q8_0_for_backend
    if qtype == QTYPES["Q2_K"]:
        return _libgguf._quantize_q2_k_for_backend
    if qtype == QTYPES["Q3_K"]:
        return _libgguf._quantize_q3_k_for_backend
    if qtype == QTYPES["Q4_K"]:
        return _libgguf._quantize_q4_k_for_backend
    if qtype == QTYPES["Q5_K"]:
        return _libgguf._quantize_q5_k_for_backend
    if qtype == QTYPES["Q6_K"]:
        return _libgguf._quantize_q6_k_for_backend
    if qtype == QTYPES["TQ1_0"]:
        return _libgguf._quantize_tq1_0_for_backend
    if qtype == QTYPES["TQ2_0"]:
        return _libgguf._quantize_tq2_0_for_backend
    if qtype == QTYPES["MXFP4"]:
        return _libgguf._quantize_mxfp4_for_backend
    if qtype == QTYPES["NVFP4"]:
        return _libgguf._quantize_nvfp4_for_backend
    return None


def supports_backend(qtype_name_: str, backend: str) -> bool:
    qtype = QTYPES[qtype_name_]
    direct_supports = direct_backend_support_fn(qtype)
    assert direct_supports is not None
    return bool(direct_supports(backend))


def build_input(rows: int, n_per_row: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, n_per_row), dtype=np.float32).astype(np.float32)


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


def benchmark_direct_backend(
    qtype_name_: str,
    backend: str,
    source: np.ndarray,
    rows: int,
    n_per_row: int,
    iterations: int,
    repetitions: int,
) -> dict[str, float | str]:
    qtype = QTYPES[qtype_name_]
    quantize = direct_quantize_fn(qtype)
    assert quantize is not None
    samples = []
    checksum = 0
    for _ in range(repetitions):
        quantize(backend, source, rows, n_per_row)
        start = time.perf_counter()
        for _ in range(iterations):
            result = quantize(backend, source, rows, n_per_row)
        elapsed = time.perf_counter() - start
        output = np.frombuffer(result, dtype=np.uint8)
        checksum = int(output[: min(4096, output.size)].sum(dtype=np.uint64))
        samples.append(
            {
                "elapsed_s": elapsed / iterations,
                "throughput_gib_s": rows * n_per_row * 4 * iterations / elapsed / (1024**3),
            }
        )
    summary = summarize(samples, iterations)
    summary.update({"qtype": qtype_name_, "backend": backend, "method": "private_hook", "checksum": checksum})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark non-IQ libgguf quantize backends.")
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
    backends = [backend.lower() for backend in parse_csv(args.backends, list(BACKENDS), "backend")]

    benchmarks: list[dict[str, float | str]] = []
    for qtype_name_ in qtypes:
        block_size = BLOCK_SIZES[QTYPES[qtype_name_]]
        n_per_row = args.n_per_row or block_size * 8
        if n_per_row % block_size != 0:
            raise SystemExit(f"--n-per-row must be a multiple of {block_size} for {qtype_name_}")
        source = build_input(args.rows, n_per_row, args.seed)
        for backend in backends:
            if not supports_backend(qtype_name_, backend):
                benchmarks.append(
                    {
                        "qtype": qtype_name_,
                        "backend": backend,
                        "method": "private_hook",
                        "skipped": "unsupported CPU backend",
                    }
                )
                continue
            benchmarks.append(
                benchmark_direct_backend(
                    qtype_name_,
                    backend,
                    source,
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
