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

BACKENDS = ("ref", "sse2", "sse4_1", "avx2")
MODES = ("unweighted", "weighted")

# Common-helper benchmark qtypes. IQ qtypes are intentionally omitted.
QTYPES: dict[str, tuple[int, int]] = {
    "Q4_0": (2, 32),
    "Q4_1": (3, 32),
    "Q5_0": (6, 32),
    "Q5_1": (7, 32),
    "Q2_K": (10, 256),
    "Q3_K": (11, 256),
    "Q4_K": (12, 256),
    "Q5_K": (13, 256),
    "Q6_K": (14, 256),
}


def parse_csv(value: str, choices: tuple[str, ...] | list[str], label: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(choices)
    selected = []
    valid = set(choices)
    for item in value.split(","):
        key = item.strip().lower() if label in {"backend", "mode"} else item.strip().upper()
        if not key:
            continue
        if key not in valid:
            supported = ", ".join(choices)
            raise argparse.ArgumentTypeError(f"unsupported {label} {item!r}; supported: all,{supported}")
        selected.append(key)
    if not selected:
        raise argparse.ArgumentTypeError(f"empty {label} list")
    return selected


def benchmark_one(
    qtype_name: str,
    backend: str,
    mode: str,
    rows: int,
    n_per_row_arg: int,
    iterations: int,
    repetitions: int,
    seed: int,
) -> dict[str, float | str | bool]:
    import libgguf
    from libgguf import _libgguf

    if not _libgguf._common_quant_cpu_supports_backend(backend):
        return {
            "qtype": qtype_name,
            "backend": backend,
            "weighted": mode == "weighted",
            "skipped": "unsupported CPU backend",
        }
    _libgguf._common_quant_set_backend(backend)
    if _libgguf._common_quant_backend() != backend:
        raise SystemExit(f"selected backend is {_libgguf._common_quant_backend()}, expected {backend}")

    qtype, block_size = QTYPES[qtype_name]
    n_per_row = n_per_row_arg or block_size
    if n_per_row % block_size != 0:
        raise SystemExit(f"--n-per-row must be a multiple of {block_size} for {qtype_name}")

    rng = np.random.default_rng(seed)
    source = rng.standard_normal((rows, n_per_row), dtype=np.float32).astype(np.float32)
    imatrix = None
    if mode == "weighted":
        imatrix = np.linspace(0.25, 1.75, n_per_row, dtype=np.float32)

    elapsed = []
    throughput = []
    checksum = 0
    for _ in range(repetitions):
        libgguf.quantize_rows_raw(qtype, source, rows, n_per_row, imatrix)
        start = time.perf_counter()
        for _ in range(iterations):
            result = libgguf.quantize_rows_raw(qtype, source, rows, n_per_row, imatrix)
        duration = time.perf_counter() - start
        elapsed.append(duration / iterations)
        throughput.append(rows * n_per_row * 4 * iterations / duration / (1024**3))
        checksum = int(np.frombuffer(result, dtype=np.uint8)[:4096].sum(dtype=np.uint64))

    return {
        "qtype": qtype_name,
        "backend": backend,
        "weighted": mode == "weighted",
        "elapsed_ms_mean": statistics.mean(elapsed) * 1e3,
        "elapsed_ms_median": statistics.median(elapsed) * 1e3,
        "throughput_gib_s_mean": statistics.mean(throughput),
        "throughput_gib_s_median": statistics.median(throughput),
        "repetitions": float(repetitions),
        "iterations_per_run": float(iterations),
        "checksum": checksum,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark non-IQ common quant helper backend dispatch.")
    parser.add_argument("--qtypes", default="all", help="Comma-separated qtypes or all. IQ qtypes are not included.")
    parser.add_argument("--qtype", default=None, help="Deprecated alias for --qtypes.")
    parser.add_argument("--backends", default="all", help="Comma-separated backends or all.")
    parser.add_argument("--backend", default=None, help="Deprecated alias for --backends.")
    parser.add_argument("--modes", default="all", help="Comma-separated modes: unweighted,weighted,all.")
    parser.add_argument("--weighted", action="store_true", help="Deprecated alias for --modes weighted.")
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--n-per-row", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    qtype_arg = args.qtype if args.qtype is not None else args.qtypes
    backend_arg = args.backend if args.backend is not None else args.backends
    mode_arg = "weighted" if args.weighted else args.modes

    qtypes = parse_csv(qtype_arg, list(QTYPES), "qtype")
    backends = parse_csv(backend_arg, list(BACKENDS), "backend")
    modes = parse_csv(mode_arg, list(MODES), "mode")

    benchmarks: list[dict[str, float | str | bool]] = []
    for qtype_name in qtypes:
        for backend in backends:
            for mode in modes:
                benchmarks.append(
                    benchmark_one(
                        qtype_name,
                        backend,
                        mode,
                        args.rows,
                        args.n_per_row,
                        args.iterations,
                        args.repetitions,
                        args.seed,
                    )
                )

    print(
        json.dumps(
            {
                "config": {
                    "qtypes": qtypes,
                    "backends": backends,
                    "modes": modes,
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
