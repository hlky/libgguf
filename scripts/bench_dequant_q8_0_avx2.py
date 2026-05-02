#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import tempfile
import shutil
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_libgguf import build_shared_lib, default_output_path


Q8_0_TYPE = 8


def configure_libgguf_dequant_api(lib: ctypes.CDLL) -> None:
    lib.libgguf_row_size.argtypes = [ctypes.c_int, ctypes.c_longlong]
    lib.libgguf_row_size.restype = ctypes.c_size_t
    lib.libgguf_quantize_chunk.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.libgguf_quantize_chunk.restype = ctypes.c_size_t
    lib.libgguf_dequantize_chunk.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.c_longlong,
    ]
    lib.libgguf_dequantize_chunk.restype = ctypes.c_size_t


def benchmark_child(
    shared_lib: Path,
    quantized: Path,
    rows: int,
    n_per_row: int,
    iterations: int,
) -> dict[str, float | str]:
    lib = ctypes.CDLL(str(shared_lib))
    configure_libgguf_dequant_api(lib)

    quantized_rows = np.load(quantized)
    output = np.empty((rows, n_per_row), dtype=np.float32)

    lib.libgguf_dequantize_chunk(
        Q8_0_TYPE,
        quantized_rows.ctypes.data,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        0,
        rows,
        n_per_row,
    )

    start = time.perf_counter()
    for _ in range(iterations):
        lib.libgguf_dequantize_chunk(
            Q8_0_TYPE,
            quantized_rows.ctypes.data,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            0,
            rows,
            n_per_row,
        )
    elapsed = time.perf_counter() - start

    throughput = (
        (rows * n_per_row * 4 * iterations) / elapsed / (1024**3)
    )
    return {
        "backend": os.environ.get("LIBGGUF_DEQUANT_Q8_0_BACKEND", "auto"),
        "elapsed_s": elapsed / iterations,
        "throughput_gib_s": throughput,
        "bytes_per_iter": float(rows * n_per_row * 4),
        "output_checksum": float(output[0, 0]),
    }


def run_child(argv: argparse.Namespace) -> None:
    result = benchmark_child(
        shared_lib=Path(argv.shared_lib),
        quantized=Path(argv.quantized),
        rows=argv.rows,
        n_per_row=argv.n_per_row,
        iterations=argv.iterations,
    )
    print(json.dumps(result))


def benchmark_backend(
    python_path: Path,
    script_path: Path,
    shared_lib: Path,
    quantized: Path,
    backend: str,
    rows: int,
    n_per_row: int,
    iterations: int,
) -> dict[str, float | str]:
    env = os.environ.copy()
    env["LIBGGUF_DEQUANT_Q8_0_BACKEND"] = backend
    env["LIBGGUF_NUM_THREADS"] = "1"

    cmd = [
        str(python_path),
        str(script_path),
        "--child",
        "--shared-lib",
        str(shared_lib),
        "--quantized",
        str(quantized),
        "--rows",
        str(rows),
        "--n-per-row",
        str(n_per_row),
        "--iterations",
        str(iterations),
    ]
    proc = subprocess.run(
        cmd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout.strip())
    payload["backend"] = backend
    return payload


def build_inputs(lib: ctypes.CDLL, rows: int, n_per_row: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    source = rng.standard_normal((rows, n_per_row), dtype=np.float32).astype(np.float32)
    row_size = lib.libgguf_row_size(Q8_0_TYPE, n_per_row)
    quantized = np.empty(rows * row_size, dtype=np.uint8)

    written = lib.libgguf_quantize_chunk(
        Q8_0_TYPE,
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        quantized.ctypes.data,
        0,
        rows,
        n_per_row,
        None,
    )
    assert written == quantized.nbytes
    return quantized


def summarize(samples: list[dict[str, float | str]], iterations: int) -> dict[str, float | str]:
    elapsed = [sample["elapsed_s"] for sample in samples]
    throughput = [sample["throughput_gib_s"] for sample in samples]
    return {
        "elapsed_ms_mean": statistics.mean(elapsed) * 1e3,
        "elapsed_ms_median": statistics.median(elapsed) * 1e3,
        "throughput_gib_s_mean": statistics.mean(throughput),
        "throughput_gib_s_median": statistics.median(throughput),
        "repetitions": float(len(samples)),
        "iterations_per_run": float(iterations),
        "backend": samples[0]["backend"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Q8_0 dequantization kernels.",
    )
    parser.add_argument("--shared-lib", type=str, default="")
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--n-per-row", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--backends", type=str, default="sse2,sse4_1,avx2")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--quantized", type=str, default="")

    args = parser.parse_args()
    script_path = Path(__file__).resolve()
    if args.child:
        run_child(args)
        return

    tmp_dir = Path(tempfile.mkdtemp(prefix="libgguf-bench-"))
    try:
        shared_lib = Path(args.shared_lib) if args.shared_lib else None
        if shared_lib is None:
            shared_lib = tmp_dir / default_output_path().name

        shared_lib = build_shared_lib(output=shared_lib, build_dir=tmp_dir / "build")
        lib = ctypes.CDLL(str(shared_lib))
        configure_libgguf_dequant_api(lib)

        quantized = build_inputs(lib, args.rows, args.n_per_row, args.seed)
        quantized_path = tmp_dir / "quantized.npy"
        np.save(quantized_path, quantized)

        backends = [backend.strip() for backend in args.backends.split(",") if backend.strip()]
        summaries: dict[str, dict[str, float | str]] = {}

        for backend in backends:
            runs = [
                benchmark_backend(
                    Path(sys.executable),
                    script_path,
                    shared_lib,
                    quantized_path,
                    backend,
                    args.rows,
                    args.n_per_row,
                    args.iterations,
                )
                for _ in range(args.repetitions)
            ]
            summaries[backend] = summarize(runs, args.iterations)

        print(json.dumps({"benchmarks": summaries}, sort_keys=True, indent=2))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
