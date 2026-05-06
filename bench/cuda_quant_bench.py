from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

import numpy as np
import torch

import libgguf
from libgguf import GGMLQuantizationType, GGML_QUANT_SIZES
import libgguf.libgguf_cuda as libgguf_cuda


DEFAULT_BASELINE = ROOT / "bench/results/cuda_cpu_quant_dequant_20260506T160633Z/results.csv"
ROOFLINE_GB_S = 936.2


def parse_qtypes(value: str) -> list[GGMLQuantizationType]:
    if value.lower() == "all":
        names = [qtype.name for qtype in GGMLQuantizationType]
    else:
        names = [part.strip().upper() for part in value.split(",") if part.strip()]

    qtypes: list[GGMLQuantizationType] = []
    for name in names:
        try:
            qtypes.append(GGMLQuantizationType[name])
        except KeyError as exc:
            valid = ", ".join(qtype.name for qtype in GGMLQuantizationType)
            raise argparse.ArgumentTypeError(f"unknown qtype {name!r}; valid: {valid}") from exc
    return qtypes


def parse_shapes(value: str) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for part in value.split(","):
        item = part.strip().lower()
        if not item:
            continue
        pieces = item.split("x")
        if len(pieces) != 2:
            raise argparse.ArgumentTypeError(f"shape {part!r} must look like ROWSxWIDTH")
        rows, width = int(pieces[0]), int(pieces[1])
        if rows <= 0 or width <= 0:
            raise argparse.ArgumentTypeError(f"shape {part!r} must have positive dimensions")
        shapes.append((rows, width))
    if not shapes:
        raise argparse.ArgumentTypeError("at least one shape is required")
    return shapes


def parse_shape_iterations(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    if not value:
        return result
    for part in value.split(","):
        item = part.strip().lower()
        if not item:
            continue
        shape, iterations = item.split(":", 1)
        result[shape] = int(iterations)
    return result


def load_baselines(path: Path | None) -> dict[tuple[str, str], float]:
    if path is None or not path.exists():
        return {}

    baselines: dict[tuple[str, str], float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("backend") == "cuda" and row.get("operation") == "quantize":
                baselines[(row["qtype"], row["shape"])] = float(row["ms"])
    return baselines


def make_rows(qtype: GGMLQuantizationType, rows: int, width: int) -> torch.Tensor:
    index = torch.arange(rows * width, device="cuda", dtype=torch.float32).reshape(rows, width)
    wave = torch.sin(index * 0.013 + float(qtype.value)) + torch.cos(index * 0.007)
    ramp = ((index.remainder(17.0)) - 8.0) / 6.0
    return (wave * 0.75 + ramp).contiguous()


def make_imatrix(data: torch.Tensor, qtype: GGMLQuantizationType) -> tuple[torch.Tensor | None, np.ndarray | None]:
    if not libgguf.quantize_requires_imatrix(qtype):
        return None, None
    imatrix = torch.sum(data * data, dim=0).contiguous()
    return imatrix, imatrix.cpu().numpy()


def time_cuda(fn: Any, iterations: int, warmup: int) -> tuple[float, torch.Tensor]:
    result = None
    for _ in range(warmup):
        result = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        result = fn()
    end.record()
    torch.cuda.synchronize()
    assert result is not None
    return start.elapsed_time(end) / iterations, result


def benchmark_qtype_shape(
    qtype: GGMLQuantizationType,
    rows: int,
    width: int,
    *,
    iterations: int,
    warmup: int,
    kernel_variant: str,
    baseline_ms: float,
    exact_rows: int,
) -> dict[str, Any]:
    block_size, _ = GGML_QUANT_SIZES[qtype]
    if width % block_size != 0:
        raise ValueError(f"{qtype.name} requires width divisible by {block_size}, got {width}")

    data = make_rows(qtype, rows, width)
    imatrix, imatrix_np = make_imatrix(data, qtype)
    ms, actual = time_cuda(lambda: libgguf_cuda.quantize(data, int(qtype), imatrix), iterations, warmup)

    exact = "skip"
    if rows <= exact_rows:
        expected = libgguf.quantize_rows(data.cpu().numpy(), qtype, imatrix=imatrix_np)
        exact = str(np.array_equal(actual.cpu().numpy(), expected))

    decoded_bytes = data.numel() * data.element_size()
    encoded_bytes = actual.numel() * actual.element_size()
    traffic_gb_s = (decoded_bytes + encoded_bytes) / (ms / 1000.0) / 1e9
    n_blocks = data.numel() // block_size
    shape = f"{rows}x{width}"
    return {
        "qtype": qtype.name,
        "shape": shape,
        "rows": rows,
        "width": width,
        "kernel_variant": kernel_variant,
        "iterations": iterations,
        "old_baseline_ms": baseline_ms,
        "ms": ms,
        "delta_pct": ((ms - baseline_ms) / baseline_ms * 100.0) if baseline_ms == baseline_ms else math.nan,
        "decoded_bytes": decoded_bytes,
        "encoded_bytes": encoded_bytes,
        "traffic_gb_s": traffic_gb_s,
        "roofline_pct": traffic_gb_s / ROOFLINE_GB_S * 100.0,
        "blocks_per_s": n_blocks / (ms / 1000.0),
        "exact": exact,
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    print(f"{'qtype':<10} {'shape':<12} {'variant':<28} {'ms':>10} {'delta %':>10} {'exact':>7}")
    print("-" * 84)
    for row in rows:
        delta = row["delta_pct"]
        delta_text = "" if delta != delta else f"{delta:9.2f}"
        print(
            f"{row['qtype']:<10} {row['shape']:<12} {row['kernel_variant']:<28} "
            f"{row['ms']:10.3f} {delta_text:>10} {row['exact']:>7}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark direct libgguf CUDA quantize kernels.")
    parser.add_argument("--qtypes", type=parse_qtypes, required=True, help="Comma-separated qtype names, or all.")
    parser.add_argument(
        "--shapes",
        type=parse_shapes,
        default=parse_shapes("64x4096,4096x4096,11008x4096"),
        help="Comma-separated ROWSxWIDTH shapes.",
    )
    parser.add_argument("--kernel-variant", default="cuda_quantize", help="Label to write into the CSV.")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument(
        "--shape-iterations",
        type=parse_shape_iterations,
        default=parse_shape_iterations("64x4096:3,4096x4096:1,11008x4096:1"),
        help="Optional comma-separated ROWSxWIDTH:ITERATIONS overrides.",
    )
    parser.add_argument("--exact-rows", type=int, default=64, help="Run CPU byte-exact check for shapes up to this row count.")
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--csv", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--json", type=Path, default=None, help="Optional output JSON path.")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        parser.error("CUDA is not available")

    baselines = load_baselines(args.baseline_csv)
    rows_out: list[dict[str, Any]] = []
    for qtype in args.qtypes:
        for rows, width in args.shapes:
            shape = f"{rows}x{width}"
            iterations = args.shape_iterations.get(shape.lower(), args.iterations)
            row = benchmark_qtype_shape(
                qtype,
                rows,
                width,
                iterations=iterations,
                warmup=args.warmup,
                kernel_variant=args.kernel_variant,
                baseline_ms=baselines.get((qtype.name, shape), math.nan),
                exact_rows=args.exact_rows,
            )
            rows_out.append(row)
            print(f"{qtype.name} {shape} {row['ms']:.3f} ms {row['exact']}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "qtypes": [qtype.name for qtype in args.qtypes],
                "shapes": [f"{rows}x{width}" for rows, width in args.shapes],
                "kernel_variant": args.kernel_variant,
                "iterations": args.iterations,
                "warmup": args.warmup,
                "shape_iterations": args.shape_iterations,
                "exact_rows": args.exact_rows,
                "baseline_csv": str(args.baseline_csv) if args.baseline_csv is not None else None,
            },
            "results": rows_out,
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print_table(rows_out)
    print(args.csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
