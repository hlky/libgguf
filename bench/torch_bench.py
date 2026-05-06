from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

import torch

from libgguf import GGMLQuantizationType, GGML_QUANT_SIZES
import libgguf.libgguf_torch as libgguf_torch


DEFAULT_QTYPES = (
    "Q4_0",
    "Q4_1",
    "Q5_0",
    "Q5_1",
    "Q8_0",
    "Q2_K",
    "Q3_K",
    "Q4_K",
    "Q5_K",
    "Q6_K",
    "IQ4_NL",
    "IQ4_XS",
    "TQ1_0",
    "TQ2_0",
    "MXFP4",
    "NVFP4",
)


def parse_qtypes(value: str) -> list[GGMLQuantizationType]:
    if value.lower() == "default":
        names = list(DEFAULT_QTYPES)
    elif value.lower() == "all":
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


def make_rows(qtype: GGMLQuantizationType, rows: int, blocks_per_row: int, device: torch.device) -> torch.Tensor:
    block_size, _ = GGML_QUANT_SIZES[qtype]
    width = block_size * blocks_per_row
    index = torch.arange(rows * width, device=device, dtype=torch.float32).reshape(rows, width)
    wave = torch.sin(index * 0.013 + float(qtype.value)) + torch.cos(index * 0.007)
    ramp = ((index.remainder(17.0)) - 8.0) / 6.0
    return (wave * 0.75 + ramp).contiguous()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_op(fn: Any, *, iterations: int, warmup: int, device: torch.device) -> tuple[float, Any]:
    result = None
    for _ in range(warmup):
        result = fn()
    sync(device)

    started = perf_counter()
    for _ in range(iterations):
        result = fn()
    sync(device)
    return (perf_counter() - started) / float(iterations), result


def benchmark_qtype(
    qtype: GGMLQuantizationType,
    *,
    rows: int,
    blocks_per_row: int,
    iterations: int,
    warmup: int,
    device: torch.device,
    compile_ops: bool,
) -> list[dict[str, Any]]:
    data = make_rows(qtype, rows, blocks_per_row, device)
    elements = data.numel()
    quantize_op = lambda input_data: libgguf_torch.quantize(input_data, qtype)
    if compile_ops:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this torch installation")
        quantize_op = torch.compile(quantize_op, dynamic=False)

    quant_time, quantized = time_op(
        lambda: quantize_op(data),
        iterations=iterations,
        warmup=warmup,
        device=device,
    )

    encoded_bytes = quantized.numel() * quantized.element_size()
    results = [
        {
            "qtype": qtype.name,
            "operation": "quantize",
            "device": str(device),
            "rows": rows,
            "width": data.shape[-1],
            "iterations": iterations,
            "compiled": compile_ops,
            "seconds_per_iter": quant_time,
            "elements_per_second": elements / quant_time,
            "encoded_mb_per_second": (encoded_bytes / 1_000_000.0) / quant_time,
        }
    ]

    dequantize_op = lambda input_data: libgguf_torch.dequantize(input_data, qtype, data.shape, dtype=torch.float32)
    if compile_ops:
        dequantize_op = torch.compile(dequantize_op, dynamic=False)

    dequant_time, dequantized = time_op(
        lambda: dequantize_op(quantized),
        iterations=iterations,
        warmup=warmup,
        device=device,
    )
    decoded_bytes = dequantized.numel() * dequantized.element_size()
    results.append(
        {
            "qtype": qtype.name,
            "operation": "dequantize",
            "device": str(device),
            "rows": rows,
            "width": data.shape[-1],
            "iterations": iterations,
            "compiled": compile_ops,
            "seconds_per_iter": dequant_time,
            "elements_per_second": elements / dequant_time,
            "decoded_mb_per_second": (decoded_bytes / 1_000_000.0) / dequant_time,
        }
    )
    return results


def format_rate(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:9.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:9.2f}K"
    return f"{value:10.2f}"


def print_table(results: list[dict[str, Any]]) -> None:
    print(
        f"{'qtype':<10} {'op':<10} {'device':<8} {'shape':<14} "
        f"{'ms/iter':>10} {'elem/s':>11} {'MB/s':>11}"
    )
    print("-" * 80)
    for row in results:
        mbps = row.get("encoded_mb_per_second", row.get("decoded_mb_per_second", 0.0))
        shape = f"{row['rows']}x{row['width']}"
        print(
            f"{row['qtype']:<10} {row['operation']:<10} {row['device']:<8} {shape:<14} "
            f"{row['seconds_per_iter'] * 1000.0:10.3f} "
            f"{format_rate(row['elements_per_second']):>11} "
            f"{mbps:10.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark libgguf.libgguf_torch quantization kernels.")
    parser.add_argument("--qtypes", type=parse_qtypes, default=parse_qtypes("default"))
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--blocks-per-row", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compile", action="store_true", help="Benchmark torch.compile-wrapped kernels")
    parser.add_argument("--json", type=Path, default=None, help="Optional path for JSON results.")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    results: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for qtype in args.qtypes:
        try:
            results.extend(
                benchmark_qtype(
                    qtype,
                    rows=args.rows,
                    blocks_per_row=args.blocks_per_row,
                    iterations=args.iterations,
                    warmup=args.warmup,
                    device=device,
                    compile_ops=args.compile,
                )
            )
        except (KeyError, NotImplementedError, RuntimeError, ValueError) as exc:
            skipped.append({"qtype": qtype.name, "reason": str(exc)})

    print_table(results)
    if skipped:
        print("\nskipped:")
        for item in skipped:
            print(f"  {item['qtype']}: {item['reason']}")

    if args.json is not None:
        payload = {
            "config": {
                "device": str(device),
                "rows": args.rows,
                "blocks_per_row": args.blocks_per_row,
                "iterations": args.iterations,
                "warmup": args.warmup,
                "compile": args.compile,
            },
            "results": results,
            "skipped": skipped,
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
