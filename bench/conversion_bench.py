from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = ROOT / "bench" / "results"
TIMING_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_-]*)\s*(?:=|:|\s)\s*"
    r"(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>us|ms|s|sec|secs|seconds)?\b",
    re.IGNORECASE,
)
COUNT_RE = re.compile(r"\b([A-Z][A-Z0-9_]+)\s*=\s*([0-9]+)\b")
TIMING_KEYS = {
    "total",
    "metadata",
    "read",
    "quant",
    "cpu_convert",
    "h2d",
    "cuda_quant",
    "d2h",
    "write",
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_converter(path_or_name: str) -> str:
    path = Path(path_or_name)
    if path.exists():
        return str(path)
    resolved = shutil.which(path_or_name)
    if resolved is None:
        raise FileNotFoundError(f"converter executable not found: {path_or_name}")
    return resolved


def seconds_from(value: str, unit: str | None) -> float:
    number = float(value)
    normalized = (unit or "s").lower()
    if normalized == "us":
        return number / 1_000_000.0
    if normalized == "ms":
        return number / 1_000.0
    return number


def parse_timings(stderr: str) -> dict[str, float]:
    timings: dict[str, float] = {}
    for match in TIMING_RE.finditer(stderr):
        key = match.group("key").lower().replace("-", "_")
        value = seconds_from(match.group("value"), match.group("unit"))
        if key in TIMING_KEYS:
            timings[f"{key}_s"] = value
    return timings


def parse_key_value_counts(text: str, prefix: str) -> dict[str, int]:
    for line in text.splitlines():
        if line.lower().startswith(prefix.lower()):
            return {name: int(count) for name, count in COUNT_RE.findall(line)}
    return {}


def unique_output_path(run_dir: Path, src: Path, qtype: str, run_index: int) -> Path:
    stem = src.name
    if stem.endswith(".safetensors"):
        stem = stem[: -len(".safetensors")]
    return run_dir / f"{stem}-{qtype}-run{run_index:03d}.gguf"


def run_conversion(
    *,
    converter: str,
    src: Path,
    dst: Path,
    qtype: str,
    policy: str,
    backend: str,
    run_index: int,
    threads: int | None,
    scratch_bytes: int | None,
    extra_args: list[str],
) -> dict[str, Any]:
    command = [
        converter,
        "--src",
        str(src),
        "--dst",
        str(dst),
        "--qtype",
        qtype,
        "--policy",
        policy,
        "--overwrite",
        "--timings",
    ]
    if threads is not None:
        command.extend(["--threads", str(threads)])
    if scratch_bytes is not None:
        command.extend(["--scratch-bytes", str(scratch_bytes)])
    command.extend(extra_args)

    started = perf_counter()
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    wall_s = perf_counter() - started
    output_size = dst.stat().st_size if dst.exists() else None
    timings = parse_timings(result.stderr)

    row: dict[str, Any] = {
        "run": run_index,
        "backend": backend,
        "converter": converter,
        "src": str(src),
        "dst": str(dst),
        "qtype": qtype,
        "policy": policy,
        "returncode": result.returncode,
        "wall_s": wall_s,
        "total_s": timings.get("total_s"),
        "metadata_s": timings.get("metadata_s"),
        "read_s": timings.get("read_s"),
        "quant_s": timings.get("quant_s"),
        "cpu_convert_s": timings.get("cpu_convert_s"),
        "h2d_s": timings.get("h2d_s"),
        "cuda_quant_s": timings.get("cuda_quant_s"),
        "d2h_s": timings.get("d2h_s"),
        "write_s": timings.get("write_s"),
        "output_size_bytes": output_size,
        "qtype_counts": parse_key_value_counts(result.stdout, "Tensor types:"),
        "fallback_counts": parse_key_value_counts(result.stdout, "Fallbacks:"),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": command,
    }
    if result.returncode != 0:
        row["error"] = result.stderr.strip() or result.stdout.strip() or f"converter exited with {result.returncode}"
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["returncode"] == 0]
    total_values = [row["total_s"] for row in successful if isinstance(row.get("total_s"), float)]
    wall_values = [row["wall_s"] for row in successful]
    sizes = [row["output_size_bytes"] for row in successful if isinstance(row.get("output_size_bytes"), int)]
    return {
        "runs": len(rows),
        "successful_runs": len(successful),
        "failed_runs": len(rows) - len(successful),
        "total_s_min": min(total_values) if total_values else None,
        "total_s_mean": (sum(total_values) / len(total_values)) if total_values else None,
        "wall_s_min": min(wall_values) if wall_values else None,
        "wall_s_mean": (sum(wall_values) / len(wall_values)) if wall_values else None,
        "output_size_bytes": sizes[0] if sizes and all(size == sizes[0] for size in sizes) else None,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run",
        "backend",
        "qtype",
        "policy",
        "returncode",
        "wall_s",
        "total_s",
        "metadata_s",
        "read_s",
        "quant_s",
        "cpu_convert_s",
        "h2d_s",
        "cuda_quant_s",
        "d2h_s",
        "write_s",
        "output_size_bytes",
        "qtype_counts_json",
        "fallback_counts_json",
        "dst",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{name: row.get(name) for name in fieldnames},
                    "qtype_counts_json": json.dumps(row.get("qtype_counts", {}), sort_keys=True),
                    "fallback_counts_json": json.dumps(row.get("fallback_counts", {}), sort_keys=True),
                }
            )


def print_table(rows: list[dict[str, Any]]) -> None:
    print(f"{'run':>3} {'backend':<8} {'qtype':<10} {'total s':>10} {'wall s':>10} {'size MB':>10} {'status':>7}")
    print("-" * 68)
    for row in rows:
        total = row.get("total_s")
        total_text = f"{total:10.3f}" if isinstance(total, float) else f"{'':>10}"
        size = row.get("output_size_bytes")
        size_text = f"{size / 1_000_000.0:10.2f}" if isinstance(size, int) else f"{'':>10}"
        status = "ok" if row["returncode"] == 0 else "fail"
        print(
            f"{row['run']:3d} {row['backend']:<8} {row['qtype']:<10} {total_text} "
            f"{row['wall_s']:10.3f} {size_text} {status:>7}"
        )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark libgguf_quantize_gguf end-to-end conversion.")
    parser.add_argument("--src", type=Path, required=True, help="Local .safetensors input path.")
    parser.add_argument("--qtype", required=True, help="Output qtype/file type, for example Q4_K_M or Q8_0.")
    parser.add_argument("--converter", default=os.environ.get("LIBGGUF_QUANTIZE_GGUF_EXE", "libgguf_quantize_gguf"))
    parser.add_argument("--policy", default="comfy", choices=("comfy", "dynamic", "uniform"))
    parser.add_argument("--runs", type=positive_int, default=1, help="Number of repeated conversion runs.")
    parser.add_argument("--backend", default="native", help="Result label for future native/cuda comparisons.")
    parser.add_argument("--threads", type=positive_int, default=None)
    parser.add_argument("--scratch-bytes", type=positive_int, default=None)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", default=None, help="Optional results subdirectory name.")
    parser.add_argument(
        "--converter-arg",
        action="append",
        default=[],
        help="Extra argument passed to the converter after standard arguments; repeat as needed.",
    )
    args = parser.parse_args(argv)

    src = args.src.resolve()
    if not src.is_file():
        parser.error(f"--src does not exist or is not a file: {src}")

    try:
        converter = resolve_converter(args.converter)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    run_name = args.run_name or utc_timestamp()
    run_dir = args.results_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    rows = [
        run_conversion(
            converter=converter,
            src=src,
            dst=unique_output_path(run_dir, src, args.qtype, run_index),
            qtype=args.qtype,
            policy=args.policy,
            backend=args.backend,
            run_index=run_index,
            threads=args.threads,
            scratch_bytes=args.scratch_bytes,
            extra_args=args.converter_arg,
        )
        for run_index in range(1, args.runs + 1)
    ]

    payload = {
        "config": {
            "src": str(src),
            "qtype": args.qtype,
            "policy": args.policy,
            "runs": args.runs,
            "backend": args.backend,
            "converter": converter,
            "threads": args.threads,
            "scratch_bytes": args.scratch_bytes,
            "extra_args": args.converter_arg,
            "results_dir": str(run_dir),
        },
        "summary": summarize(rows),
        "results": rows,
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(run_dir / "summary.csv", rows)

    print_table(rows)
    print(run_dir)
    return 0 if all(row["returncode"] == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
