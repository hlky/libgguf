from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
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
BYTE_KEYS = {
    "cuda_vram",
    "cuda_max_input",
    "cuda_max_output",
}
COUNT_KEYS = {
    "cuda_chunks",
    "cuda_pipeline",
}
COMPARISON_NOTE = (
    "One-run comparison; read and write timings are storage/cache sensitive. "
    "total_speedup is CPU total / CUDA total. encode_speedup is CPU encode / CUDA encode."
)


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


def converter_command_prefix(converter: str) -> list[str]:
    if Path(converter).suffix.lower() == ".py":
        return [sys.executable, converter]
    return [converter]


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
        if key in TIMING_KEYS:
            value = seconds_from(match.group("value"), match.group("unit"))
            timings[f"{key}_s"] = value
        elif key in BYTE_KEYS:
            timings[f"{key}_bytes"] = int(float(match.group("value")))
        elif key in COUNT_KEYS:
            timings[key] = int(float(match.group("value")))
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
    delete_output: bool,
) -> dict[str, Any]:
    command = [
        *converter_command_prefix(converter),
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
    output_deleted = False
    if delete_output and dst.exists():
        dst.unlink()
        output_deleted = True

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
        "cuda_vram_bytes": timings.get("cuda_vram_bytes"),
        "cuda_max_input_bytes": timings.get("cuda_max_input_bytes"),
        "cuda_max_output_bytes": timings.get("cuda_max_output_bytes"),
        "cuda_chunks": timings.get("cuda_chunks"),
        "cuda_pipeline": timings.get("cuda_pipeline"),
        "output_size_bytes": output_size,
        "output_deleted": output_deleted,
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


def load_aggregate(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"aggregate file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"invalid aggregate in {path}: top-level JSON must be an object")
    if not isinstance(payload.get("config"), dict):
        raise ValueError(f"invalid aggregate in {path}: missing object field 'config'")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"invalid aggregate in {path}: missing array field 'results'")

    seen: set[str] = set()
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            raise ValueError(f"invalid aggregate in {path}: results[{index}] must be an object")
        qtype = row.get("qtype")
        if not isinstance(qtype, str) or not qtype:
            raise ValueError(f"invalid aggregate in {path}: results[{index}] is missing qtype")
        if qtype in seen:
            raise ValueError(f"invalid aggregate in {path}: duplicate qtype {qtype!r}")
        seen.add(qtype)
    return payload


def numeric_value(row: dict[str, Any] | None, key: str) -> float | None:
    if row is None:
        return None
    value = row.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def int_value(row: dict[str, Any] | None, key: str) -> int | None:
    if row is None:
        return None
    value = row.get(key)
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def bool_value(row: dict[str, Any] | None, key: str) -> bool | None:
    if row is None:
        return None
    value = row.get(key)
    return value if isinstance(value, bool) else None


def speedup(cpu_value: float | None, cuda_value: float | None) -> float | None:
    if cpu_value is None or cuda_value is None or cuda_value == 0:
        return None
    return cpu_value / cuda_value


def rows_by_qtype(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["qtype"]: row for row in payload["results"]}


def qtype_join_order(cpu_payload: dict[str, Any], cuda_payload: dict[str, Any]) -> list[str]:
    cpu_order = [row["qtype"] for row in cpu_payload["results"]]
    cuda_qtypes = {row["qtype"] for row in cuda_payload["results"]}
    return [*cpu_order, *sorted(cuda_qtypes.difference(cpu_order))]


def compare_aggregates(cpu_payload: dict[str, Any], cuda_payload: dict[str, Any]) -> list[dict[str, Any]]:
    cpu_rows = rows_by_qtype(cpu_payload)
    cuda_rows = rows_by_qtype(cuda_payload)
    joined: list[dict[str, Any]] = []
    for qtype in qtype_join_order(cpu_payload, cuda_payload):
        cpu_row = cpu_rows.get(qtype)
        cuda_row = cuda_rows.get(qtype)
        cpu_total = numeric_value(cpu_row, "total_s")
        cuda_total = numeric_value(cuda_row, "total_s")
        cpu_encode = numeric_value(cpu_row, "encode_s")
        cuda_encode = numeric_value(cuda_row, "encode_s")
        output_size = int_value(cpu_row, "output_size_bytes")
        if output_size is None:
            output_size = int_value(cuda_row, "output_size_bytes")
        joined.append(
            {
                "qtype": qtype,
                "cpu_total_s": cpu_total,
                "cuda_total_s": cuda_total,
                "total_speedup_cuda_vs_cpu": speedup(cpu_total, cuda_total),
                "cpu_read_s": numeric_value(cpu_row, "read_s"),
                "cuda_read_s": numeric_value(cuda_row, "read_s"),
                "cpu_encode_s": cpu_encode,
                "cuda_encode_s": cuda_encode,
                "encode_speedup_cuda_vs_cpu": speedup(cpu_encode, cuda_encode),
                "cpu_write_s": numeric_value(cpu_row, "write_s"),
                "cuda_write_s": numeric_value(cuda_row, "write_s"),
                "cuda_chunks": int_value(cuda_row, "cuda_chunks"),
                "cuda_pipeline": int_value(cuda_row, "cuda_pipeline"),
                "cuda_vram_bytes": int_value(cuda_row, "cuda_vram_bytes"),
                "cuda_max_input_bytes": int_value(cuda_row, "cuda_max_input_bytes"),
                "cuda_max_output_bytes": int_value(cuda_row, "cuda_max_output_bytes"),
                "output_size_bytes": output_size,
                "output_size_gb": (output_size / 1_000_000_000.0) if output_size is not None else None,
                "cpu_output_deleted": bool_value(cpu_row, "output_deleted"),
                "cuda_output_deleted": bool_value(cuda_row, "output_deleted"),
            }
        )
    return joined


def first_present_config(cpu_payload: dict[str, Any], cuda_payload: dict[str, Any], key: str) -> Any:
    cpu_value = cpu_payload["config"].get(key)
    if cpu_value is not None:
        return cpu_value
    return cuda_payload["config"].get(key)


def comparison_payload(cpu_path: Path, cuda_path: Path, cpu_payload: dict[str, Any], cuda_payload: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {
        "benchmark": "CPU vs CUDA conversion qtype comparison",
        "cpu_results": str(cpu_path),
        "cuda_results": str(cuda_path),
        "note": COMPARISON_NOTE,
    }
    for key in ("src", "policy", "runs_per_qtype"):
        value = first_present_config(cpu_payload, cuda_payload, key)
        if value is not None:
            config[key] = value
    return {
        "config": config,
        "results": compare_aggregates(cpu_payload, cuda_payload),
    }


def format_seconds(value: Any) -> str:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f"{float(value):.3f}".rstrip("0").rstrip(".")
    return ""


def format_speedup(value: Any) -> str:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f"{float(value):.2f}".rstrip("0").rstrip(".") + "x"
    return ""


def format_gb(value: Any) -> str:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return f"{float(value):.2f}".rstrip("0").rstrip(".")
    return ""


def markdown_path(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def comparison_markdown(payload: dict[str, Any], *, cpu_path: Path, cuda_path: Path, out_path: Path) -> str:
    config = payload["config"]
    lines = ["# CPU vs CUDA Conversion Qtype Comparison", ""]
    if config.get("src") is not None:
        lines.append(f"- Source: `{config['src']}`")
    if config.get("policy") is not None:
        lines.append(f"- Policy: {config['policy']}")
    if config.get("runs_per_qtype") is not None:
        lines.append(f"- Runs: {config['runs_per_qtype']} per qtype per backend")
    lines.append("- Caveat: storage cache/order affects `read_s` and end-to-end totals; compare `encode_s` for converter work.")
    lines.extend(
        [
            "",
            "| qtype | CPU total s | CUDA total s | total speedup | CPU encode s | CUDA encode s | encode speedup | output GB |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["results"]:
        lines.append(
            "| "
            f"`{row['qtype']}` | "
            f"{format_seconds(row.get('cpu_total_s'))} | "
            f"{format_seconds(row.get('cuda_total_s'))} | "
            f"{format_speedup(row.get('total_speedup_cuda_vs_cpu'))} | "
            f"{format_seconds(row.get('cpu_encode_s'))} | "
            f"{format_seconds(row.get('cuda_encode_s'))} | "
            f"{format_speedup(row.get('encode_speedup_cuda_vs_cpu'))} | "
            f"{format_gb(row.get('output_size_gb'))} |"
        )
    cuda_metric_rows = [
        row
        for row in payload["results"]
        if any(
            row.get(key) is not None
            for key in (
                "cuda_chunks",
                "cuda_pipeline",
                "cuda_vram_bytes",
                "cuda_max_input_bytes",
                "cuda_max_output_bytes",
            )
        )
    ]
    if cuda_metric_rows:
        lines.extend(
            [
                "",
                "CUDA execution details:",
                "",
                "| qtype | CUDA chunks | CUDA pipeline | CUDA VRAM bytes | max input bytes | max output bytes |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in cuda_metric_rows:
            lines.append(
                "| "
                f"`{row['qtype']}` | "
                f"{row.get('cuda_chunks') if row.get('cuda_chunks') is not None else ''} | "
                f"{row.get('cuda_pipeline') if row.get('cuda_pipeline') is not None else ''} | "
                f"{row.get('cuda_vram_bytes') if row.get('cuda_vram_bytes') is not None else ''} | "
                f"{row.get('cuda_max_input_bytes') if row.get('cuda_max_input_bytes') is not None else ''} | "
                f"{row.get('cuda_max_output_bytes') if row.get('cuda_max_output_bytes') is not None else ''} |"
            )
    lines.extend(
        [
            "",
            "Saved artifacts:",
            "",
            f"- CPU aggregate: `{markdown_path(cpu_path, out_path.parent)}`",
            f"- CUDA aggregate: `{markdown_path(cuda_path, out_path.parent)}`",
            f"- Joined comparison: `aggregate.json` and this `{out_path.name}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_compare(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Compare CPU and CUDA conversion aggregate benchmark results.")
    parser.add_argument("--cpu", type=Path, required=True, help="CPU aggregate JSON path.")
    parser.add_argument("--cuda", type=Path, required=True, help="CUDA aggregate JSON path.")
    parser.add_argument("--out", type=Path, required=True, help="Markdown summary path to write.")
    args = parser.parse_args(argv)

    try:
        cpu_payload = load_aggregate(args.cpu)
        cuda_payload = load_aggregate(args.cuda)
    except ValueError as exc:
        parser.error(str(exc))

    payload = comparison_payload(args.cpu, args.cuda, cpu_payload, cuda_payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path = args.out.parent / "aggregate.json"
    aggregate_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.out.write_text(comparison_markdown(payload, cpu_path=args.cpu, cuda_path=args.cuda, out_path=args.out), encoding="utf-8")
    print(args.out)
    print(aggregate_path)
    return 0


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
        "cuda_vram_bytes",
        "cuda_max_input_bytes",
        "cuda_max_output_bytes",
        "cuda_chunks",
        "cuda_pipeline",
        "output_size_bytes",
        "output_deleted",
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


def split_passthrough_args(argv: list[str] | None) -> tuple[list[str], list[str]]:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in raw_argv:
        return raw_argv, []
    separator_index = raw_argv.index("--")
    return raw_argv[:separator_index], raw_argv[separator_index + 1 :]


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] == "compare":
        return run_compare(raw_argv[1:])

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
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional root for generated GGUF outputs. Defaults to the results run directory.",
    )
    parser.add_argument(
        "--delete-outputs",
        action="store_true",
        help="Delete each generated GGUF after recording its size and timings.",
    )
    parser.add_argument("--run-name", default=None, help="Optional results subdirectory name.")
    parser.add_argument(
        "--converter-arg",
        action="append",
        default=[],
        help=(
            "Extra argument passed to the converter after standard arguments; repeat as needed. "
            "Arguments after '--' are also forwarded."
        ),
    )
    benchmark_argv, passthrough_args = split_passthrough_args(raw_argv)
    args = parser.parse_args(benchmark_argv)
    extra_args = [*args.converter_arg, *passthrough_args]

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
    output_dir = (args.output_root / run_name) if args.output_root is not None else run_dir
    if output_dir != run_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        run_conversion(
            converter=converter,
            src=src,
            dst=unique_output_path(output_dir, src, args.qtype, run_index),
            qtype=args.qtype,
            policy=args.policy,
            backend=args.backend,
            run_index=run_index,
            threads=args.threads,
            scratch_bytes=args.scratch_bytes,
            extra_args=extra_args,
            delete_output=args.delete_outputs,
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
            "extra_args": extra_args,
            "results_dir": str(run_dir),
            "output_dir": str(output_dir),
            "delete_outputs": args.delete_outputs,
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
