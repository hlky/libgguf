from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

import libgguf
from libgguf import GGMLQuantizationType, GGML_QUANT_SIZES


DEFAULT_CASES = (
    "zeros",
    "constants",
    "absmax_ties",
    "outliers",
    "normal",
    "uniform",
    "tiny",
    "large",
)


@dataclass(frozen=True)
class ExactnessResult:
    qtype: str
    shape: str
    case: str
    encoded_nbytes: int
    encoded_sha256: str
    decoded_all_finite: bool | None


def _parse_csv(values: str) -> list[str]:
    return [value.strip() for value in values.split(",") if value.strip()]


def _parse_qtypes(values: str) -> list[GGMLQuantizationType]:
    qtypes: list[GGMLQuantizationType] = []
    for name in _parse_csv(values):
        try:
            qtypes.append(GGMLQuantizationType[name])
        except KeyError as exc:
            raise argparse.ArgumentTypeError(f"unknown qtype: {name}") from exc
    if not qtypes:
        raise argparse.ArgumentTypeError("at least one qtype is required")
    return qtypes


def _parse_shapes(values: str) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for value in _parse_csv(values):
        try:
            rows, width = (int(part) for part in value.lower().split("x", 1))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid shape: {value}") from exc
        if rows <= 0 or width <= 0:
            raise argparse.ArgumentTypeError(f"shape must be positive: {value}")
        shapes.append((rows, width))
    if not shapes:
        raise argparse.ArgumentTypeError("at least one shape is required")
    return shapes


def _case_rows(case: str, rows: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if case == "zeros":
        data = np.zeros((rows, width), dtype=np.float32)
    elif case == "constants":
        data = np.full((rows, width), 0.25, dtype=np.float32)
    elif case == "absmax_ties":
        data = np.zeros((rows, width), dtype=np.float32)
        data[:, 0::4] = 1.5
        data[:, 1::4] = -1.5
        data[:, 2::4] = 0.5
        data[:, 3::4] = -0.5
    elif case == "outliers":
        data = rng.normal(0.0, 0.1, size=(rows, width)).astype(np.float32)
        data[:, 0::64] = 8.0
        data[:, 1::64] = -8.0
    elif case == "normal":
        data = rng.normal(0.0, 0.75, size=(rows, width)).astype(np.float32)
    elif case == "uniform":
        data = rng.uniform(-2.0, 2.0, size=(rows, width)).astype(np.float32)
    elif case == "tiny":
        data = rng.normal(0.0, 1.0e-5, size=(rows, width)).astype(np.float32)
    elif case == "large":
        data = rng.normal(0.0, 16.0, size=(rows, width)).astype(np.float32)
    else:
        raise argparse.ArgumentTypeError(f"unknown case: {case}")
    return np.ascontiguousarray(data, dtype=np.float32)


def _encoded_bytes(encoded: np.ndarray) -> bytes:
    return np.ascontiguousarray(encoded).view(np.uint8).tobytes()


def _shape_id(shape: tuple[int, int]) -> str:
    return f"{shape[0]}x{shape[1]}"


def _check_one(qtype: GGMLQuantizationType, shape: tuple[int, int], case: str) -> ExactnessResult:
    rows, width = shape
    block_size, _ = GGML_QUANT_SIZES[qtype]
    if width % block_size != 0:
        raise ValueError(f"{_shape_id(shape)} is not divisible by {qtype.name} block size {block_size}")

    seed = qtype.value * 1009 + rows * 131 + width * 17 + sum(case.encode("utf-8"))
    data = _case_rows(case, rows, width, seed)
    encoded = libgguf.quantize_rows(data, qtype)
    encoded_bytes = _encoded_bytes(encoded)

    decoded_all_finite: bool | None = None
    if block_size > 1:
        decoded = libgguf.dequantize_rows(encoded, qtype, n_per_row=width)
        decoded_all_finite = bool(np.all(np.isfinite(decoded)))

    return ExactnessResult(
        qtype=qtype.name,
        shape=_shape_id(shape),
        case=case,
        encoded_nbytes=len(encoded_bytes),
        encoded_sha256=hashlib.sha256(encoded_bytes).hexdigest(),
        decoded_all_finite=decoded_all_finite,
    )


def _run_checks(
    qtypes: Iterable[GGMLQuantizationType],
    shapes: Iterable[tuple[int, int]],
    cases: Iterable[str],
) -> list[ExactnessResult]:
    return [_check_one(qtype, shape, case) for qtype in qtypes for shape in shapes for case in cases]


def _load_expected(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected fixture must be a list: {path}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check deterministic GGUF row exactness hashes.")
    parser.add_argument("--qtypes", required=True, type=_parse_qtypes, help="Comma-separated qtypes, e.g. Q4_K,Q5_K")
    parser.add_argument("--shapes", required=True, type=_parse_shapes, help="Comma-separated row shapes, e.g. 4x4096")
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES), help="Comma-separated edge cases")
    parser.add_argument("--write-json", type=Path, help="Write observed hashes to a JSON fixture")
    parser.add_argument("--expect-json", type=Path, help="Compare observed hashes with a JSON fixture")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = _parse_csv(args.cases)
    if not cases:
        raise SystemExit("at least one case is required")
    results = _run_checks(args.qtypes, args.shapes, cases)
    result_dicts = [asdict(result) for result in results]

    if args.expect_json is not None:
        expected = _load_expected(args.expect_json)
        if result_dicts != expected:
            raise SystemExit(f"exactness mismatch against {args.expect_json}")

    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        with args.write_json.open("w", encoding="utf-8") as f:
            json.dump(result_dicts, f, indent=2)
            f.write("\n")

    for result in results:
        finite = "n/a" if result.decoded_all_finite is None else str(result.decoded_all_finite).lower()
        print(
            f"{result.qtype} {result.shape} {result.case} "
            f"bytes={result.encoded_nbytes} sha256={result.encoded_sha256} decoded_finite={finite}"
        )


if __name__ == "__main__":
    main()
