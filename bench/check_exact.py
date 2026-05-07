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
GOLDEN_MANIFEST_VERSION = 1
GOLDEN_FIXTURE = "libgguf-native-cpu-encoded-sha256"
GOLDEN_QTYPES = ("Q4_0", "Q8_0", "Q4_K", "Q5_K", "Q6_K", "IQ2_XS", "IQ4_NL")
GOLDEN_SHAPES = ((1, 4096), (4, 4096))
GOLDEN_PATTERNS = (
    "zeros",
    "absmax_tie_positive_first",
    "absmax_tie_negative_first",
    "outlier",
    "random_normal_seed0",
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


def _pattern_rows(pattern: str, rows: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if pattern == "zeros":
        data = np.zeros((rows, width), dtype=np.float32)
    elif pattern == "constants":
        data = np.full((rows, width), 0.25, dtype=np.float32)
    elif pattern in {"absmax_ties", "absmax_tie_positive_first"}:
        data = np.zeros((rows, width), dtype=np.float32)
        data[:, 0::4] = 1.5
        data[:, 1::4] = -1.5
        data[:, 2::4] = 0.5
        data[:, 3::4] = -0.5
    elif pattern == "absmax_tie_negative_first":
        data = np.zeros((rows, width), dtype=np.float32)
        data[:, 0::4] = -1.5
        data[:, 1::4] = 1.5
        data[:, 2::4] = -0.5
        data[:, 3::4] = 0.5
    elif pattern == "outlier":
        data = np.random.default_rng(0).normal(0.0, 0.1, size=(rows, width)).astype(np.float32)
        data[:, 0::64] = 8.0
        data[:, 1::64] = -8.0
    elif pattern == "outliers":
        data = rng.normal(0.0, 0.1, size=(rows, width)).astype(np.float32)
        data[:, 0::64] = 8.0
        data[:, 1::64] = -8.0
    elif pattern == "random_normal_seed0":
        data = np.random.default_rng(0).normal(0.0, 0.75, size=(rows, width)).astype(np.float32)
    elif pattern == "normal":
        data = rng.normal(0.0, 0.75, size=(rows, width)).astype(np.float32)
    elif pattern == "uniform":
        data = rng.uniform(-2.0, 2.0, size=(rows, width)).astype(np.float32)
    elif pattern == "tiny":
        data = rng.normal(0.0, 1.0e-5, size=(rows, width)).astype(np.float32)
    elif pattern == "large":
        data = rng.normal(0.0, 16.0, size=(rows, width)).astype(np.float32)
    else:
        raise argparse.ArgumentTypeError(f"unknown case: {pattern}")
    return np.ascontiguousarray(data, dtype=np.float32)


def _case_rows(case: str, rows: int, width: int, seed: int) -> np.ndarray:
    return _pattern_rows(case, rows, width, seed)


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


def _golden_qtypes() -> list[GGMLQuantizationType]:
    return [GGMLQuantizationType[name] for name in GOLDEN_QTYPES]


def _run_golden_checks() -> list[ExactnessResult]:
    return _run_checks(_golden_qtypes(), GOLDEN_SHAPES, GOLDEN_PATTERNS)


def _manifest_entry(result: ExactnessResult) -> dict[str, object]:
    data = asdict(result)
    data["pattern"] = data.pop("case")
    return data


def _manifest_from_results(results: Iterable[ExactnessResult]) -> dict[str, object]:
    return {
        "version": GOLDEN_MANIFEST_VERSION,
        "generator": "scripts/update_golden.py",
        "fixture": GOLDEN_FIXTURE,
        "qtypes": list(GOLDEN_QTYPES),
        "shapes": [_shape_id(shape) for shape in GOLDEN_SHAPES],
        "patterns": list(GOLDEN_PATTERNS),
        "entries": [_manifest_entry(result) for result in results],
    }


def golden_manifest() -> dict[str, object]:
    return _manifest_from_results(_run_golden_checks())


def _load_expected(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected fixture must be a list: {path}")
    return data


def _entry_key(entry: dict[str, object]) -> tuple[str, str, str]:
    return (str(entry.get("qtype")), str(entry.get("shape")), str(entry.get("pattern")))


def _entries_from_manifest(manifest: dict[str, object]) -> list[dict[str, object]]:
    if manifest.get("version") != GOLDEN_MANIFEST_VERSION:
        raise ValueError(f"unsupported golden manifest version: {manifest.get('version')!r}")
    if manifest.get("fixture") != GOLDEN_FIXTURE:
        raise ValueError(f"unsupported golden fixture: {manifest.get('fixture')!r}")
    if manifest.get("qtypes") != list(GOLDEN_QTYPES):
        raise ValueError("golden manifest qtypes do not match the frozen set")
    if manifest.get("shapes") != [_shape_id(shape) for shape in GOLDEN_SHAPES]:
        raise ValueError("golden manifest shapes do not match the frozen set")
    if manifest.get("patterns") != list(GOLDEN_PATTERNS):
        raise ValueError("golden manifest patterns do not match the frozen set")

    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("golden manifest entries must be a list")
    entries = [dict(entry) for entry in raw_entries]
    expected_keys = {
        (qtype, shape, pattern)
        for qtype in GOLDEN_QTYPES
        for shape in [_shape_id(shape) for shape in GOLDEN_SHAPES]
        for pattern in GOLDEN_PATTERNS
    }
    seen_keys: set[tuple[str, str, str]] = set()
    for entry in entries:
        key = _entry_key(entry)
        if key in seen_keys:
            raise ValueError(f"duplicate golden manifest entry: {key}")
        seen_keys.add(key)
        if key not in expected_keys:
            raise ValueError(f"unexpected golden manifest entry: {key}")
    missing = sorted(expected_keys - seen_keys)
    if missing:
        raise ValueError(f"missing golden manifest entries: {missing[:3]}")
    return entries


def _load_manifest(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError(f"golden manifest must be an object: {path}")
    _entries_from_manifest(manifest)
    return manifest


def _compare_entries(observed: dict[str, object], expected: dict[str, object]) -> list[str]:
    observed_entries = {_entry_key(entry): entry for entry in _entries_from_manifest(observed)}
    expected_entries = {_entry_key(entry): entry for entry in _entries_from_manifest(expected)}
    messages: list[str] = []
    for key in sorted(expected_entries):
        observed_entry = observed_entries[key]
        expected_entry = expected_entries[key]
        for field in ("encoded_nbytes", "encoded_sha256", "decoded_all_finite"):
            if observed_entry.get(field) != expected_entry.get(field):
                messages.append(
                    f"{key[0]} {key[1]} {key[2]} {field} "
                    f"expected={expected_entry.get(field)!r} observed={observed_entry.get(field)!r}"
                )
                break
    return messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check deterministic GGUF row exactness hashes.")
    parser.add_argument("--qtypes", type=_parse_qtypes, help="Comma-separated qtypes, e.g. Q4_K,Q5_K")
    parser.add_argument("--shapes", type=_parse_shapes, help="Comma-separated row shapes, e.g. 4x4096")
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES), help="Comma-separated edge cases")
    parser.add_argument("--golden", action="store_true", help="Use the frozen golden qtype, shape, and pattern set")
    parser.add_argument("--write-json", type=Path, help="Write observed hashes to a JSON fixture")
    parser.add_argument("--expect-json", type=Path, help="Compare observed hashes with a JSON fixture")
    parser.add_argument("--write-manifest", type=Path, help="Write observed hashes to a golden manifest")
    parser.add_argument("--expect-manifest", type=Path, help="Compare observed hashes with a golden manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.golden or args.write_manifest is not None or args.expect_manifest is not None:
        qtypes = _golden_qtypes()
        shapes = list(GOLDEN_SHAPES)
        cases = list(GOLDEN_PATTERNS)
    else:
        if args.qtypes is None:
            raise SystemExit("--qtypes is required unless --golden is used")
        if args.shapes is None:
            raise SystemExit("--shapes is required unless --golden is used")
        qtypes = args.qtypes
        shapes = args.shapes
        cases = _parse_csv(args.cases)
    if not cases:
        raise SystemExit("at least one case is required")
    results = _run_checks(qtypes, shapes, cases)
    result_dicts = [asdict(result) for result in results]
    manifest = _manifest_from_results(results)

    if args.expect_json is not None:
        expected = _load_expected(args.expect_json)
        if result_dicts != expected:
            raise SystemExit(f"exactness mismatch against {args.expect_json}")

    if args.expect_manifest is not None:
        expected_manifest = _load_manifest(args.expect_manifest)
        messages = _compare_entries(manifest, expected_manifest)
        if messages:
            details = "\n".join(messages[:10])
            raise SystemExit(f"exactness mismatch against {args.expect_manifest}\n{details}")

    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        with args.write_json.open("w", encoding="utf-8") as f:
            json.dump(result_dicts, f, indent=2)
            f.write("\n")

    if args.write_manifest is not None:
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.write_manifest.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")

    for result in results:
        finite = "n/a" if result.decoded_all_finite is None else str(result.decoded_all_finite).lower()
        print(
            f"{result.qtype} {result.shape} {result.case} "
            f"bytes={result.encoded_nbytes} sha256={result.encoded_sha256} decoded_finite={finite}"
        )


if __name__ == "__main__":
    main()
