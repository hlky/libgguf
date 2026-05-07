from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .inspect import GGUFFile, GGUFTensorInfo, inspect_gguf


_DESCRIPTOR_FIELDS = ("shape", "qtype", "qtype_value", "nbytes")


@dataclass(frozen=True)
class _Difference:
    kind: str
    message: str
    tensor_name: str | None = None
    metadata_key: str | None = None
    index: int | None = None
    left: Any = None
    right: Any = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": self.kind,
            "message": self.message,
        }
        if self.tensor_name is not None:
            data["tensor_name"] = self.tensor_name
        if self.metadata_key is not None:
            data["metadata_key"] = self.metadata_key
        if self.index is not None:
            data["index"] = self.index
        if self.left is not None:
            data["left"] = self.left
        if self.right is not None:
            data["right"] = self.right
        return data


@dataclass(frozen=True)
class _ComparisonResult:
    left_path: Path
    right_path: Path
    metadata: bool
    tensor_bytes: bool
    differences: tuple[_Difference, ...]

    @property
    def ok(self) -> bool:
        return not self.differences

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "paths": {
                "left": os.fspath(self.left_path),
                "right": os.fspath(self.right_path),
            },
            "modes": {
                "descriptors": True,
                "metadata": self.metadata,
                "tensor_bytes": self.tensor_bytes,
            },
            "differences": [difference.to_dict() for difference in self.differences],
        }


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.max_array_values is None:
        max_array_values = None if args.metadata else 0
    else:
        max_array_values = None if args.max_array_values < 0 else args.max_array_values
    result = _compare_gguf(
        args.left,
        args.right,
        metadata=args.metadata,
        tensor_bytes=args.tensor_bytes,
        max_array_values=max_array_values,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _print_result(result)

    if not result.ok:
        raise SystemExit(1)


def _compare_gguf(
    left_path: str | os.PathLike[str],
    right_path: str | os.PathLike[str],
    *,
    metadata: bool = False,
    tensor_bytes: bool = False,
    max_array_values: int | None = None,
) -> _ComparisonResult:
    left = inspect_gguf(left_path, max_array_values=max_array_values)
    right = inspect_gguf(right_path, max_array_values=max_array_values)

    differences: list[_Difference] = []
    differences.extend(_compare_tensor_descriptors(left, right))
    if metadata:
        differences.extend(_compare_metadata(left, right))
    if tensor_bytes:
        differences.extend(_compare_tensor_bytes(left, right))

    return _ComparisonResult(
        left_path=Path(left_path),
        right_path=Path(right_path),
        metadata=metadata,
        tensor_bytes=tensor_bytes,
        differences=tuple(differences),
    )


def _compare_tensor_descriptors(left: GGUFFile, right: GGUFFile) -> list[_Difference]:
    differences: list[_Difference] = []
    left_names = [tensor.name for tensor in left.tensors]
    right_names = [tensor.name for tensor in right.tensors]
    if left_names != right_names:
        differences.append(
            _Difference(
                kind="tensor_order",
                message="tensor names/order differ",
                left=left_names,
                right=right_names,
            )
        )

    for index, (left_tensor, right_tensor) in enumerate(zip(left.tensors, right.tensors)):
        if left_tensor.name != right_tensor.name:
            continue
        for field in _DESCRIPTOR_FIELDS:
            left_value = _descriptor_value(left_tensor, field)
            right_value = _descriptor_value(right_tensor, field)
            if left_value != right_value:
                differences.append(
                    _Difference(
                        kind=f"tensor_{field}",
                        message=f"tensor {left_tensor.name!r} {field} differs",
                        tensor_name=left_tensor.name,
                        index=index,
                        left=left_value,
                        right=right_value,
                    )
                )
    return differences


def _compare_metadata(left: GGUFFile, right: GGUFFile) -> list[_Difference]:
    differences: list[_Difference] = []
    left_metadata = {key: value.to_dict() for key, value in left.metadata.items()}
    right_metadata = {key: value.to_dict() for key, value in right.metadata.items()}

    for key in sorted(set(left_metadata) | set(right_metadata)):
        left_value = left_metadata.get(key)
        right_value = right_metadata.get(key)
        if left_value != right_value:
            differences.append(
                _Difference(
                    kind="metadata_value",
                    message=f"metadata {key!r} differs",
                    metadata_key=key,
                    left=left_value,
                    right=right_value,
                )
            )

    for key in sorted(set(left.metadata_key_counts) | set(right.metadata_key_counts)):
        left_count = left.metadata_key_counts.get(key, 0)
        right_count = right.metadata_key_counts.get(key, 0)
        if left_count != right_count:
            differences.append(
                _Difference(
                    kind="metadata_duplicate_count",
                    message=f"metadata {key!r} duplicate count differs",
                    metadata_key=key,
                    left=left_count,
                    right=right_count,
                )
            )
    return differences


def _compare_tensor_bytes(left: GGUFFile, right: GGUFFile) -> list[_Difference]:
    differences: list[_Difference] = []
    left_by_name = _unique_tensors_by_name(left)
    right_by_name = _unique_tensors_by_name(right)
    duplicate_names = _duplicate_tensor_names(left) | _duplicate_tensor_names(right)

    for name in sorted((set(left_by_name) & set(right_by_name)) | duplicate_names):
        if name in duplicate_names:
            differences.append(
                _Difference(
                    kind="tensor_bytes_skipped",
                    message=f"tensor {name!r} has duplicate descriptors; raw bytes are ambiguous",
                    tensor_name=name,
                )
            )
            continue

        left_tensor = left_by_name[name]
        right_tensor = right_by_name[name]
        if not _byte_descriptors_match(left_tensor, right_tensor):
            differences.append(
                _Difference(
                    kind="tensor_bytes_skipped",
                    message=f"tensor {name!r} descriptors differ; raw bytes were not compared",
                    tensor_name=name,
                    left=_descriptor_summary(left_tensor),
                    right=_descriptor_summary(right_tensor),
                )
            )
            continue

        if left_tensor.nbytes is None:
            differences.append(
                _Difference(
                    kind="tensor_bytes_skipped",
                    message=f"tensor {name!r} byte length is unknown; raw bytes were not compared",
                    tensor_name=name,
                    left=_descriptor_summary(left_tensor),
                    right=_descriptor_summary(right_tensor),
                )
            )
            continue

        left_bytes = left.read_tensor_bytes(left_tensor)
        right_bytes = right.read_tensor_bytes(right_tensor)
        if left_bytes != right_bytes:
            differences.append(
                _Difference(
                    kind="tensor_bytes",
                    message=f"tensor {name!r} raw payload bytes differ",
                    tensor_name=name,
                    left=_payload_summary(left_bytes),
                    right=_payload_summary(right_bytes),
                )
            )
    return differences


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GGUF tensor descriptors and optional exact content")
    parser.add_argument("left", help="Left GGUF file")
    parser.add_argument("right", help="Right GGUF file")
    parser.add_argument("--metadata", action="store_true", help="Compare metadata values and duplicate key counts")
    parser.add_argument("--tensor-bytes", action="store_true", help="Compare raw tensor payload bytes when safe")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    parser.add_argument(
        "--max-array-values",
        type=int,
        default=None,
        help="Maximum metadata array values to keep; use -1 for full arrays",
    )
    args = parser.parse_args(argv)
    if args.max_array_values is not None and args.max_array_values < -1:
        parser.error("--max-array-values must be -1 or non-negative")
    return args


def _descriptor_value(tensor: GGUFTensorInfo, field: str) -> Any:
    value = getattr(tensor, field)
    if field == "shape":
        return list(value)
    return value


def _descriptor_summary(tensor: GGUFTensorInfo) -> dict[str, Any]:
    return {field: _descriptor_value(tensor, field) for field in _DESCRIPTOR_FIELDS}


def _byte_descriptors_match(left: GGUFTensorInfo, right: GGUFTensorInfo) -> bool:
    for field in _DESCRIPTOR_FIELDS:
        if _descriptor_value(left, field) != _descriptor_value(right, field):
            return False
    return True


def _unique_tensors_by_name(info: GGUFFile) -> dict[str, GGUFTensorInfo]:
    counts: dict[str, int] = {}
    for tensor in info.tensors:
        counts[tensor.name] = counts.get(tensor.name, 0) + 1
    return {tensor.name: tensor for tensor in info.tensors if counts[tensor.name] == 1}


def _duplicate_tensor_names(info: GGUFFile) -> set[str]:
    counts: dict[str, int] = {}
    for tensor in info.tensors:
        counts[tensor.name] = counts.get(tensor.name, 0) + 1
    return {name for name, count in counts.items() if count > 1}


def _payload_summary(data: bytes) -> dict[str, Any]:
    return {
        "nbytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _print_result(result: _ComparisonResult) -> None:
    print(f"Left: {result.left_path}")
    print(f"Right: {result.right_path}")
    modes = ["descriptors"]
    if result.metadata:
        modes.append("metadata")
    if result.tensor_bytes:
        modes.append("tensor-bytes")
    print(f"Modes: {', '.join(modes)}")

    if result.ok:
        print("No differences found.")
        return

    print(f"Differences: {len(result.differences)}")
    for kind in sorted({difference.kind for difference in result.differences}):
        print(f"{kind}:")
        for difference in result.differences:
            if difference.kind != kind:
                continue
            location = _difference_location(difference)
            suffix = f" ({location})" if location else ""
            print(f"  - {difference.message}{suffix}")
            if difference.left is not None or difference.right is not None:
                print(f"    left: {_format_detail(difference.left)}")
                print(f"    right: {_format_detail(difference.right)}")


def _difference_location(difference: _Difference) -> str:
    parts: list[str] = []
    if difference.index is not None:
        parts.append(f"index={difference.index}")
    if difference.tensor_name is not None:
        parts.append(f"tensor={difference.tensor_name}")
    if difference.metadata_key is not None:
        parts.append(f"metadata={difference.metadata_key}")
    return ", ".join(parts)


def _format_detail(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


if __name__ == "__main__":
    main()
