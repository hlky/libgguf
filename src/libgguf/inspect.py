from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import struct
from typing import Any, BinaryIO

from ._metadata import GGMLQuantizationType, GGML_QUANT_SIZES, LlamaFileType


GGUF_MAGIC = b"GGUF"
GGUF_DEFAULT_ALIGNMENT = 32
GGUF_MAX_DIMS = 4

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

GGUF_VALUE_TYPE_NAMES = {
    GGUF_TYPE_UINT8: "UINT8",
    GGUF_TYPE_INT8: "INT8",
    GGUF_TYPE_UINT16: "UINT16",
    GGUF_TYPE_INT16: "INT16",
    GGUF_TYPE_UINT32: "UINT32",
    GGUF_TYPE_INT32: "INT32",
    GGUF_TYPE_FLOAT32: "FLOAT32",
    GGUF_TYPE_BOOL: "BOOL",
    GGUF_TYPE_STRING: "STRING",
    GGUF_TYPE_ARRAY: "ARRAY",
    GGUF_TYPE_UINT64: "UINT64",
    GGUF_TYPE_INT64: "INT64",
    GGUF_TYPE_FLOAT64: "FLOAT64",
}

_SCALAR_FORMATS = {
    GGUF_TYPE_UINT8: "B",
    GGUF_TYPE_INT8: "b",
    GGUF_TYPE_UINT16: "H",
    GGUF_TYPE_INT16: "h",
    GGUF_TYPE_UINT32: "I",
    GGUF_TYPE_INT32: "i",
    GGUF_TYPE_FLOAT32: "f",
    GGUF_TYPE_BOOL: "?",
    GGUF_TYPE_UINT64: "Q",
    GGUF_TYPE_INT64: "q",
    GGUF_TYPE_FLOAT64: "d",
}


class GGUFFormatError(ValueError):
    """Raised when a file does not contain a supported GGUF header."""


@dataclass(frozen=True)
class GGUFMetadataValue:
    raw_type: int
    value_type: str
    value: Any
    raw_array_type: int | None = None
    array_type: str | None = None
    length: int | None = None
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = {
            "type": self.value_type,
            "value": self.value,
        }
        if self.value_type == "ARRAY":
            data["array_type"] = self.array_type
            data["length"] = self.length
            data["truncated"] = self.truncated
        return data


@dataclass(frozen=True)
class GGUFTensorInfo:
    name: str
    shape: tuple[int, ...]
    qtype: str
    qtype_value: int
    offset: int
    data_offset: int
    nbytes: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "qtype": self.qtype,
            "qtype_value": self.qtype_value,
            "offset": self.offset,
            "data_offset": self.data_offset,
            "nbytes": self.nbytes,
        }


@dataclass(frozen=True)
class GGUFFile:
    path: Path
    version: int
    tensor_count: int
    metadata_kv_count: int
    metadata: dict[str, GGUFMetadataValue]
    tensors: tuple[GGUFTensorInfo, ...]
    alignment: int
    data_offset: int
    file_size: int

    @property
    def tensor_type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for tensor in self.tensors:
            counts[tensor.qtype] = counts.get(tensor.qtype, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": os.fspath(self.path),
            "version": self.version,
            "tensor_count": self.tensor_count,
            "metadata_kv_count": self.metadata_kv_count,
            "alignment": self.alignment,
            "data_offset": self.data_offset,
            "file_size": self.file_size,
            "metadata": {key: value.to_dict() for key, value in self.metadata.items()},
            "tensors": [tensor.to_dict() for tensor in self.tensors],
            "tensor_type_counts": self.tensor_type_counts,
        }


@dataclass(frozen=True)
class GGUFValidationIssue:
    severity: str
    code: str
    message: str
    tensor_name: str | None = None
    metadata_key: str | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.tensor_name is not None:
            data["tensor_name"] = self.tensor_name
        if self.metadata_key is not None:
            data["metadata_key"] = self.metadata_key
        if self.details is not None:
            data["details"] = self.details
        return data


@dataclass(frozen=True)
class GGUFValidationResult:
    path: Path
    file: GGUFFile | None
    issues: tuple[GGUFValidationIssue, ...]

    @property
    def errors(self) -> tuple[GGUFValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[GGUFValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": os.fspath(self.path),
            "valid": self.ok,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [issue.to_dict() for issue in self.issues],
            "file": _validation_file_summary(self.file),
        }


def inspect_gguf(path: str | os.PathLike[str], *, max_array_values: int | None = None) -> GGUFFile:
    """Read GGUF metadata and tensor descriptors without reading tensor payloads.

    Args:
        path: GGUF file to inspect.
        max_array_values: Optional cap for array metadata values returned in
            memory. Remaining array entries are skipped. Use ``None`` to return
            full metadata arrays.
    """

    gguf_path = Path(path)
    if max_array_values is not None and max_array_values < 0:
        raise ValueError("max_array_values must be non-negative or None")

    with gguf_path.open("rb") as handle:
        if _read_exact(handle, 4) != GGUF_MAGIC:
            raise GGUFFormatError("not a GGUF file")

        version, tensor_count, metadata_kv_count = _read_struct(handle, "IQQ")
        if version not in (2, 3):
            raise GGUFFormatError(f"unsupported GGUF version {version}")

        metadata: dict[str, GGUFMetadataValue] = {}
        alignment = GGUF_DEFAULT_ALIGNMENT
        for _ in range(int(metadata_kv_count)):
            key = _read_string(handle)
            value = _read_value(handle, max_array_values=max_array_values)
            metadata[key] = value
            if key == "general.alignment" and value.value_type in {"UINT32", "UINT64"}:
                alignment = int(value.value)

        if alignment <= 0:
            raise GGUFFormatError(f"invalid GGUF alignment {alignment}")

        raw_tensors: list[tuple[str, tuple[int, ...], int, int]] = []
        for _ in range(int(tensor_count)):
            name = _read_string(handle)
            (n_dims,) = _read_struct(handle, "I")
            if n_dims > GGUF_MAX_DIMS:
                raise GGUFFormatError(f"unsupported GGUF tensor dimension count {n_dims}")
            shape = _read_struct(handle, "Q" * int(n_dims))
            qtype_value, offset = _read_struct(handle, "IQ")
            raw_tensors.append(
                (
                    name,
                    tuple(int(dim) for dim in shape),
                    int(qtype_value),
                    int(offset),
                )
            )

        header_end = handle.tell()

    data_offset = _align_offset(header_end, alignment)
    tensors = tuple(
        GGUFTensorInfo(
            name=name,
            shape=shape,
            qtype=_qtype_name(qtype_value),
            qtype_value=qtype_value,
            offset=offset,
            data_offset=data_offset + offset,
            nbytes=_tensor_nbytes(shape, qtype_value),
        )
        for name, shape, qtype_value, offset in raw_tensors
    )

    return GGUFFile(
        path=gguf_path,
        version=int(version),
        tensor_count=int(tensor_count),
        metadata_kv_count=int(metadata_kv_count),
        metadata=metadata,
        tensors=tensors,
        alignment=alignment,
        data_offset=data_offset,
        file_size=gguf_path.stat().st_size,
    )


read_gguf_header = inspect_gguf


def validate_gguf(path: str | os.PathLike[str], *, max_array_values: int | None = 0) -> GGUFValidationResult:
    """Validate GGUF structure without reading tensor payload bytes.

    The validator uses :func:`inspect_gguf` for format parsing, then checks
    descriptor-level invariants such as common metadata presence, tensor qtypes,
    payload ranges, ordering, overlap, and duplicate tensor names.
    """

    gguf_path = Path(path)
    try:
        info = inspect_gguf(gguf_path, max_array_values=max_array_values)
    except GGUFFormatError as exc:
        return GGUFValidationResult(
            path=gguf_path,
            file=None,
            issues=(
                GGUFValidationIssue(
                    severity="error",
                    code=_format_error_code(str(exc)),
                    message=str(exc),
                ),
            ),
        )
    except OSError as exc:
        return GGUFValidationResult(
            path=gguf_path,
            file=None,
            issues=(
                GGUFValidationIssue(
                    severity="error",
                    code="io",
                    message=str(exc),
                ),
            ),
        )

    issues: list[GGUFValidationIssue] = []

    if info.alignment <= 0:
        issues.append(
            GGUFValidationIssue(
                severity="error",
                code="alignment_invalid",
                message=f"GGUF alignment must be positive, got {info.alignment}",
                details={"alignment": info.alignment},
            )
        )

    for key in ("general.architecture", "general.quantization_version"):
        if key not in info.metadata:
            issues.append(
                GGUFValidationIssue(
                    severity="warning",
                    code="metadata_missing",
                    message=f"common metadata key {key!r} is missing",
                    metadata_key=key,
                )
            )

    seen_names: dict[str, int] = {}
    for tensor in info.tensors:
        seen_names[tensor.name] = seen_names.get(tensor.name, 0) + 1
    for name, count in sorted(seen_names.items()):
        if count > 1:
            issues.append(
                GGUFValidationIssue(
                    severity="error",
                    code="tensor_duplicate_name",
                    message=f"tensor name {name!r} appears {count} times",
                    tensor_name=name,
                    details={"count": count},
                )
            )

    known_ranges: list[tuple[int, int, GGUFTensorInfo]] = []
    previous_known_offset: int | None = None
    for tensor in info.tensors:
        if any(dim == 0 for dim in tensor.shape):
            issues.append(
                GGUFValidationIssue(
                    severity="error",
                    code="tensor_shape_invalid",
                    message=f"tensor {tensor.name!r} shape {list(tensor.shape)} contains a zero dimension",
                    tensor_name=tensor.name,
                    details={"shape": list(tensor.shape)},
                )
            )

        if tensor.offset < 0:
            issues.append(
                GGUFValidationIssue(
                    severity="error",
                    code="tensor_offset_negative",
                    message=f"tensor {tensor.name!r} has negative relative data offset {tensor.offset}",
                    tensor_name=tensor.name,
                    details={"offset": tensor.offset},
                )
            )

        if tensor.offset % info.alignment != 0:
            issues.append(
                GGUFValidationIssue(
                    severity="error",
                    code="tensor_offset_alignment",
                    message=(
                        f"tensor {tensor.name!r} relative data offset {tensor.offset} is not "
                        f"aligned to GGUF alignment {info.alignment}"
                    ),
                    tensor_name=tensor.name,
                    details={"offset": tensor.offset, "alignment": info.alignment},
                )
            )

        qtype_spec = _qtype_spec(tensor.qtype_value)
        if qtype_spec is None:
            issues.append(
                GGUFValidationIssue(
                    severity="warning",
                    code="qtype_unknown",
                    message=f"tensor {tensor.name!r} uses unknown qtype value {tensor.qtype_value}",
                    tensor_name=tensor.name,
                    details={"qtype_value": tensor.qtype_value},
                )
            )
            continue

        block_size, _type_size = qtype_spec
        n_per_row = tensor.shape[0] if tensor.shape else 1
        if tensor.nbytes is None:
            issues.append(
                GGUFValidationIssue(
                    severity="warning",
                    code="qtype_row_width",
                    message=(
                        f"tensor {tensor.name!r} row width {n_per_row} is not valid for "
                        f"{tensor.qtype} block size {block_size}"
                    ),
                    tensor_name=tensor.name,
                    details={
                        "qtype": tensor.qtype,
                        "qtype_value": tensor.qtype_value,
                        "row_width": n_per_row,
                        "block_size": block_size,
                    },
                )
            )
            continue

        start = tensor.data_offset
        end = start + tensor.nbytes
        known_ranges.append((start, end, tensor))
        if end > info.file_size:
            issues.append(
                GGUFValidationIssue(
                    severity="error",
                    code="tensor_payload_range",
                    message=(
                        f"tensor {tensor.name!r} payload range [{start}, {end}) exceeds "
                        f"file size {info.file_size}"
                    ),
                    tensor_name=tensor.name,
                    details={
                        "data_offset": start,
                        "nbytes": tensor.nbytes,
                        "end": end,
                        "file_size": info.file_size,
                    },
                )
            )
        if previous_known_offset is not None and start < previous_known_offset:
            issues.append(
                GGUFValidationIssue(
                    severity="warning",
                    code="tensor_offset_order",
                    message=(
                        f"tensor {tensor.name!r} data offset {start} is before the previous "
                        f"known tensor offset {previous_known_offset}"
                    ),
                    tensor_name=tensor.name,
                    details={"data_offset": start, "previous_data_offset": previous_known_offset},
                )
            )
        previous_known_offset = start

    previous_range: tuple[int, int, GGUFTensorInfo] | None = None
    for start, end, tensor in sorted(known_ranges, key=lambda item: (item[0], item[1], item[2].name)):
        if previous_range is not None:
            previous_start, previous_end, previous_tensor = previous_range
            if start < previous_end:
                issues.append(
                    GGUFValidationIssue(
                        severity="error",
                        code="tensor_payload_overlap",
                        message=(
                            f"tensor {tensor.name!r} payload range [{start}, {end}) overlaps "
                            f"{previous_tensor.name!r} range [{previous_start}, {previous_end})"
                        ),
                        tensor_name=tensor.name,
                        details={
                            "data_offset": start,
                            "end": end,
                            "overlap_tensor_name": previous_tensor.name,
                            "overlap_data_offset": previous_start,
                            "overlap_end": previous_end,
                        },
                    )
                )
            if end > previous_end:
                previous_range = (start, end, tensor)
        else:
            previous_range = (start, end, tensor)

    return GGUFValidationResult(path=gguf_path, file=info, issues=tuple(issues))


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    max_array_values = None if args.max_array_values < 0 else args.max_array_values
    info = inspect_gguf(args.path, max_array_values=max_array_values)

    if args.json:
        print(json.dumps(info.to_dict(), indent=2, sort_keys=True))
        return

    _print_summary(info)
    if args.metadata:
        _print_metadata(info, limit=args.limit)
    if args.tensors:
        _print_tensors(info, limit=args.limit)


def validate_main(argv: list[str] | None = None) -> None:
    args = _parse_validate_args(argv)
    max_array_values = None if args.max_array_values < 0 else args.max_array_values
    result = validate_gguf(args.path, max_array_values=max_array_values)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        _print_validation_result(result)

    if not result.ok:
        raise SystemExit(1)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect GGUF metadata and tensor descriptors")
    parser.add_argument("path", help="GGUF file to inspect")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    parser.add_argument("--metadata", action="store_true", help="Print metadata entries")
    parser.add_argument("--tensors", action="store_true", help="Print tensor descriptors")
    parser.add_argument("--limit", type=int, default=20, help="Maximum metadata/tensor rows to print in text mode")
    parser.add_argument(
        "--max-array-values",
        type=int,
        default=20,
        help="Maximum metadata array values to keep; use -1 for full arrays",
    )
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.max_array_values < -1:
        parser.error("--max-array-values must be -1 or non-negative")
    return args


def _parse_validate_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate GGUF structure without reading tensor payload bytes")
    parser.add_argument("path", help="GGUF file to validate")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    parser.add_argument(
        "--max-array-values",
        type=int,
        default=0,
        help="Maximum metadata array values to keep; use -1 for full arrays",
    )
    args = parser.parse_args(argv)
    if args.max_array_values < -1:
        parser.error("--max-array-values must be -1 or non-negative")
    return args


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise GGUFFormatError("unexpected end of GGUF header")
    return data


def _read_struct(handle: BinaryIO, fmt: str) -> tuple[Any, ...]:
    return struct.unpack("<" + fmt, _read_exact(handle, struct.calcsize("<" + fmt)))


def _read_string(handle: BinaryIO) -> str:
    (size,) = _read_struct(handle, "Q")
    return _read_exact(handle, int(size)).decode("utf-8")


def _read_scalar(handle: BinaryIO, value_type: int) -> Any:
    fmt = _SCALAR_FORMATS.get(value_type)
    if fmt is not None:
        (value,) = _read_struct(handle, fmt)
        return value
    if value_type == GGUF_TYPE_STRING:
        return _read_string(handle)
    raise GGUFFormatError(f"unsupported GGUF metadata value type {value_type}")


def _read_value(handle: BinaryIO, *, max_array_values: int | None) -> GGUFMetadataValue:
    (value_type,) = _read_struct(handle, "I")
    value_type = int(value_type)
    if value_type != GGUF_TYPE_ARRAY:
        return GGUFMetadataValue(
            raw_type=value_type,
            value_type=_value_type_name(value_type),
            value=_read_scalar(handle, value_type),
        )

    array_type, array_len = _read_struct(handle, "IQ")
    array_type = int(array_type)
    array_len = int(array_len)
    if array_type == GGUF_TYPE_ARRAY:
        raise GGUFFormatError("nested GGUF metadata arrays are not supported")

    keep = array_len if max_array_values is None else min(array_len, max_array_values)
    values = [_read_scalar(handle, array_type) for _ in range(keep)]
    if keep < array_len:
        _skip_array_values(handle, array_type, array_len - keep)

    return GGUFMetadataValue(
        raw_type=value_type,
        value_type="ARRAY",
        value=values,
        raw_array_type=array_type,
        array_type=_value_type_name(array_type),
        length=array_len,
        truncated=keep < array_len,
    )


def _skip_array_values(handle: BinaryIO, array_type: int, count: int) -> None:
    if count <= 0:
        return
    if array_type == GGUF_TYPE_STRING:
        for _ in range(count):
            _read_string(handle)
        return
    fmt = _SCALAR_FORMATS.get(array_type)
    if fmt is None:
        raise GGUFFormatError(f"unsupported GGUF array type {array_type}")
    handle.seek(struct.calcsize("<" + fmt) * count, os.SEEK_CUR)


def _value_type_name(value_type: int) -> str:
    return GGUF_VALUE_TYPE_NAMES.get(value_type, str(value_type))


def _qtype_name(qtype_value: int) -> str:
    try:
        return GGMLQuantizationType(qtype_value).name
    except ValueError:
        return str(qtype_value)


def _tensor_nbytes(shape: tuple[int, ...], qtype_value: int) -> int | None:
    qtype_spec = _qtype_spec(qtype_value)
    if qtype_spec is None:
        return None

    block_size, type_size = qtype_spec
    n_per_row = shape[0] if shape else 1
    if n_per_row % block_size != 0:
        return None
    n_rows = math.prod(shape[1:]) if len(shape) > 1 else 1
    return n_rows * (n_per_row // block_size * type_size)


def _qtype_spec(qtype_value: int) -> tuple[int, int] | None:
    try:
        qtype = GGMLQuantizationType(qtype_value)
    except ValueError:
        return None
    return GGML_QUANT_SIZES.get(qtype)


def _align_offset(offset: int, alignment: int) -> int:
    padding = offset % alignment
    if padding:
        return offset + alignment - padding
    return offset


def _metadata_scalar(info: GGUFFile, key: str) -> Any:
    value = info.metadata.get(key)
    if value is None or value.value_type == "ARRAY":
        return None
    return value.value


def _print_summary(info: GGUFFile) -> None:
    print(f"Path: {info.path}")
    print(f"Version: {info.version}")
    print(f"Tensors: {info.tensor_count}")
    print(f"Metadata: {info.metadata_kv_count}")
    print(f"Alignment: {info.alignment}")
    print(f"Data offset: {info.data_offset}")

    arch = _metadata_scalar(info, "general.architecture")
    if arch is not None:
        print(f"Architecture: {arch}")

    file_type = _metadata_scalar(info, "general.file_type")
    if isinstance(file_type, int):
        try:
            file_type = LlamaFileType(file_type).name
        except ValueError:
            pass
        print(f"File type: {file_type}")

    counts = ", ".join(f"{name}={count}" for name, count in sorted(info.tensor_type_counts.items()))
    if counts:
        print(f"Tensor types: {counts}")


def _print_metadata(info: GGUFFile, *, limit: int) -> None:
    print("Metadata entries:")
    items = list(info.metadata.items())
    for key, value in items[:limit]:
        print(f"  {key}: {_format_metadata_value(value)}")
    if len(items) > limit:
        print(f"  ... {len(items) - limit} more")


def _print_tensors(info: GGUFFile, *, limit: int) -> None:
    print("Tensors:")
    for tensor in info.tensors[:limit]:
        nbytes = "unknown" if tensor.nbytes is None else str(tensor.nbytes)
        print(
            f"  {tensor.name}: shape={list(tensor.shape)} "
            f"type={tensor.qtype} offset={tensor.offset} data_offset={tensor.data_offset} nbytes={nbytes}"
        )
    if len(info.tensors) > limit:
        print(f"  ... {len(info.tensors) - limit} more")


def _format_metadata_value(value: GGUFMetadataValue) -> str:
    if value.value_type != "ARRAY":
        return repr(value.value)
    suffix = " ..." if value.truncated else ""
    return f"{value.array_type}[{value.length}] {value.value!r}{suffix}"


def _format_error_code(message: str) -> str:
    if message == "not a GGUF file":
        return "magic"
    if message.startswith("unsupported GGUF version"):
        return "version"
    if message.startswith("invalid GGUF alignment"):
        return "alignment_invalid"
    return "format"


def _validation_file_summary(info: GGUFFile | None) -> dict[str, Any] | None:
    if info is None:
        return None
    return {
        "version": info.version,
        "tensor_count": info.tensor_count,
        "metadata_kv_count": info.metadata_kv_count,
        "alignment": info.alignment,
        "data_offset": info.data_offset,
        "file_size": info.file_size,
        "tensor_type_counts": info.tensor_type_counts,
    }


def _print_validation_result(result: GGUFValidationResult) -> None:
    status = "VALID" if result.ok else "INVALID"
    print(f"{status}: {result.path}")
    if result.file is not None:
        print(
            f"Version: {result.file.version}  Tensors: {result.file.tensor_count}  "
            f"Metadata: {result.file.metadata_kv_count}"
        )

    if not result.issues:
        print("No validation issues.")
        return

    errors = result.errors
    warnings = result.warnings
    if errors:
        print("Errors:")
        for issue in errors:
            print(f"  [{issue.code}] {_format_validation_issue(issue)}")
    if warnings:
        print("Warnings:")
        for issue in warnings:
            print(f"  [{issue.code}] {_format_validation_issue(issue)}")


def _format_validation_issue(issue: GGUFValidationIssue) -> str:
    prefix = ""
    if issue.tensor_name is not None:
        prefix = f"{issue.tensor_name}: "
    elif issue.metadata_key is not None:
        prefix = f"{issue.metadata_key}: "
    return prefix + issue.message


__all__ = [
    "GGUFFile",
    "GGUFFormatError",
    "GGUFMetadataValue",
    "GGUFTensorInfo",
    "GGUFValidationIssue",
    "GGUFValidationResult",
    "inspect_gguf",
    "read_gguf_header",
    "validate_gguf",
    "validate_main",
]


if __name__ == "__main__":
    main()
