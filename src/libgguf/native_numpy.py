"""Independent NumPy/Python reference for the native clip-grid encoder."""

from __future__ import annotations

import math
import struct

import numpy as np

from .native_formats import NATIVE_QUANT_SIZES, NativeQuantizationType, native_row_size


def _half(value: float) -> tuple[bytes, float]:
    bits = struct.pack("<e", min(65504.0, max(2**-24, value))) if value else b"\0\0"
    return bits, struct.unpack("<e", bits)[0]


def _score(x, importance, scale, q) -> float:
    total = 0.0
    weighted = importance is not None and any(w > 0 for w in importance)
    for k, value in enumerate(x):
        decoded = float(np.float32(np.float32(scale) * np.float32(q[k])))
        error = float(value) - decoded
        square = error * error
        term = square * (float(importance[k]) if weighted else 1.0)
        total = total + term
    return total


def _fit(x, importance, lo, hi):
    initial = max(float(v) / (hi if v >= 0 else lo) for v in x)
    if not initial:
        return b"\0\0", 0.0, [0] * len(x)
    best = None
    for j in range(17):
        bits, scale = _half(initial * (16 + j) / 32.0)
        q = [max(lo, min(hi, round(float(v) / scale))) for v in x]
        error = _score(x, importance, scale, q)
        if best is None or error < best[0]:
            best = error, bits, scale, q
    return best[1:]


def _payload(q, eight=False):
    if eight:
        return bytes(v & 255 for v in q)
    return bytes((q[k] & 15) | ((q[k + 1] & 15) << 4) for k in range(0, len(q), 2))


def _encode_block(x, format, importance):
    if format != NativeQuantizationType.S4H256:
        eight = format == NativeQuantizationType.S8G32
        bits, _, q = _fit(x, importance, -127 if eight else -8, 127 if eight else 7)
        return bits + _payload(q, eight)
    local = []
    for start in range(0, 256, 32):
        _, scale, _ = _fit(x[start:start + 32],
            None if importance is None else importance[start:start + 32], -8, 7)
        local.append(scale)
    maximum = max(local)
    if not maximum:
        return bytes(138)
    best = None
    weighted = importance is not None and any(w > 0 for w in importance)
    for j in range(17):
        bits, base = _half((maximum / 255.0) * (16 + j) / 32.0)
        codes = [max(1, min(255, round(s / base))) if s else 0 for s in local]
        q, error = [], 0.0
        for k, value in enumerate(x):
            scale = float(np.float32(base) * np.float32(codes[k // 32]))
            quant = max(-8, min(7, round(float(value) / scale))) if scale else 0
            q.append(quant)
            delta = float(value) - float(np.float32(scale) * np.float32(quant))
            square = delta * delta
            term = square * (float(importance[k]) if weighted else 1.0)
            error = error + term
        if best is None or error < best[0]:
            best = error, bits + bytes(codes) + _payload(q)
    return best[1]


def validate_rows(rows, format, importance=None):
    format = NativeQuantizationType(format)
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[0] <= 0:
        raise ValueError("native quantization requires a nonempty rank-two matrix")
    native_row_size(format, rows.shape[1])
    if not np.isfinite(rows).all():
        raise ValueError("native encoder requires finite input")
    if importance is not None:
        importance = np.ascontiguousarray(importance, dtype=np.float32)
        if importance.shape != (rows.shape[1],) or not np.isfinite(importance).all() or (importance < 0).any():
            raise ValueError("importance must be a finite nonnegative vector along K")
    return rows, format, importance


def quantize_rows(rows, format, importance=None):
    rows, format, importance = validate_rows(rows, format, importance)
    block_values, _, _ = NATIVE_QUANT_SIZES[format]
    output = bytearray()
    for row in rows:
        for start in range(0, row.size, block_values):
            output.extend(_encode_block(row[start:start + block_values], format,
                None if importance is None else importance[start:start + block_values]))
    return np.frombuffer(output, dtype=np.uint8).reshape(rows.shape[0], -1).copy()


def dequantize_rows(encoded, format, columns):
    format = NativeQuantizationType(format)
    values, block_bytes, _ = NATIVE_QUANT_SIZES[format]
    row_bytes = native_row_size(format, columns)
    data = np.ascontiguousarray(encoded, dtype=np.uint8)
    if data.ndim != 2 or data.shape[0] <= 0 or data.shape[1] != row_bytes:
        raise ValueError("encoded native row shape differs from its format")
    output = np.empty((data.shape[0], columns), dtype=np.float32)
    for n, row in enumerate(data):
        for start in range(0, columns, values):
            offset = start // values * block_bytes
            block = bytes(row[offset:offset + block_bytes])
            bits = int.from_bytes(block[:2], "little")
            base = struct.unpack("<e", block[:2])[0]
            if bits & 0x8000 or not math.isfinite(base):
                raise ValueError("invalid native scale")
            if not base and format == NativeQuantizationType.S4H256 and any(block[2:10]):
                raise ValueError("zero base must have zero local scale codes")
            for k in range(values):
                if format == NativeQuantizationType.S8G32:
                    q = block[2 + k]
                    q = q if q < 128 else q - 256
                    if q == -128:
                        raise ValueError("-128 is not a canonical S8G32 value")
                else:
                    header = 10 if format == NativeQuantizationType.S4H256 else 2
                    q = (block[header + k // 2] >> ((k & 1) * 4)) & 15
                    q = q if q < 8 else q - 16
                scale = np.float32(base)
                if format == NativeQuantizationType.S4H256:
                    scale = scale * np.float32(block[2 + k // 32])
                if not scale and q:
                    raise ValueError("zero-scale group must have zero quantized values")
                output[n, start + k] = scale * np.float32(q)
    return output
