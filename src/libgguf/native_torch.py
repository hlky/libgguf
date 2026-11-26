"""Independent Torch reference for native encodings, including fake tensor shapes."""

import torch

from .native_formats import NATIVE_QUANT_SIZES, NativeQuantizationType, native_row_size


def _score(values, scales, quants, importance):
    errors = values.to(torch.float64) - (scales.float() * quants.float()).to(torch.float64)
    score = torch.zeros(values.shape[0], device=values.device, dtype=torch.float64)
    for k in range(values.shape[1]):
        score = score + errors[:, k].square() * importance[:, k].to(torch.float64)
    return score


def _half(value):
    return torch.where(value == 0, value, value.clamp(2**-24, 65504)).to(torch.float16).to(torch.float64)


def _fit(values, importance, lo, hi):
    values = values.to(torch.float64)
    initial = torch.where(values >= 0, values / hi, values / lo).amax(dim=1, keepdim=True)
    best_score = torch.full((values.shape[0],), float("inf"), dtype=torch.float64, device=values.device)
    best_scale = torch.zeros_like(initial)
    best_q = torch.zeros_like(values, dtype=torch.int16)
    for j in range(17):
        scale = _half(initial * (16 + j) / 32)
        q = torch.where(scale == 0, 0, (values / torch.where(scale == 0, 1, scale)).round().clamp(lo, hi)).to(torch.int16)
        score = _score(values, scale, q, importance)
        update = score < best_score
        best_score = torch.where(update, score, best_score)
        best_scale = torch.where(update[:, None], scale, best_scale)
        best_q = torch.where(update[:, None], q, best_q)
    return best_scale, best_q


def quantize_rows(rows, format, importance=None):
    """Encode finite F32 rows on their Torch device with ordered FP64 scoring."""
    format = NativeQuantizationType(format)
    values, block_bytes, _ = NATIVE_QUANT_SIZES[format]
    if rows.dtype != torch.float32 or rows.ndim != 2 or rows.shape[0] <= 0:
        raise ValueError("native Torch encoder requires nonempty rank-two F32 input")
    row_bytes = native_row_size(format, rows.shape[1])
    if importance is not None and (importance.dtype != torch.float32 or importance.device != rows.device or importance.shape != (rows.shape[1],)):
        raise ValueError("importance must be an F32 K vector on the input device")
    x = rows.reshape(-1, values)
    w = torch.ones_like(x) if importance is None else importance.repeat(rows.shape[0]).reshape_as(x)
    # Shape-only fake execution never reads tensor values.
    from torch._subclasses.fake_tensor import is_fake
    if not is_fake(rows) and rows.device.type != "meta":
        if not bool(torch.isfinite(rows).all()) or (importance is not None and (not bool(torch.isfinite(importance).all()) or bool((importance < 0).any()))):
            raise ValueError("native encoder requires finite values and nonnegative importance")
    eight = format == NativeQuantizationType.S8G32
    if format != NativeQuantizationType.S4H256:
        w = torch.where(w.amax(dim=1, keepdim=True) > 0, w, 1)
        scale, q = _fit(x, w, -127 if eight else -8, 127 if eight else 7)
        header = scale.to(torch.float16).view(torch.uint8).reshape(-1, 2)
    else:
        local_w = w.reshape(-1, 32)
        local_w = torch.where(local_w.amax(dim=1, keepdim=True) > 0, local_w, 1)
        local, _ = _fit(x.reshape(-1, 32), local_w, -8, 7)
        local = local.reshape(-1, 8)
        maximum = local.amax(dim=1, keepdim=True)
        w = torch.where(w.amax(dim=1, keepdim=True) > 0, w, 1)
        best = torch.full((x.shape[0],), float("inf"), dtype=torch.float64, device=x.device)
        q = torch.zeros_like(x, dtype=torch.int16)
        header = torch.zeros((x.shape[0], 10), dtype=torch.uint8, device=x.device)
        for j in range(17):
            base = _half((maximum / 255) * (16 + j) / 32)
            codes = torch.where(local == 0, 0, (local / torch.where(base == 0, 1, base)).round().clamp(1, 255)).to(torch.uint8)
            effective = (base.float() * codes.float()).repeat_interleave(32, dim=1)
            candidate = torch.where(effective == 0, 0, (x.double() / torch.where(effective == 0, 1, effective).double()).round().clamp(-8, 7)).to(torch.int16)
            score = _score(x, effective, candidate, w)
            update = score < best
            best = torch.where(update, score, best)
            q = torch.where(update[:, None], candidate, q)
            candidate_header = torch.cat((base.to(torch.float16).view(torch.uint8).reshape(-1, 2), codes), dim=1)
            header = torch.where(update[:, None], candidate_header, header)
    payload = q.to(torch.uint8) if eight else ((q[:, ::2] & 15) | ((q[:, 1::2] & 15) << 4)).to(torch.uint8)
    return torch.cat((header, payload), dim=1).reshape(rows.shape[0], row_bytes)


def dequantize_rows(encoded, format, columns):
    """Decode canonical U8 rows to F32 on the same device, validating canonical values."""
    format = NativeQuantizationType(format)
    values, block_bytes, _ = NATIVE_QUANT_SIZES[format]
    if encoded.dtype != torch.uint8 or encoded.ndim != 2 or encoded.shape[0] <= 0 or encoded.shape[1] != native_row_size(format, columns):
        raise ValueError("native Torch encoded shape differs from its format")
    blocks = encoded.reshape(-1, block_bytes)
    base = blocks[:, :2].contiguous().view(torch.float16).float()
    hierarchical = format == NativeQuantizationType.S4H256
    start = 10 if hierarchical else 2
    payload = blocks[:, start:].to(torch.int16)
    if format == NativeQuantizationType.S8G32:
        q = torch.where(payload < 128, payload, payload - 256)
    else:
        q = torch.stack((payload & 15, payload >> 4), dim=-1).reshape(-1, values)
        q = torch.where(q < 8, q, q - 16)
    scales = (base * blocks[:, 2:10].float()).repeat_interleave(32, dim=1) if hierarchical else base.expand(-1, values)
    from torch._subclasses.fake_tensor import is_fake
    if not is_fake(encoded) and encoded.device.type != "meta":
        invalid = ((blocks[:, 1] & 128) != 0).any() | (~torch.isfinite(base)).any() | ((scales == 0) & (q != 0)).any()
        if hierarchical:
            invalid |= ((base == 0) & (blocks[:, 2:10] != 0)).any()
        if format == NativeQuantizationType.S8G32:
            invalid |= (q == -128).any()
        if bool(invalid):
            raise ValueError("invalid canonical native encoded values")
    return (scales * q.float()).reshape(encoded.shape[0], columns)
