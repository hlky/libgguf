# Copyright 2026 hlky
# Copyright 2025 City96
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions derived from ComfyUI-GGUF:
# https://github.com/city96/ComfyUI-GGUF/blob/795e45156ece99afbc3efef911e63fcb46e6a20d/dequant.py
#
# Modifications:
# - Extended torch-native dequantization.
# - Implemented torch-native quantization.
# - Added byte-parity fixes and validation.

from libgguf._metadata import GGMLQuantizationType, GGML_QUANT_SIZES, quant_shape_to_byte_shape
from libgguf.libgguf_numpy.libgguf_numpy import (
    GROUP_MAX_EPS,
    IQ2_XXS,
    dequantize as dequantize_np,
    _get_iq3_xxs_lookup,
    _get_iq3_s_lookup,
    _get_iq2_xxs_lookup,
    _get_iq2_xs_lookup,
    _get_iq1_s_lookup,
    GROUP_MAX_EPS_IQ3_XXS,
    GROUP_MAX_EPS_IQ2_S,
    GROUP_MAX_EPS_IQ1_M,
    GROUP_MAX_EPS_IQ1_S,
    IQ3_S,
    IQ3_XXS,
    _get_iq2_s_lookup,
    IQ1_S,
    IQ2_XS,
    IQ2_S,
    IQ1_M,
)
import torch

TORCH_COMPATIBLE_QTYPES = (None, GGMLQuantizationType.F32, GGMLQuantizationType.F16)


def is_torch_compatible(tensor):
    return (
        tensor is None
        or getattr(tensor, "tensor_type", None) in TORCH_COMPATIBLE_QTYPES
    )


def is_quantized(tensor):
    return not is_torch_compatible(tensor)


def dequantize_tensor(tensor, dtype=None, dequant_dtype=None):
    qtype = getattr(tensor, "tensor_type", None)
    oshape = getattr(tensor, "tensor_shape", tensor.shape)

    if qtype in TORCH_COMPATIBLE_QTYPES:
        return tensor.to(dtype)
    elif qtype in dequantize_functions:
        dequant_dtype = dtype if dequant_dtype == "target" else dequant_dtype
        return dequantize(tensor.data, qtype, oshape, dtype=dequant_dtype).to(dtype)
    else:
        new = dequantize_np(tensor.cpu().numpy(), qtype)
        return torch.from_numpy(new).to(tensor.device, dtype=dtype)


def dequantize(data, qtype, oshape, dtype=None):
    """
    Dequantize tensor back to usable shape/dtype
    """
    block_size, type_size = GGML_QUANT_SIZES[qtype]
    dequantize_blocks = dequantize_functions[qtype]

    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)

    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    blocks = dequantize_blocks(blocks, block_size, type_size, dtype)
    return blocks.reshape(oshape)


def quantize(data, qtype):
    """
    Quantize tensor to GGUF block bytes.
    """
    if qtype == GGMLQuantizationType.F32:
        return data.to(torch.float32)
    if qtype == GGMLQuantizationType.F16:
        return data.to(torch.float16)

    block_size, type_size = GGML_QUANT_SIZES[qtype]
    if data.shape[-1] % block_size != 0:
        raise ValueError(
            f"Can't quantize tensor with shape {tuple(data.shape)} to {qtype.name}"
        )

    quantize_blocks = quantize_functions.get(qtype)
    if quantize_blocks is None:
        raise NotImplementedError(
            f"Quantization for {qtype.name} is not yet implemented"
        )
    rows = data.reshape((-1, data.shape[-1])).to(torch.float32)
    n_blocks = rows.numel() // block_size
    blocks = rows.reshape((n_blocks, block_size))
    blocks = quantize_blocks(blocks, block_size, type_size)
    return blocks.reshape(quant_shape_to_byte_shape(tuple(data.shape), qtype))


def to_uint32(x):
    # no uint32 :(
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)


def to_uint32_int64(x):
    x = x.view(torch.uint8).to(torch.int64)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)


def to_uint16(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8).unsqueeze(1)


def split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


def _fp16_to_bytes(x):
    return x.to(torch.float16).contiguous().view(torch.uint8)


def _round_away_from_zero(x):
    ax = torch.abs(x)
    floored = torch.floor(ax)
    return torch.sign(x) * (floored + torch.floor(2 * (ax - floored)))


def _pack_bits_little(bits):
    bits = bits.to(torch.uint8).reshape(bits.shape[0], -1, 8)
    shifts = torch.arange(8, device=bits.device, dtype=torch.uint8).reshape(1, 1, 8)
    return torch.sum(bits << shifts, dim=-1).to(torch.uint8)


def _nearest_int(values):
    fvals = values.to(torch.float32)
    biased = fvals + 12_582_912.0
    ints = biased.contiguous().view(torch.int32)
    return (ints & 0x007FFFFF) - 0x00400000


def _make_qx_quants(x, nmax, quant_weights=None):
    x = x.to(torch.float32)
    ax = torch.abs(x)
    imax = int(torch.argmax(ax).item())
    amax = ax[imax]
    max_v = x[imax]
    if bool(amax < GROUP_MAX_EPS):
        return torch.tensor(
            0.0, device=x.device, dtype=torch.float32
        ), torch.zeros_like(x, dtype=torch.int8)

    iscale = -float(nmax) / max_v
    l_vals = torch.clamp(_nearest_int(iscale * x), -nmax, nmax - 1).to(torch.int8)
    w = x * x if quant_weights is None else quant_weights.to(torch.float32)
    sumlx = torch.tensor(0.0, device=x.device, dtype=torch.float32)
    suml2 = torch.tensor(0.0, device=x.device, dtype=torch.float32)
    for i in range(x.numel()):
        lf = l_vals[i].to(torch.float32)
        sumlx = sumlx + w[i] * x[i] * lf
        suml2 = suml2 + w[i] * lf * lf
    scale = (
        sumlx / suml2
        if bool(suml2 != 0)
        else torch.tensor(0.0, device=x.device, dtype=torch.float32)
    )
    best = scale * sumlx
    best_l = l_vals.clone()

    for is_ in range(-9, 10):
        if is_ == 0:
            continue
        iscale = -(float(nmax) + 0.1 * float(is_)) / max_v
        l_try = torch.clamp(_nearest_int(iscale * x), -nmax, nmax - 1).to(torch.int8)
        sumlx = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        suml2 = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        for i in range(x.numel()):
            lf = l_try[i].to(torch.float32)
            sumlx = sumlx + w[i] * x[i] * lf
            suml2 = suml2 + w[i] * lf * lf
        if bool((suml2 > 0) & (sumlx * sumlx > best * suml2)):
            best_l = l_try
            scale = sumlx / suml2
            best = scale * sumlx

    return scale, (best_l + nmax).to(torch.int8)


def _make_q3_quants(x, nmax):
    x = x.to(torch.float32)
    ax = torch.abs(x)
    imax = int(torch.argmax(ax).item())
    amax = ax[imax]
    max_v = x[imax]
    if bool(amax < GROUP_MAX_EPS):
        return torch.tensor(
            0.0, device=x.device, dtype=torch.float32
        ), torch.zeros_like(x, dtype=torch.int8)

    iscale = -float(nmax) / max_v
    l_vals = torch.clamp(_nearest_int(iscale * x), -nmax, nmax - 1).to(torch.int8)
    w = x * x
    sumlx = torch.tensor(0.0, device=x.device, dtype=torch.float32)
    suml2 = torch.tensor(0.0, device=x.device, dtype=torch.float32)
    for i in range(x.numel()):
        lf = l_vals[i].to(torch.float32)
        sumlx = sumlx + w[i] * x[i] * lf
        suml2 = suml2 + w[i] * lf * lf

    for _ in range(5):
        n_changed = 0
        for i in range(x.numel()):
            lf = l_vals[i].to(torch.float32)
            wi = w[i]
            slx = sumlx - wi * x[i] * lf
            if bool(slx > 0):
                sl2 = suml2 - wi * lf * lf
                new_l = int(_nearest_int((x[i] * sl2 / slx).reshape(1))[0].item())
                new_l = max(-nmax, min(nmax - 1, new_l))
                if new_l != int(l_vals[i].item()):
                    new_l_f = torch.tensor(
                        float(new_l), device=x.device, dtype=torch.float32
                    )
                    slx = slx + wi * x[i] * new_l_f
                    sl2 = sl2 + wi * new_l_f * new_l_f
                    if bool((sl2 > 0) & (slx * slx * suml2 > sumlx * sumlx * sl2)):
                        l_vals[i] = new_l
                        sumlx = slx
                        suml2 = sl2
                        n_changed += 1
        if n_changed == 0:
            break

    scale = (
        sumlx / suml2
        if bool(suml2 > 0)
        else torch.tensor(0.0, device=x.device, dtype=torch.float32)
    )
    return scale, (l_vals + nmax).to(torch.int8)


def _make_qkx2_quants(x, weights, nmax, rmin, rdelta, nstep, use_mad=False):
    x = x.to(torch.float32)
    weights = weights.to(torch.float32)

    min_v = x[0].clone()
    max_v = x[0].clone()
    sum_w = weights[0].clone()
    sum_x = sum_w * x[0]
    for i in range(1, x.numel()):
        if bool(x[i] < min_v):
            min_v = x[i].clone()
        if bool(x[i] > max_v):
            max_v = x[i].clone()
        w = weights[i]
        sum_w = sum_w + w
        sum_x = sum_x + w * x[i]

    if bool(min_v > 0):
        min_v = torch.tensor(0.0, device=x.device, dtype=torch.float32)
    if bool(max_v == min_v):
        return (
            torch.tensor(0.0, device=x.device, dtype=torch.float32),
            -min_v,
            torch.zeros_like(x, dtype=torch.uint8),
        )

    iscale = float(nmax) / (max_v - min_v)
    scale = 1.0 / iscale
    l_vals = torch.zeros_like(x, dtype=torch.uint8)
    best_error = torch.tensor(0.0, device=x.device, dtype=torch.float32)
    for i in range(x.numel()):
        l = int(_nearest_int((iscale * (x[i] - min_v)).reshape(1))[0].item())
        l = max(0, min(nmax, l))
        l_vals[i] = l
        diff = scale * float(l) + min_v - x[i]
        diff = torch.abs(diff) if use_mad else diff * diff
        best_error = best_error + weights[i] * diff

    for is_ in range(nstep + 1):
        iscale = (float(rmin) + float(rdelta) * float(is_) + float(nmax)) / (
            max_v - min_v
        )
        laux = torch.zeros_like(x, dtype=torch.uint8)
        sum_l = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        sum_l2 = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        sum_xl = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        for i in range(x.numel()):
            l = int(_nearest_int((iscale * (x[i] - min_v)).reshape(1))[0].item())
            l = max(0, min(nmax, l))
            laux[i] = l
            w = weights[i]
            lf = torch.tensor(float(l), device=x.device, dtype=torch.float32)
            sum_l = sum_l + w * lf
            sum_l2 = sum_l2 + w * lf * lf
            sum_xl = sum_xl + w * lf * x[i]

        denom = sum_w * sum_l2 - sum_l * sum_l
        if bool(denom > 0):
            this_scale = (sum_w * sum_xl - sum_x * sum_l) / denom
            this_min = (sum_l2 * sum_x - sum_l * sum_xl) / denom
            if bool(this_min > 0):
                this_min = torch.tensor(0.0, device=x.device, dtype=torch.float32)
                this_scale = sum_xl / sum_l2

            cur_error = torch.tensor(0.0, device=x.device, dtype=torch.float32)
            for i in range(x.numel()):
                diff = this_scale * laux[i].to(torch.float32) + this_min - x[i]
                diff = torch.abs(diff) if use_mad else diff * diff
                cur_error = cur_error + weights[i] * diff

            if bool(cur_error < best_error):
                l_vals = laux
                best_error = cur_error
                scale = this_scale
                min_v = this_min

    return scale, -min_v, l_vals


# Full weights #
def quantize_blocks_BF16(blocks, block_size, type_size):
    n = blocks.contiguous().view(torch.int32).to(torch.int64)
    n = torch.where(
        (n & 0x7FFFFFFF) > 0x7F800000,
        (n & 0xFFFF0000) | (64 << 16),
        n,
    )
    n = (n + (0x7FFF + ((n >> 16) & 1))) >> 16
    return n.to(torch.int16).contiguous().view(torch.uint8)


def dequantize_blocks_BF16(blocks, block_size, type_size, dtype=None):
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)


# Legacy Quants #
def quantize_blocks_Q1_0(blocks, block_size, type_size):
    n_blocks = blocks.shape[0]

    d = _fp16_to_bytes(torch.mean(torch.abs(blocks), dim=-1, keepdim=True))
    signs = (blocks >= 0).to(torch.uint8).reshape((n_blocks, block_size // 8, 8))
    qs = _pack_bits_little(signs)

    return torch.cat([d, qs], dim=-1)


def dequantize_blocks_Q1_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)

    bits = qs.reshape((n_blocks, block_size // 8, 1)) >> torch.arange(
        8, device=qs.device, dtype=torch.uint8
    ).reshape((1, 1, 8))
    bits = (bits & 0x01).reshape((n_blocks, block_size)).bool()

    return torch.where(bits, d, -d)


def quantize_blocks_Q8_0(blocks, block_size, type_size):
    d = torch.max(torch.abs(blocks), dim=1, keepdim=True).values / 127
    id = torch.where(d == 0, torch.zeros_like(d), 1 / d)
    qs = _round_away_from_zero(blocks * id).to(torch.int8).view(torch.uint8)
    return torch.cat([_fp16_to_bytes(d), qs], dim=1)


def dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    d, x = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return d * x


def quantize_blocks_Q5_1(blocks, block_size, type_size):
    n_blocks = blocks.shape[0]

    max_v = torch.max(blocks, dim=-1, keepdim=True).values
    min_v = torch.min(blocks, dim=-1, keepdim=True).values

    d = (max_v - min_v) / 31
    id = torch.where(d == 0, torch.zeros_like(d), 1 / d)
    q = torch.trunc((blocks - min_v) * id + 0.5).to(torch.uint8).clip(0, 31)

    qs = q.reshape((n_blocks, 2, block_size // 2))
    qs = (qs[:, 0, :] & 0x0F) | (qs[:, 1, :] << 4)
    qh = _pack_bits_little(q >> 4).reshape(n_blocks, 4)

    return torch.cat([_fp16_to_bytes(d), _fp16_to_bytes(min_v), qh, qs], dim=-1)


def dequantize_blocks_Q5_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, m, qh, qs = split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = to_uint32(qh)

    qh = qh.reshape((n_blocks, 1)) >> torch.arange(
        32, device=d.device, dtype=torch.int32
    ).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))

    qs = ql | (qh << 4)
    return (d * qs) + m


def quantize_blocks_Q5_0(blocks, block_size, type_size):
    n_blocks = blocks.shape[0]

    imax = torch.argmax(torch.abs(blocks), dim=-1, keepdim=True)
    max_v = torch.gather(blocks, dim=-1, index=imax)

    d = max_v / -16
    id = torch.where(d == 0, torch.zeros_like(d), 1 / d)
    q = torch.trunc((blocks * id) + 16.5).to(torch.uint8).clip(0, 31)

    qs = q.reshape((n_blocks, 2, block_size // 2))
    qs = (qs[:, 0, :] & 0x0F) | (qs[:, 1, :] << 4)
    qh = _pack_bits_little(q >> 4).reshape(n_blocks, 4)

    return torch.cat([_fp16_to_bytes(d), qh, qs], dim=-1)


def dequantize_blocks_Q5_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qh, qs = split_block_dims(blocks, 2, 4)
    d = d.view(torch.float16).to(dtype)
    qh = to_uint32(qh)

    qh = qh.reshape(n_blocks, 1) >> torch.arange(
        32, device=d.device, dtype=torch.int32
    ).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)

    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)

    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return d * qs


def quantize_blocks_Q4_1(blocks, block_size, type_size):
    n_blocks = blocks.shape[0]

    max_v = torch.max(blocks, dim=-1, keepdim=True).values
    min_v = torch.min(blocks, dim=-1, keepdim=True).values

    d = (max_v - min_v) / 15
    id = torch.where(d == 0, torch.zeros_like(d), 1 / d)
    qs = torch.trunc((blocks - min_v) * id + 0.5).to(torch.uint8).clip(0, 15)

    qs = qs.reshape((n_blocks, 2, block_size // 2))
    qs = qs[:, 0, :] | (qs[:, 1, :] << 4)

    return torch.cat([_fp16_to_bytes(d), _fp16_to_bytes(min_v), qs], dim=-1)


def dequantize_blocks_Q4_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, m, qs = split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)

    return (d * qs) + m


def quantize_blocks_Q4_0(blocks, block_size, type_size):
    n_blocks = blocks.shape[0]

    imax = torch.argmax(torch.abs(blocks), dim=-1, keepdim=True)
    max_v = torch.gather(blocks, dim=-1, index=imax)

    d = max_v / -8
    id = torch.where(d == 0, torch.zeros_like(d), 1 / d)
    qs = torch.trunc((blocks * id) + 8.5).to(torch.uint8).clip(0, 15)

    qs = qs.reshape((n_blocks, 2, block_size // 2))
    qs = qs[:, 0, :] | (qs[:, 1, :] << 4)

    return torch.cat([_fp16_to_bytes(d), qs], dim=-1)


def dequantize_blocks_Q4_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int8) - 8
    return d * qs


# K Quants #
QK_K = 256
K_SCALE_SIZE = 12


def get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))

    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)

    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    min = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)

    return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))


def quantize_blocks_Q2_K(blocks, block_size, type_size):
    n_blocks = blocks.shape[0]
    out = torch.zeros((n_blocks, type_size), device=blocks.device, dtype=torch.uint8)

    for i, x in enumerate(blocks):
        L = torch.zeros(QK_K, device=blocks.device, dtype=torch.uint8)
        scales = torch.zeros(QK_K // 16, device=blocks.device, dtype=torch.float32)
        mins = torch.zeros(QK_K // 16, device=blocks.device, dtype=torch.float32)
        max_scale = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
        max_min = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)

        for j in range(QK_K // 16):
            start = 16 * j
            sub = x[start : start + 16]
            scale, min_v, quants = _make_qkx2_quants(
                sub, torch.abs(sub), 3, -0.5, 0.1, 15, use_mad=True
            )
            scales[j] = scale
            mins[j] = min_v
            L[start : start + 16] = quants
            if bool(scale > max_scale):
                max_scale = scale
            if bool(min_v > max_min):
                max_min = min_v

        packed_scales = out[i, : QK_K // 16]
        if bool(max_scale > 0):
            iscale = 15.0 / max_scale
            for j in range(QK_K // 16):
                packed_scales[j] = int(
                    _nearest_int((iscale * scales[j]).reshape(1))[0].item()
                )
            d = (max_scale / 15.0).reshape(1).to(torch.float16)
        else:
            d = torch.zeros(1, device=blocks.device, dtype=torch.float16)

        if bool(max_min > 0):
            iscale = 15.0 / max_min
            for j in range(QK_K // 16):
                packed_scales[j] |= (
                    int(_nearest_int((iscale * mins[j]).reshape(1))[0].item()) << 4
                )
            dmin = (max_min / 15.0).reshape(1).to(torch.float16)
        else:
            dmin = torch.zeros(1, device=blocks.device, dtype=torch.float16)

        out[i, -4:-2] = d.contiguous().view(torch.uint8)
        out[i, -2:] = dmin.contiguous().view(torch.uint8)
        d_f32 = d.to(torch.float32)[0]
        dmin_f32 = dmin.to(torch.float32)[0]

        for j in range(QK_K // 16):
            ps = int(packed_scales[j].item())
            d_sub = d_f32 * float(ps & 0x0F)
            if bool(d_sub == 0):
                continue
            dm_sub = dmin_f32 * float(ps >> 4)
            start = 16 * j
            L[start : start + 16] = torch.clamp(
                _nearest_int((x[start : start + 16] + dm_sub) / d_sub), 0, 3
            ).to(torch.uint8)

        qs = out[i, QK_K // 16 : QK_K // 16 + QK_K // 4]
        for j in range(0, QK_K, 128):
            for l in range(32):
                qs[j // 4 + l] = (
                    L[j + l]
                    | (L[j + l + 32] << 2)
                    | (L[j + l + 64] << 4)
                    | (L[j + l + 96] << 6)
                )

    return out


def dequantize_blocks_Q6_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    (
        ql,
        qh,
        scales,
        d,
    ) = split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)

    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))

    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 2, 4, 6], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 4, 1))
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))

    return (d * q).reshape((n_blocks, QK_K))


def quantize_blocks_Q6_K(blocks, block_size, type_size):
    n_blocks = blocks.shape[0]
    out = torch.zeros((n_blocks, type_size), device=blocks.device, dtype=torch.uint8)

    for i, x in enumerate(blocks):
        L = torch.zeros(QK_K, device=blocks.device, dtype=torch.int8)
        scales = torch.zeros(QK_K // 16, device=blocks.device, dtype=torch.float32)
        max_scale = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
        max_abs_scale = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)

        for ib in range(QK_K // 16):
            start = 16 * ib
            scale, quants = _make_qx_quants(x[start : start + 16], 32)
            L[start : start + 16] = quants
            scales[ib] = scale
            abs_scale = torch.abs(scale)
            if bool(abs_scale > max_abs_scale):
                max_abs_scale = abs_scale
                max_scale = scale

        if bool(max_abs_scale < GROUP_MAX_EPS):
            continue

        iscale = -128.0 / max_scale
        d = (1.0 / iscale).reshape(1).to(torch.float16)
        d_f32 = d.to(torch.float32)[0]
        q_scales = torch.minimum(
            torch.full_like(scales, 127, dtype=torch.int32),
            _nearest_int(iscale * scales),
        ).to(torch.int8)

        for j in range(QK_K // 16):
            d_sub = d_f32 * q_scales[j].to(torch.float32)
            if bool(d_sub == 0):
                continue
            start = 16 * j
            L[start : start + 16] = (
                torch.clamp(_nearest_int(x[start : start + 16] / d_sub), -32, 31).to(
                    torch.int8
                )
                + 32
            )

        ql = out[i, : QK_K // 2]
        qh = out[i, QK_K // 2 : QK_K // 2 + QK_K // 4]
        for j in range(0, QK_K, 128):
            ql_base = j // 2
            qh_base = j // 4
            for l in range(32):
                q1 = int(L[j + l + 0].item()) & 0x0F
                q2 = int(L[j + l + 32].item()) & 0x0F
                q3 = int(L[j + l + 64].item()) & 0x0F
                q4 = int(L[j + l + 96].item()) & 0x0F
                ql[ql_base + l + 0] = q1 | (q3 << 4)
                ql[ql_base + l + 32] = q2 | (q4 << 4)
                qh[qh_base + l] = (
                    (int(L[j + l].item()) >> 4)
                    | ((int(L[j + l + 32].item()) >> 4) << 2)
                    | ((int(L[j + l + 64].item()) >> 4) << 4)
                    | ((int(L[j + l + 96].item()) >> 4) << 6)
                )

        scales_start = QK_K // 2 + QK_K // 4
        out[i, scales_start : scales_start + QK_K // 16] = q_scales.view(torch.uint8)
        out[i, -2:] = d.contiguous().view(torch.uint8)

    return out


def dequantize_blocks_Q5_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, dmin, scales, qh, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)

    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [i for i in range(8)], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = ql | (qh << 4)

    return (d * q - dm).reshape((n_blocks, QK_K))


def _pack_k_scale_min(scales_packed, ls_j, lm_j, j):
    if j < 4:
        scales_packed[j] = ls_j
        scales_packed[j + 4] = lm_j
    else:
        scales_packed[j + 4] = (ls_j & 0x0F) | ((lm_j & 0x0F) << 4)
        scales_packed[j - 4] |= (ls_j >> 4) << 6
        scales_packed[j] |= (lm_j >> 4) << 6


def quantize_blocks_Q5_K(blocks, block_size, type_size):
    n_blocks = blocks.shape[0]
    out = torch.zeros((n_blocks, type_size), device=blocks.device, dtype=torch.uint8)

    for i, x in enumerate(blocks):
        L = torch.zeros(QK_K, device=blocks.device, dtype=torch.uint8)
        scales = torch.zeros(QK_K // 32, device=blocks.device, dtype=torch.float32)
        mins = torch.zeros(QK_K // 32, device=blocks.device, dtype=torch.float32)
        max_scale = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
        max_min = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)

        for j in range(QK_K // 32):
            start = 32 * j
            sub = x[start : start + 32]
            av_x = torch.sqrt(torch.sum(sub * sub) / 32.0)
            scale, min_v, quants = _make_qkx2_quants(
                sub, av_x + torch.abs(sub), 31, -0.5, 0.1, 15
            )
            scales[j] = scale
            mins[j] = min_v
            L[start : start + 32] = quants
            if bool(scale > max_scale):
                max_scale = scale
            if bool(min_v > max_min):
                max_min = min_v

        inv_scale = (
            63.0 / max_scale
            if bool(max_scale > 0)
            else torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
        )
        inv_min = (
            63.0 / max_min
            if bool(max_min > 0)
            else torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
        )
        ls = torch.zeros(QK_K // 32, device=blocks.device, dtype=torch.uint8)
        lm = torch.zeros(QK_K // 32, device=blocks.device, dtype=torch.uint8)
        scales_packed = out[i, 4:16]
        for j in range(QK_K // 32):
            ls_j = min(
                63, int(_nearest_int((inv_scale * scales[j]).reshape(1))[0].item())
            )
            lm_j = min(63, int(_nearest_int((inv_min * mins[j]).reshape(1))[0].item()))
            ls[j] = ls_j
            lm[j] = lm_j
            _pack_k_scale_min(scales_packed, ls_j, lm_j, j)

        d = (max_scale / 63.0).reshape(1).to(torch.float16)
        dmin = (max_min / 63.0).reshape(1).to(torch.float16)
        out[i, :2] = d.contiguous().view(torch.uint8)
        out[i, 2:4] = dmin.contiguous().view(torch.uint8)
        d_f32 = d.to(torch.float32)[0]
        dmin_f32 = dmin.to(torch.float32)[0]

        for j in range(QK_K // 32):
            d_sub = d_f32 * ls[j].to(torch.float32)
            if bool(d_sub == 0):
                continue
            dm_sub = dmin_f32 * lm[j].to(torch.float32)
            start = 32 * j
            L[start : start + 32] = torch.clamp(
                _nearest_int((x[start : start + 32] + dm_sub) / d_sub), 0, 31
            ).to(torch.uint8)

        qh = out[i, 16 : 16 + QK_K // 8]
        ql = out[i, 16 + QK_K // 8 :]
        m1 = 1
        m2 = 2
        for n in range(0, QK_K, 64):
            ql_base = n // 2
            for j in range(32):
                l1 = int(L[n + j].item())
                if l1 > 15:
                    l1 -= 16
                    qh[j] |= m1
                l2 = int(L[n + j + 32].item())
                if l2 > 15:
                    l2 -= 16
                    qh[j] |= m2
                ql[ql_base + j] = l1 | (l2 << 4)
            m1 <<= 2
            m2 <<= 2

    return out


def dequantize_blocks_Q4_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, dmin, scales, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))

    return (d * qs - dm).reshape((n_blocks, QK_K))


def quantize_blocks_Q4_K(blocks, block_size, type_size):
    n_blocks = blocks.shape[0]
    out = torch.zeros((n_blocks, type_size), device=blocks.device, dtype=torch.uint8)

    for i, x in enumerate(blocks):
        L = torch.zeros(QK_K, device=blocks.device, dtype=torch.uint8)
        scales = torch.zeros(QK_K // 32, device=blocks.device, dtype=torch.float32)
        mins = torch.zeros(QK_K // 32, device=blocks.device, dtype=torch.float32)
        max_scale = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
        max_min = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)

        for j in range(QK_K // 32):
            start = 32 * j
            sub = x[start : start + 32]
            av_x = torch.sqrt(torch.sum(sub * sub) / 32.0)
            scale, min_v, quants = _make_qkx2_quants(
                sub, av_x + torch.abs(sub), 15, -1.0, 0.1, 20
            )
            scales[j] = scale
            mins[j] = min_v
            L[start : start + 32] = quants
            if bool(scale > max_scale):
                max_scale = scale
            if bool(min_v > max_min):
                max_min = min_v

        inv_scale = (
            63.0 / max_scale
            if bool(max_scale > 0)
            else torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
        )
        inv_min = (
            63.0 / max_min
            if bool(max_min > 0)
            else torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
        )
        ls = torch.zeros(QK_K // 32, device=blocks.device, dtype=torch.uint8)
        lm = torch.zeros(QK_K // 32, device=blocks.device, dtype=torch.uint8)
        scales_packed = out[i, 4:16]
        for j in range(QK_K // 32):
            ls_j = min(
                63, int(_nearest_int((inv_scale * scales[j]).reshape(1))[0].item())
            )
            lm_j = min(63, int(_nearest_int((inv_min * mins[j]).reshape(1))[0].item()))
            ls[j] = ls_j
            lm[j] = lm_j
            _pack_k_scale_min(scales_packed, ls_j, lm_j, j)

        d = (max_scale / 63.0).reshape(1).to(torch.float16)
        dmin = (max_min / 63.0).reshape(1).to(torch.float16)
        out[i, :2] = d.contiguous().view(torch.uint8)
        out[i, 2:4] = dmin.contiguous().view(torch.uint8)
        d_f32 = d.to(torch.float32)[0]
        dmin_f32 = dmin.to(torch.float32)[0]

        for j in range(QK_K // 32):
            d_sub = d_f32 * ls[j].to(torch.float32)
            if bool(d_sub == 0):
                continue
            dm_sub = dmin_f32 * lm[j].to(torch.float32)
            start = 32 * j
            L[start : start + 32] = torch.clamp(
                _nearest_int((x[start : start + 32] + dm_sub) / d_sub), 0, 15
            ).to(torch.uint8)

        qs = out[i, 16:]
        for j in range(0, QK_K, 64):
            for l in range(32):
                qs[j // 2 + l] = L[j + l] | (L[j + l + 32] << 4)

    return out


def quantize_blocks_Q3_K(blocks, block_size, type_size):
    n_blocks = blocks.shape[0]
    out = torch.zeros((n_blocks, type_size), device=blocks.device, dtype=torch.uint8)

    for i, x in enumerate(blocks):
        L = torch.zeros(QK_K, device=blocks.device, dtype=torch.int8)
        scales = torch.zeros(QK_K // 16, device=blocks.device, dtype=torch.float32)
        max_scale = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
        amax = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)

        for j in range(QK_K // 16):
            start = 16 * j
            scale, quants = _make_q3_quants(x[start : start + 16], 4)
            scales[j] = scale
            L[start : start + 16] = quants
            scale_abs = torch.abs(scale)
            if bool(scale_abs > amax):
                amax = scale_abs
                max_scale = scale

        if bool(max_scale != 0):
            iscale = -32.0 / max_scale
            Ls = (_nearest_int(iscale * scales).clamp(-32, 31) + 32).to(torch.int8)
            d = (1.0 / iscale).reshape(1).to(torch.float16)
        else:
            Ls = torch.zeros(QK_K // 16, device=blocks.device, dtype=torch.int8)
            d = torch.zeros(1, device=blocks.device, dtype=torch.float16)

        scales_packed = out[i, QK_K // 8 + QK_K // 4 : QK_K // 8 + QK_K // 4 + 12]
        for j in range(QK_K // 16):
            l = int(Ls[j].item())
            if j < 8:
                scales_packed[j] = l & 0x0F
            else:
                scales_packed[j - 8] |= (l & 0x0F) << 4
            l >>= 4
            scales_packed[j % 4 + 8] |= l << (2 * (j // 4))

        d_f32 = d.to(torch.float32)[0]
        out[i, -2:] = d.contiguous().view(torch.uint8)

        for j in range(QK_K // 16):
            sc = int(Ls[j].item()) - 32
            d_sub = d_f32 * float(sc)
            if bool(d_sub == 0):
                continue
            start = 16 * j
            L[start : start + 16] = (
                torch.clamp(_nearest_int(x[start : start + 16] / d_sub), -4, 3).to(
                    torch.int8
                )
                + 4
            )

        hmask = out[i, : QK_K // 8]
        m = 0
        hm = 1
        for j in range(QK_K):
            if int(L[j].item()) > 3:
                hmask[m] |= hm
                L[j] -= 4
            m += 1
            if m == QK_K // 8:
                m = 0
                hm <<= 1

        qs = out[i, QK_K // 8 : QK_K // 8 + QK_K // 4]
        for j in range(0, QK_K, 128):
            for l in range(32):
                qs[j // 4 + l] = (
                    (int(L[j + l].item()) & 0x03)
                    | ((int(L[j + l + 32].item()) & 0x03) << 2)
                    | ((int(L[j + l + 64].item()) & 0x03) << 4)
                    | ((int(L[j + l + 96].item()) & 0x03) << 6)
                )

    return out


def dequantize_blocks_Q3_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    hmask, qs, scales, d = split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)

    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor(
        [0, 2, 4, 6], device=d.device, dtype=torch.uint8
    ).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = scales.to(torch.int8) - 32

    dl = (d * scales).reshape((n_blocks, 16, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 2, 4, 6], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.tensor(
        [i for i in range(8)], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = ql.to(torch.int8) - (qh << 2).to(torch.int8)

    return (dl * q).reshape((n_blocks, QK_K))


def dequantize_blocks_Q2_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    scales, qs, d, dmin = split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    # (n_blocks, 16, 1)
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))

    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 4, 1)
    )

    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml

    return qs.reshape((n_blocks, -1))


# IQ quants
KVALUES = torch.tensor(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=torch.int8,
)


def quantize_blocks_TQ1_0(blocks, block_size, type_size):
    n_blocks = blocks.shape[0]

    d = torch.max(torch.abs(blocks), dim=-1, keepdim=True).values
    id = torch.where(d == 0, torch.zeros_like(d), 1 / d)
    qs = (_round_away_from_zero(blocks * id).to(torch.int8) + 1).to(torch.uint8)

    qs0, qs1, qh = qs[:, :160], qs[:, 160:240], qs[:, 240:]
    factors5 = torch.tensor(
        [81, 27, 9, 3, 1], device=blocks.device, dtype=torch.uint8
    ).reshape(1, 1, 5, 1)
    factors4 = torch.tensor(
        [81, 27, 9, 3], device=blocks.device, dtype=torch.uint8
    ).reshape(1, 1, 4, 1)
    qs0 = torch.sum(qs0.reshape((n_blocks, -1, 5, 32)) * factors5, dim=-2).reshape(
        (n_blocks, -1)
    )
    qs1 = torch.sum(qs1.reshape((n_blocks, -1, 5, 16)) * factors5, dim=-2).reshape(
        (n_blocks, -1)
    )
    qh = torch.sum(qh.reshape((n_blocks, -1, 4, 4)) * factors4, dim=-2).reshape(
        (n_blocks, -1)
    )
    qs = torch.cat([qs0, qs1, qh], dim=-1)
    qs = ((qs.to(torch.int32) * 256 + 242) // 243).to(torch.uint8)

    return torch.cat([qs, _fp16_to_bytes(d)], dim=-1)


def dequantize_blocks_TQ1_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    qs, rest = split_block_dims(blocks, (QK_K - 4 * QK_K // 64) // 5)
    qh, d = split_block_dims(rest, QK_K // 64)
    d = d.view(torch.float16).to(dtype)

    qs0, qs1 = torch.split(qs, [32, qs.shape[1] - 32], dim=1)
    qs0 = qs0.reshape((n_blocks, -1, 1, 32)) * torch.tensor(
        [1, 3, 9, 27, 81], device=blocks.device, dtype=torch.uint8
    ).reshape((1, 1, 5, 1))
    qs0 = qs0.reshape((n_blocks, -1))

    qs1 = qs1.reshape((n_blocks, -1, 1, 16)) * torch.tensor(
        [1, 3, 9, 27, 81], device=blocks.device, dtype=torch.uint8
    ).reshape((1, 1, 5, 1))
    qs1 = qs1.reshape((n_blocks, -1))

    qh = qh.reshape((n_blocks, -1, 1, 4)) * torch.tensor(
        [1, 3, 9, 27], device=blocks.device, dtype=torch.uint8
    ).reshape((1, 1, 4, 1))
    qh = qh.reshape((n_blocks, -1))

    qs = torch.cat([qs0, qs1, qh], dim=-1)
    qs = ((qs.to(torch.int32) * 3) >> 8).to(torch.int8) - 1

    return d * qs.to(d.dtype)


def quantize_blocks_TQ2_0(blocks, block_size, type_size):
    n_blocks = blocks.shape[0]

    d = torch.max(torch.abs(blocks), dim=-1, keepdim=True).values
    id = torch.where(d == 0, torch.zeros_like(d), 1 / d)
    qs = (_round_away_from_zero(blocks * id).to(torch.int8) + 1).to(torch.uint8)

    qs = qs.reshape((n_blocks, -1, 4, 32)) << torch.tensor(
        [0, 2, 4, 6], device=blocks.device, dtype=torch.uint8
    ).reshape((1, 1, 4, 1))
    qs = qs[:, :, 0, :] | qs[:, :, 1, :] | qs[:, :, 2, :] | qs[:, :, 3, :]
    qs = qs.reshape((n_blocks, -1))

    return torch.cat([qs, _fp16_to_bytes(d)], dim=-1)


def dequantize_blocks_TQ2_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    qs, d = split_block_dims(blocks, QK_K // 4)
    d = d.view(torch.float16).to(dtype)

    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 2, 4, 6], device=blocks.device, dtype=torch.uint8
    ).reshape((1, 1, 4, 1))
    qs = (qs & 0x03).reshape((n_blocks, -1)).to(torch.int8) - 1

    return d * qs.to(d.dtype)


def e8m0_to_fp32_half(x):
    bits = torch.where(
        x < 2,
        torch.tensor(0x00200000, device=x.device, dtype=torch.int32)
        << x.to(torch.int32),
        (x.to(torch.int32) - 1) << 23,
    )
    return bits.view(torch.float32)


def quantize_blocks_MXFP4(blocks, block_size, type_size):
    n_blocks = blocks.shape[0]

    d = torch.max(torch.abs(blocks), dim=-1, keepdim=True).values
    e = torch.where(
        d > 0,
        torch.floor(torch.log2(d)) - 2 + 127,
        torch.zeros_like(d),
    ).to(torch.uint8)
    d = e8m0_to_fp32_half(e)

    kvalues = torch.tensor(
        [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
        device=blocks.device,
        dtype=torch.float32,
    ).reshape((1, 1, 16))
    errs = torch.abs(
        d.reshape((n_blocks, 1, 1)) * kvalues
        - blocks.reshape((n_blocks, block_size, 1))
    )
    best = torch.argmin(errs, dim=-1, keepdim=True).to(torch.uint8)

    qs = best.reshape(n_blocks, 2, block_size // 2)
    qs = qs[:, 0] | (qs[:, 1] << 4)

    return torch.cat([e, qs.reshape((n_blocks, block_size // 2))], dim=-1)


def dequantize_blocks_MXFP4(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    e, qs = split_block_dims(blocks, 1)
    d = e8m0_to_fp32_half(e).to(dtype if dtype is not None else torch.float32)

    qs = qs.reshape((n_blocks, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=blocks.device, dtype=torch.uint8
    ).reshape((1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int64)

    kvalues = torch.tensor(
        [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
        device=blocks.device,
        dtype=d.dtype,
    )
    qs = kvalues[qs].reshape((n_blocks, block_size))

    return d * qs


def ue4m3_to_fp32(x):
    exp = (x >> 3).to(torch.int32) & 0xF
    man = (x & 0x07).to(torch.float32)
    two = torch.tensor(2.0, device=x.device, dtype=torch.float32)
    raw = torch.where(
        exp == 0,
        man * (2.0**-9),
        (1.0 + man / 8.0) * torch.pow(two, exp.to(torch.float32) - 7.0),
    )
    return torch.where((x == 0) | (x == 0x7F), 0.0, raw * 0.5)


def fp32_to_ue4m3(x):
    x = torch.clamp(x, 0.0, 448.0).to(torch.float32).contiguous()
    bits = x.view(torch.int32)
    fp32_exp = ((bits >> 23) & 0xFF) - 127
    fp32_man = (bits >> 20) & 0x07
    ue4m3_exp = fp32_exp + 7

    sub_man = torch.clamp((x * 512.0 + 0.5).to(torch.int32), 0, 7)
    sub_result = torch.where(sub_man >= 1, sub_man, torch.zeros_like(sub_man)).to(
        torch.uint8
    )

    round_bit = (bits >> 19) & 1
    man = fp32_man + round_bit
    exp = ue4m3_exp
    overflow = man > 7
    man = torch.where(overflow, torch.zeros_like(man), man)
    exp = torch.where(overflow, exp + 1, exp)
    normal_result = torch.where(
        exp >= 15,
        torch.full_like(exp, 0x7E),
        (exp << 3) | man,
    ).to(torch.uint8)

    return torch.where(
        x <= 0.0,
        torch.zeros_like(sub_result),
        torch.where(
            ue4m3_exp <= 0,
            sub_result,
            torch.where(
                ue4m3_exp >= 15, torch.full_like(sub_result, 0x7E), normal_result
            ),
        ),
    )


def _grid_tensor(cls, device):
    cls.init_grid()
    return torch.from_numpy(cls.grid.reshape(cls.grid_shape)).to(device=device)


def _ksigns_tensor(device):
    return torch.tensor(list(IQ2_XXS.ksigns), device=device, dtype=torch.uint8)


def _best_lattice_neighbour_torch(neighbours, grid_l, xval, weight, scale, tie_eps=0.0):
    best_d2 = torch.tensor(float("inf"), device=xval.device, dtype=torch.float32)
    best_index = -1
    for index in neighbours:
        pos = (2 * grid_l[int(index)].to(torch.float32)) + 1.0
        diff = scale * pos - xval
        d2 = torch.sum(weight * diff * diff)
        if bool(
            (d2 < best_d2 - tie_eps)
            | (
                (tie_eps > 0.0)
                & (torch.abs(d2 - best_d2) <= tie_eps)
                & (int(index) < best_index)
            )
        ):
            best_d2 = d2
            best_index = int(index)
    return best_index, grid_l[best_index].to(torch.int8)


def _best_iq1_neighbour_torch(neighbours, grid_l, xval, weight, scale, values):
    best_d2 = torch.tensor(float("inf"), device=xval.device, dtype=torch.float32)
    best_index = -1
    for index in neighbours:
        levels = grid_l[int(index)].to(torch.int64)
        q = values[levels]
        diff = scale * q - xval
        d2 = torch.sum(weight * diff * diff)
        if bool(d2 < best_d2):
            best_d2 = d2
            best_index = int(index)
    return best_index, grid_l[best_index].to(torch.int8)


def _make_qp_quants_torch(x, nmax, quant_weights):
    max_v = torch.max(x)
    L = torch.zeros_like(x, dtype=torch.uint8)
    if bool(max_v < GROUP_MAX_EPS):
        return torch.tensor(0.0, device=x.device, dtype=torch.float32), L

    iscale = float(nmax) / max_v
    for i in range(x.numel()):
        L[i] = int(_nearest_int((iscale * x[i]).reshape(1))[0].item()) & 0xFF
    scale = 1.0 / iscale
    best_mse = torch.tensor(0.0, device=x.device, dtype=torch.float32)
    for i in range(x.numel()):
        diff = x[i] - scale * float(int(L[i].item()))
        best_mse = best_mse + quant_weights[i] * diff * diff

    for is_ in range(-4, 5):
        if is_ == 0:
            continue
        iscale_is = (0.1 * float(is_) + float(nmax)) / max_v
        scale_is = 1.0 / iscale_is
        mse = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        for i in range(x.numel()):
            l = int(_nearest_int((iscale_is * x[i]).reshape(1))[0].item())
            l = min(nmax, l)
            diff = x[i] - scale_is * float(l)
            mse = mse + quant_weights[i] * diff * diff
        if bool(mse < best_mse):
            best_mse = mse
            iscale = iscale_is

    sumlx = torch.tensor(0.0, device=x.device, dtype=torch.float32)
    suml2 = torch.tensor(0.0, device=x.device, dtype=torch.float32)
    for i in range(x.numel()):
        l = int(_nearest_int((iscale * x[i]).reshape(1))[0].item())
        l = min(nmax, l)
        L[i] = l & 0xFF
        lf = float(int(L[i].item()))
        w = quant_weights[i]
        sumlx = sumlx + w * x[i] * lf
        suml2 = suml2 + w * lf * lf

    for _ in range(5):
        n_changed = 0
        for i in range(x.numel()):
            w = quant_weights[i]
            lf = float(int(L[i].item()))
            slx = sumlx - w * x[i] * lf
            sl2 = suml2 - w * lf * lf
            if bool((slx > 0) & (sl2 > 0)):
                new_l = int(_nearest_int((x[i] * sl2 / slx).reshape(1))[0].item())
                new_l = min(nmax, new_l)
                if new_l != int(L[i].item()):
                    new_lf = float(new_l)
                    slx = slx + w * x[i] * new_lf
                    sl2 = sl2 + w * new_lf * new_lf
                    if bool(slx * slx * suml2 > sumlx * sumlx * sl2):
                        L[i] = new_l & 0xFF
                        sumlx = slx
                        suml2 = sl2
                        n_changed += 1
        if n_changed == 0:
            break

    scale = (
        sumlx / suml2
        if bool(suml2 > 0)
        else torch.tensor(0.0, device=x.device, dtype=torch.float32)
    )
    return scale, L


def _iq3_lookup_torch(device, *, is_xxs):
    if is_xxs:
        kmap, neighbours, grid_l = _get_iq3_xxs_lookup()
    else:
        kmap, neighbours, grid_l = _get_iq3_s_lookup()
    return (
        torch.from_numpy(kmap).to(device=device),
        neighbours,
        torch.from_numpy(grid_l).to(device=device),
    )


def _iq2_xxs_lookup_torch(device):
    kmap, neighbours, grid_l = _get_iq2_xxs_lookup()
    return (
        torch.from_numpy(kmap).to(device=device),
        neighbours,
        torch.from_numpy(grid_l).to(device=device),
    )


def _iq2_xs_lookup_torch(device):
    kmap, neighbours, grid_l = _get_iq2_xs_lookup()
    return (
        torch.from_numpy(kmap).to(device=device),
        neighbours,
        torch.from_numpy(grid_l).to(device=device),
    )


def _iq2_s_lookup_torch(device):
    kmap, neighbours, grid_l = _get_iq2_s_lookup()
    return (
        torch.from_numpy(kmap).to(device=device),
        neighbours,
        torch.from_numpy(grid_l).to(device=device),
    )


def _iq1_s_lookup_torch(device):
    kmap, neighbours, grid_l = _get_iq1_s_lookup()
    return (
        torch.from_numpy(kmap).to(device=device),
        neighbours,
        torch.from_numpy(grid_l).to(device=device),
    )


def _quantize_iq3_blocks(blocks, type_size, *, is_xxs):
    n_blocks = blocks.shape[0]
    out = torch.zeros((n_blocks, type_size), device=blocks.device, dtype=torch.uint8)
    kmap, neighbours, grid_l = _iq3_lookup_torch(blocks.device, is_xxs=is_xxs)
    search_range = range(-15, 16) if is_xxs else range(-9, 10)
    scale_bias = 1.0125 if is_xxs else 1.033
    eps = GROUP_MAX_EPS_IQ3_XXS if is_xxs else 0.0

    for i, x in enumerate(blocks):
        qs = out[i, 2 : 2 + QK_K // 4]
        if is_xxs:
            scales_and_signs = torch.zeros(
                QK_K // 32, device=blocks.device, dtype=torch.int32
            )
            qh = signs = packed_scales = None
        else:
            qh = out[i, 2 + QK_K // 4 : 2 + QK_K // 4 + QK_K // 32]
            signs = out[
                i,
                2 + QK_K // 4 + QK_K // 32 : 2 + QK_K // 4 + QK_K // 32 + QK_K // 8,
            ]
            packed_scales = out[i, 2 + QK_K // 4 + QK_K // 32 + QK_K // 8 :]
            scales_and_signs = None

        scales = torch.zeros(QK_K // 32, device=blocks.device, dtype=torch.float32)
        max_scale = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
        for ib in range(QK_K // 32):
            xb = x[32 * ib : 32 * ib + 32]
            weight = xb * xb
            waux = torch.sqrt(weight)
            xval = torch.empty(32, device=blocks.device, dtype=torch.float32)
            block_signs = torch.zeros(4, device=blocks.device, dtype=torch.uint8)

            for k in range(4):
                nflip = 0
                sign = 0
                for j in range(8):
                    index = 8 * k + j
                    if bool(xb[index] >= 0):
                        xval[index] = xb[index]
                    else:
                        xval[index] = -xb[index]
                        nflip += 1
                        sign |= 1 << j
                if is_xxs and nflip % 2:
                    imin = 8 * k
                    min_v = weight[imin] * xb[imin] * xb[imin]
                    for j in range(1, 8):
                        index = 8 * k + j
                        ax = weight[index] * xb[index] * xb[index]
                        if bool(ax < min_v):
                            min_v = ax
                            imin = index
                    xval[imin] = -xval[imin]
                    sign ^= 1 << (imin - 8 * k)
                block_signs[k] = (sign & 0x7F) if is_xxs else sign

            max_v = torch.max(xval)
            L = torch.zeros(32, device=blocks.device, dtype=torch.int8)
            if bool(max_v <= eps):
                continue

            best = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
            scale = max_v / 15.0
            is_on_grid = (
                torch.ones(8, device=blocks.device, dtype=torch.bool)
                if is_xxs
                else torch.zeros(8, device=blocks.device, dtype=torch.bool)
            )
            for is_ in search_range:
                id_ = (15.0 + float(is_) * 0.2) / max_v
                this_scale = 1.0 / id_
                Laux = torch.zeros(32, device=blocks.device, dtype=torch.int8)
                is_on_grid_aux = torch.ones(8, device=blocks.device, dtype=torch.bool)
                for k in range(8):
                    u = 0
                    for j in range(4):
                        index = 4 * k + j
                        l = int(
                            _nearest_int((0.5 * (id_ * xval[index] - 1.0)).reshape(1))[
                                0
                            ].item()
                        )
                        l = max(0, min(7, l))
                        Laux[index] = l
                        u |= l << (3 * j)
                    grid_index = int(kmap[u].item())
                    if grid_index < 0:
                        is_on_grid_aux[k] = False
                        _, Laux[4 * k : 4 * k + 4] = _best_lattice_neighbour_torch(
                            neighbours[u],
                            grid_l,
                            xval[4 * k : 4 * k + 4],
                            waux[4 * k : 4 * k + 4],
                            this_scale,
                        )

                sumqx = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
                sumq2 = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
                for j in range(32):
                    q = 2.0 * float(int(Laux[j].item())) + 1.0
                    w = weight[j]
                    sumqx = sumqx + w * xval[j] * q
                    sumq2 = sumq2 + w * q * q
                if bool((sumq2 > 0) & (sumqx * sumqx > best * sumq2)):
                    scale = sumqx / sumq2
                    best = scale * sumqx
                    L[:] = Laux
                    is_on_grid[:] = is_on_grid_aux

            if bool((~is_on_grid).any() & (scale > 0)):
                id_ = 1.0 / scale
                for k in range(8):
                    if is_xxs and bool(is_on_grid[k]):
                        continue
                    u = 0
                    for j in range(4):
                        index = 4 * k + j
                        l = int(
                            _nearest_int((0.5 * (id_ * xval[index] - 1.0)).reshape(1))[
                                0
                            ].item()
                        )
                        l = max(0, min(7, l))
                        u |= l << (3 * j)
                    grid_index = int(kmap[u].item())
                    if grid_index < 0:
                        _, L[4 * k : 4 * k + 4] = _best_lattice_neighbour_torch(
                            neighbours[u],
                            grid_l,
                            xval[4 * k : 4 * k + 4],
                            waux[4 * k : 4 * k + 4],
                            scale,
                        )
                    else:
                        L[4 * k : 4 * k + 4] = grid_l[grid_index]

                sumqx = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
                sumq2 = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
                for j in range(32):
                    q = 2.0 * float(int(L[j].item())) + 1.0
                    w = weight[j]
                    sumqx = sumqx + w * xval[j] * q
                    sumq2 = sumq2 + w * q * q
                if bool(sumq2 > 0):
                    scale = sumqx / sumq2

            if bool(scale < 0):
                scale = -scale
                block_signs = (~block_signs).to(torch.uint8)

            for k in range(8):
                u = 0
                for j in range(4):
                    u |= int(L[4 * k + j].item()) << (3 * j)
                grid_index = int(kmap[u].item())
                qs[8 * ib + k] = grid_index if is_xxs else grid_index & 0xFF
                if not is_xxs:
                    qh[(8 * ib + k) // 8] |= ((grid_index >> 8) & 1) << (
                        (8 * ib + k) % 8
                    )

            if is_xxs:
                scales_and_signs[ib] = (
                    int(block_signs[0].item())
                    | (int(block_signs[1].item()) << 7)
                    | (int(block_signs[2].item()) << 14)
                    | (int(block_signs[3].item()) << 21)
                )
            else:
                signs[4 * ib : 4 * ib + 4] = block_signs
            scales[ib] = scale
            if bool(scale > max_scale):
                max_scale = scale

        if bool(max_scale == 0):
            continue

        d = max_scale / 31.0
        out[i, :2] = (
            (d * scale_bias).reshape(1).to(torch.float16).contiguous().view(torch.uint8)
        )
        id_ = 1.0 / d
        if is_xxs:
            out[i, 2 + QK_K // 4 :] = scales_and_signs.contiguous().view(torch.uint8)
            for ib in range(QK_K // 32):
                l = int(
                    _nearest_int((0.5 * (id_ * scales[ib] - 1.0)).reshape(1))[0].item()
                )
                l = max(0, min(15, l))
                scales_and_signs[ib] |= l << 28
            out[i, 2 + QK_K // 4 :] = scales_and_signs.contiguous().view(torch.uint8)
        else:
            for ib in range(0, QK_K // 32, 2):
                l1 = int(
                    _nearest_int((0.5 * (id_ * scales[ib] - 1.0)).reshape(1))[0].item()
                )
                l1 = max(0, min(15, l1))
                l2 = int(
                    _nearest_int((0.5 * (id_ * scales[ib + 1] - 1.0)).reshape(1))[
                        0
                    ].item()
                )
                l2 = max(0, min(15, l2))
                packed_scales[ib // 2] = l1 | (l2 << 4)

    return out


def quantize_blocks_IQ3_XXS(blocks, block_size, type_size):
    return _quantize_iq3_blocks(blocks, type_size, is_xxs=True)


def dequantize_blocks_IQ3_XXS(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, rest = split_block_dims(blocks, 2)
    qs, scales = split_block_dims(rest, QK_K // 4)

    d = d.view(torch.float16).to(dtype)
    scales = to_uint32_int64(scales.reshape(-1, 4)).reshape((n_blocks, -1))

    db = d * (0.5 + (scales >> 28).to(d.dtype)) * 0.5
    db = db.reshape((n_blocks, -1, 1, 1))

    signs = scales.reshape((n_blocks, -1, 1)) >> torch.tensor(
        [0, 7, 14, 21], device=blocks.device, dtype=torch.int64
    ).reshape((1, 1, 4))
    signs = (signs & 0x7F).to(torch.int64)
    ksigns = _ksigns_tensor(blocks.device)
    signs = ksigns[signs].reshape((n_blocks, -1, 4, 1)) >> torch.arange(
        8, device=blocks.device, dtype=torch.uint8
    ).reshape((1, 1, 1, 8))
    signs = signs & 0x01
    signs = torch.where(signs == 0, 1.0, -1.0).to(d.dtype)

    grid = _grid_tensor(IQ3_XXS, blocks.device).to(d.dtype)
    grid = grid[qs.to(torch.int64)].reshape((n_blocks, -1, 4, 8))

    return (db * grid * signs).reshape((n_blocks, -1))


def quantize_blocks_IQ3_S(blocks, block_size, type_size):
    return _quantize_iq3_blocks(blocks, type_size, is_xxs=False)


def dequantize_blocks_IQ3_S(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, rest = split_block_dims(blocks, 2)
    qs, rest = split_block_dims(rest, QK_K // 4)
    qh, rest = split_block_dims(rest, QK_K // 32)
    signs, scales = split_block_dims(rest, QK_K // 8)

    d = d.view(torch.float16).to(dtype)

    scales = scales.reshape((n_blocks, -1, 1)) >> torch.tensor(
        [0, 4], device=blocks.device, dtype=torch.uint8
    ).reshape((1, 1, 2))
    scales = (scales & 0x0F).reshape((n_blocks, -1))
    db = d * (1 + 2 * scales.to(d.dtype))
    db = db.reshape((n_blocks, -1, 1, 1))

    signs = signs.reshape((n_blocks, -1, 1)) >> torch.arange(
        8, device=blocks.device, dtype=torch.uint8
    ).reshape((1, 1, 8))
    signs = signs & 0x01
    signs = torch.where(signs == 0, 1.0, -1.0).to(d.dtype)
    signs = signs.reshape((n_blocks, -1, 4, 8))

    qh = qh.reshape((n_blocks, -1, 1)) >> torch.arange(
        8, device=blocks.device, dtype=torch.uint8
    ).reshape((1, 1, 8))
    qh = (qh & 0x01).to(torch.int64).reshape((n_blocks, -1))
    qs = qs.to(torch.int64) | (qh << 8)

    grid = _grid_tensor(IQ3_S, blocks.device).to(d.dtype)
    grid = grid[qs].reshape((n_blocks, -1, 4, 8))

    return (db * grid * signs).reshape((n_blocks, -1))


def _quantize_blocks_with_libgguf(blocks, qtype):
    quantized = _quantize_blocks_with_libgguf(
        blocks.detach().cpu().numpy().astype("float32", copy=False), qtype
    )
    return torch.from_numpy(quantized).to(blocks.device)


def quantize_blocks_IQ2_XXS(blocks, block_size, type_size):
    n_blocks = blocks.shape[0]
    out = torch.zeros((n_blocks, type_size), device=blocks.device, dtype=torch.uint8)
    kmap, neighbours, grid_l = _iq2_xxs_lookup_torch(blocks.device)
    quant_weights = torch.sum(blocks * blocks, dim=0, dtype=torch.float32)

    for i, x in enumerate(blocks):
        q2 = torch.zeros(QK_K // 16, device=blocks.device, dtype=torch.int64)
        scales = torch.zeros(QK_K // 32, device=blocks.device, dtype=torch.float32)
        max_scale = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
        sigma2 = torch.sum(x * x, dtype=torch.float32) / float(QK_K)

        for ib in range(QK_K // 32):
            xb = x[32 * ib : 32 * ib + 32]
            qw = quant_weights[32 * ib : 32 * ib + 32]
            weight = qw * torch.sqrt(sigma2 + xb * xb)
            waux = torch.sqrt(weight)
            xval = torch.empty(32, device=blocks.device, dtype=torch.float32)
            block_signs = torch.zeros(4, device=blocks.device, dtype=torch.uint8)

            for k in range(4):
                nflip = 0
                sign = 0
                for j in range(8):
                    index = 8 * k + j
                    if bool(xb[index] >= 0):
                        xval[index] = xb[index]
                    else:
                        xval[index] = -xb[index]
                        nflip += 1
                        sign |= 1 << j
                if nflip % 2:
                    imin = 8 * k
                    min_v = weight[imin] * xb[imin] * xb[imin]
                    for j in range(1, 8):
                        index = 8 * k + j
                        ax = weight[index] * xb[index] * xb[index]
                        if bool(ax < min_v):
                            min_v = ax
                            imin = index
                    xval[imin] = -xval[imin]
                    sign ^= 1 << (imin - 8 * k)
                block_signs[k] = sign & 0x7F

            if bool(torch.max(xval) < GROUP_MAX_EPS):
                continue

            scale, L = _make_qp_quants_torch(xval, 4, weight)
            L = L.to(torch.int8)
            eff_max = scale * 3.0
            if bool(eff_max <= 0):
                continue

            best = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
            for is_ in range(-6, 7):
                id_ = (5.0 + float(is_) * 0.1) / eff_max
                this_scale = 1.0 / id_
                Laux = torch.zeros(32, device=blocks.device, dtype=torch.int8)
                for k in range(4):
                    u = 0
                    for j in range(8):
                        index = 8 * k + j
                        l = int(
                            _nearest_int((0.5 * (id_ * xval[index] - 1.0)).reshape(1))[
                                0
                            ].item()
                        )
                        l = max(0, min(2, l))
                        Laux[index] = l
                        u |= l << (2 * j)
                    if int(kmap[u].item()) < 0:
                        _, Laux[8 * k : 8 * k + 8] = _best_lattice_neighbour_torch(
                            neighbours[u],
                            grid_l,
                            xval[8 * k : 8 * k + 8],
                            waux[8 * k : 8 * k + 8],
                            this_scale,
                        )

                sumqx = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
                sumq2 = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
                for j in range(32):
                    q = 2.0 * float(int(Laux[j].item())) + 1.0
                    w = weight[j]
                    sumqx = sumqx + w * xval[j] * q
                    sumq2 = sumq2 + w * q * q
                if bool((sumq2 > 0) & (sumqx * sumqx > best * sumq2)):
                    scale = sumqx / sumq2
                    best = scale * sumqx
                    L[:] = Laux

            if bool(scale > 0):
                id_ = 1.0 / scale
                for k in range(4):
                    u = 0
                    for j in range(8):
                        index = 8 * k + j
                        l = int(
                            _nearest_int((0.5 * (id_ * xval[index] - 1.0)).reshape(1))[
                                0
                            ].item()
                        )
                        l = max(0, min(2, l))
                        L[index] = l
                        u |= l << (2 * j)
                    grid_index = int(kmap[u].item())
                    if grid_index < 0:
                        _, L[8 * k : 8 * k + 8] = _best_lattice_neighbour_torch(
                            neighbours[u],
                            grid_l,
                            xval[8 * k : 8 * k + 8],
                            waux[8 * k : 8 * k + 8],
                            scale,
                        )
                    else:
                        L[8 * k : 8 * k + 8] = grid_l[grid_index].to(torch.int8)

                sumqx = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
                sumq2 = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
                for j in range(32):
                    q = 2.0 * float(int(L[j].item())) + 1.0
                    w = weight[j]
                    sumqx = sumqx + w * xval[j] * q
                    sumq2 = sumq2 + w * q * q
                if bool(sumq2 > 0):
                    scale = sumqx / sumq2

            if bool(scale < 0):
                scale = -scale
                block_signs = (~block_signs).to(torch.uint8)

            for k in range(4):
                u = 0
                for j in range(8):
                    u |= int(L[8 * k + j].item()) << (2 * j)
                q2[2 * ib] |= int(kmap[u].item()) << (8 * k)
                q2[2 * ib + 1] |= int(block_signs[k].item()) << (7 * k)

            scales[ib] = scale
            if bool(scale > max_scale):
                max_scale = scale

        if bool(max_scale == 0):
            continue

        d = max_scale / 31.0
        out[i, :2] = d.reshape(1).to(torch.float16).contiguous().view(torch.uint8)
        id_ = 1.0 / d
        for ib in range(QK_K // 32):
            l = int(_nearest_int((0.5 * (id_ * scales[ib] - 1.0)).reshape(1))[0].item())
            l = max(0, min(15, l))
            q2[2 * ib + 1] |= l << 28
        out[i, 2:] = q2.to(torch.uint32).contiguous().view(torch.uint8)

    return out


def dequantize_blocks_IQ2_XXS(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)

    qs_words = to_uint32_int64(qs.reshape(-1, 4)).reshape((n_blocks, -1, 2))
    q0 = qs_words[..., 0]
    q1 = qs_words[..., 1]

    db = d * (0.5 + (q1 >> 28).to(d.dtype)) * 0.25
    db = db.reshape((n_blocks, -1, 1, 1))

    signs = q1.reshape((n_blocks, -1, 1)) >> torch.tensor(
        [0, 7, 14, 21], device=blocks.device, dtype=torch.int64
    ).reshape((1, 1, 4))
    signs = (signs & 0x7F).to(torch.int64)
    ksigns = _ksigns_tensor(blocks.device)
    signs = ksigns[signs].reshape((n_blocks, -1, 4, 1)) >> torch.arange(
        8, device=blocks.device, dtype=torch.uint8
    ).reshape((1, 1, 1, 8))
    signs = signs & 0x01
    signs = torch.where(signs == 0, 1.0, -1.0).to(d.dtype)

    grid_idx = (
        q0.reshape((n_blocks, -1, 1))
        >> torch.tensor(
            [0, 8, 16, 24], device=blocks.device, dtype=torch.int64
        ).reshape((1, 1, 4))
    ) & 0xFF
    grid = _grid_tensor(IQ2_XXS, blocks.device).to(d.dtype)
    grid = grid[grid_idx.to(torch.int64)].reshape((n_blocks, -1, 4, 8))

    return (db * grid * signs).reshape((n_blocks, -1))


def quantize_blocks_IQ2_XS(blocks, block_size, type_size):
    n_blocks = blocks.shape[0]
    out = torch.zeros((n_blocks, type_size), device=blocks.device, dtype=torch.uint8)
    kmap, neighbours, grid_l = _iq2_xs_lookup_torch(blocks.device)
    quant_weights = torch.sum(blocks * blocks, dim=0, dtype=torch.float32)

    for i, x in enumerate(blocks):
        q2 = torch.zeros(2 * (QK_K // 16), device=blocks.device, dtype=torch.int64)
        scales = torch.zeros(QK_K // 16, device=blocks.device, dtype=torch.float32)
        max_scale = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
        sigma2 = torch.sum(x * x, dtype=torch.float32) / float(QK_K)

        for ib in range(QK_K // 16):
            xb = x[16 * ib : 16 * ib + 16]
            qw = quant_weights[16 * ib : 16 * ib + 16]
            weight = qw * torch.sqrt(sigma2 + xb * xb)
            waux = torch.sqrt(weight)
            xval = torch.empty(16, device=blocks.device, dtype=torch.float32)
            block_signs = torch.zeros(2, device=blocks.device, dtype=torch.uint8)

            for k in range(2):
                nflip = 0
                sign = 0
                for j in range(8):
                    index = 8 * k + j
                    if bool(xb[index] >= 0):
                        xval[index] = xb[index]
                    else:
                        xval[index] = -xb[index]
                        nflip += 1
                        sign |= 1 << j
                if nflip % 2:
                    imin = 8 * k
                    min_v = weight[imin] * xb[imin] * xb[imin]
                    for j in range(1, 8):
                        index = 8 * k + j
                        ax = weight[index] * xb[index] * xb[index]
                        if bool(ax < min_v):
                            min_v = ax
                            imin = index
                    xval[imin] = -xval[imin]
                    sign ^= 1 << (imin - 8 * k)
                block_signs[k] = sign & 0x7F

            L = torch.zeros(16, device=blocks.device, dtype=torch.int8)
            max_v = torch.max(xval)
            if bool(max_v < GROUP_MAX_EPS):
                continue

            best = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
            scale = max_v / 5.0
            is_on_grid = torch.ones(2, device=blocks.device, dtype=torch.bool)
            for is_ in range(-9, 10):
                id_ = (5.0 + float(is_) * 0.1) / max_v
                this_scale = 1.0 / id_
                Laux = torch.zeros(16, device=blocks.device, dtype=torch.int8)
                is_on_grid_aux = torch.ones(2, device=blocks.device, dtype=torch.bool)
                for k in range(2):
                    u = 0
                    for j in range(8):
                        index = 8 * k + j
                        l = int(
                            _nearest_int((0.5 * (id_ * xval[index] - 1.0)).reshape(1))[
                                0
                            ].item()
                        )
                        l = max(0, min(2, l))
                        Laux[index] = l
                        u |= l << (2 * j)
                    if int(kmap[u].item()) < 0:
                        is_on_grid_aux[k] = False
                        _, Laux[8 * k : 8 * k + 8] = _best_lattice_neighbour_torch(
                            neighbours[u],
                            grid_l,
                            xval[8 * k : 8 * k + 8],
                            waux[8 * k : 8 * k + 8],
                            this_scale,
                        )

                sumqx = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
                sumq2 = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
                for j in range(16):
                    q = 2.0 * float(int(Laux[j].item())) + 1.0
                    w = weight[j]
                    sumqx = sumqx + w * xval[j] * q
                    sumq2 = sumq2 + w * q * q
                if bool((sumq2 > 0) & (sumqx * sumqx > best * sumq2)):
                    scale = sumqx / sumq2
                    best = scale * sumqx
                    L[:] = Laux
                    is_on_grid[:] = is_on_grid_aux

            if bool((~is_on_grid).any() & (scale > 0)):
                id_ = 1.0 / scale
                for k in range(2):
                    if bool(is_on_grid[k]):
                        continue
                    u = 0
                    for j in range(8):
                        index = 8 * k + j
                        l = int(
                            _nearest_int((0.5 * (id_ * xval[index] - 1.0)).reshape(1))[
                                0
                            ].item()
                        )
                        l = max(0, min(2, l))
                        L[index] = l
                        u |= l << (2 * j)
                    if int(kmap[u].item()) < 0:
                        _, L[8 * k : 8 * k + 8] = _best_lattice_neighbour_torch(
                            neighbours[u],
                            grid_l,
                            xval[8 * k : 8 * k + 8],
                            waux[8 * k : 8 * k + 8],
                            scale,
                        )

                sumqx = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
                sumq2 = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
                for j in range(16):
                    q = 2.0 * float(int(L[j].item())) + 1.0
                    w = weight[j]
                    sumqx = sumqx + w * xval[j] * q
                    sumq2 = sumq2 + w * q * q
                if bool(sumq2 > 0):
                    scale = sumqx / sumq2

            if bool(scale < 0):
                scale = -scale
                block_signs = (~block_signs).to(torch.uint8)

            for k in range(2):
                u = 0
                for j in range(8):
                    u |= int(L[8 * k + j].item()) << (2 * j)
                q2[2 * ib + k] = int(kmap[u].item()) | (int(block_signs[k].item()) << 9)

            scales[ib] = scale
            if bool(scale > max_scale):
                max_scale = scale

        if bool(max_scale == 0):
            continue

        d = max_scale / 31.0
        out[i, :2] = d.reshape(1).to(torch.float16).contiguous().view(torch.uint8)
        id_ = 1.0 / d
        packed_scales = out[i, 2 + QK_K // 4 :]
        for ib in range(QK_K // 16):
            l = int(_nearest_int((0.5 * (id_ * scales[ib] - 1.0)).reshape(1))[0].item())
            l = max(0, min(15, l))
            if ib % 2 == 0:
                packed_scales[ib // 2] = l
            else:
                packed_scales[ib // 2] |= l << 4
        out[i, 2 : 2 + QK_K // 4] = q2.to(torch.uint16).contiguous().view(torch.uint8)

    return out


def dequantize_blocks_IQ2_XS(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, rest = split_block_dims(blocks, 2)
    qs, scales = split_block_dims(rest, 2 * QK_K // 8)

    d = d.view(torch.float16).to(dtype)
    qs = to_uint16(qs.reshape(-1, 2)).reshape((n_blocks, -1))

    scales = scales.reshape((n_blocks, -1, 1)) >> torch.tensor(
        [0, 4], device=blocks.device, dtype=torch.uint8
    ).reshape((1, 1, 2))
    scales = (scales & 0x0F).reshape((n_blocks, -1))
    db = d * (0.5 + scales.to(d.dtype)) * 0.25
    db = db.reshape((n_blocks, -1, 1, 1))

    signs = _ksigns_tensor(blocks.device)[(qs >> 9).to(torch.int64)]
    signs = signs.reshape((n_blocks, -1, 1)) >> torch.arange(
        8, device=blocks.device, dtype=torch.uint8
    ).reshape((1, 1, 8))
    signs = signs & 0x01
    signs = torch.where(signs == 0, 1.0, -1.0).to(d.dtype)
    signs = signs.reshape((n_blocks, -1, 2, 8))

    grid = _grid_tensor(IQ2_XS, blocks.device).to(d.dtype)
    grid = grid[(qs & 511).to(torch.int64)].reshape((n_blocks, -1, 2, 8))

    return (db * grid * signs).reshape((n_blocks, -1))


def quantize_blocks_IQ2_S(blocks, block_size, type_size):
    n_blocks = blocks.shape[0]
    out = torch.zeros((n_blocks, type_size), device=blocks.device, dtype=torch.uint8)
    kmap, neighbours, grid_l = _iq2_s_lookup_torch(blocks.device)

    for i, x in enumerate(blocks):
        qs = out[i, 2 : 2 + QK_K // 8]
        signs = out[i, 2 + QK_K // 8 : 2 + QK_K // 4]
        qh = out[i, 2 + QK_K // 4 : 2 + QK_K // 4 + QK_K // 32]
        packed_scales = out[i, 2 + QK_K // 4 + QK_K // 32 :]
        scales = torch.zeros(QK_K // 16, device=blocks.device, dtype=torch.float32)
        max_scale = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
        sigma2 = 2.0 * torch.sum(x * x, dtype=torch.float32) / float(QK_K)

        for ib in range(QK_K // 16):
            xb = x[16 * ib : 16 * ib + 16]
            weight = 0.25 * sigma2 + xb * xb
            waux = torch.sqrt(weight)
            xval = torch.empty(16, device=blocks.device, dtype=torch.float32)
            block_signs = torch.zeros(2, device=blocks.device, dtype=torch.uint8)

            for k in range(2):
                sign = 0
                for j in range(8):
                    index = 8 * k + j
                    if bool(xb[index] >= 0):
                        xval[index] = xb[index]
                    else:
                        xval[index] = -xb[index]
                        sign |= 1 << j
                block_signs[k] = sign

            L = torch.zeros(16, device=blocks.device, dtype=torch.int8)
            max_v = torch.max(xval)
            if bool(max_v < GROUP_MAX_EPS_IQ2_S):
                continue

            best = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
            scale = max_v / 5.0
            is_on_grid = torch.ones(2, device=blocks.device, dtype=torch.bool)
            for is_ in range(-9, 10):
                id_ = (5.0 + float(is_) * 0.1) / max_v
                this_scale = 1.0 / id_
                Laux = torch.zeros(16, device=blocks.device, dtype=torch.int8)
                is_on_grid_aux = torch.ones(2, device=blocks.device, dtype=torch.bool)
                for k in range(2):
                    u = 0
                    for j in range(8):
                        index = 8 * k + j
                        l = int(
                            _nearest_int((0.5 * (id_ * xval[index] - 1.0)).reshape(1))[
                                0
                            ].item()
                        )
                        l = max(0, min(2, l))
                        Laux[index] = l
                        u |= l << (2 * j)
                    if int(kmap[u].item()) < 0:
                        is_on_grid_aux[k] = False
                        _, Laux[8 * k : 8 * k + 8] = _best_lattice_neighbour_torch(
                            neighbours[u],
                            grid_l,
                            xval[8 * k : 8 * k + 8],
                            waux[8 * k : 8 * k + 8],
                            this_scale,
                        )

                sumqx = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
                sumq2 = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
                for j in range(16):
                    q = 2.0 * float(int(Laux[j].item())) + 1.0
                    w = weight[j]
                    sumqx = sumqx + w * xval[j] * q
                    sumq2 = sumq2 + w * q * q
                if bool((sumq2 > 0) & (sumqx * sumqx > best * sumq2)):
                    scale = sumqx / sumq2
                    best = scale * sumqx
                    L[:] = Laux
                    is_on_grid[:] = is_on_grid_aux

            if bool((~is_on_grid).any() & (scale > 0)):
                id_ = 1.0 / scale
                for k in range(2):
                    if bool(is_on_grid[k]):
                        continue
                    u = 0
                    for j in range(8):
                        index = 8 * k + j
                        l = int(
                            _nearest_int((0.5 * (id_ * xval[index] - 1.0)).reshape(1))[
                                0
                            ].item()
                        )
                        l = max(0, min(2, l))
                        L[index] = l
                        u |= l << (2 * j)
                    if int(kmap[u].item()) < 0:
                        _, L[8 * k : 8 * k + 8] = _best_lattice_neighbour_torch(
                            neighbours[u],
                            grid_l,
                            xval[8 * k : 8 * k + 8],
                            waux[8 * k : 8 * k + 8],
                            scale,
                            tie_eps=1.0e-6,
                        )

                sumqx = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
                sumq2 = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
                for j in range(16):
                    q = 2.0 * float(int(L[j].item())) + 1.0
                    w = weight[j]
                    sumqx = sumqx + w * xval[j] * q
                    sumq2 = sumq2 + w * q * q
                if bool(sumq2 > 0):
                    scale = sumqx / sumq2

            if bool(scale < 0):
                scale = -scale
                block_signs = (~block_signs).to(torch.uint8)

            for k in range(2):
                u = 0
                for j in range(8):
                    u |= int(L[8 * k + j].item()) << (2 * j)
                grid_index = int(kmap[u].item())
                i8 = 2 * ib + k
                qs[i8] = grid_index & 0xFF
                qh[i8 // 4] |= ((grid_index >> 8) & 0x03) << (2 * (i8 % 4))
                signs[i8] = block_signs[k]

            scales[ib] = scale
            if bool(scale > max_scale):
                max_scale = scale

        if bool(max_scale == 0):
            continue

        d = max_scale / 31.0
        out[i, :2] = (
            (d * 0.9875).reshape(1).to(torch.float16).contiguous().view(torch.uint8)
        )
        id_ = 1.0 / d
        for ib in range(QK_K // 16):
            l = int(_nearest_int((0.5 * (id_ * scales[ib] - 1.0)).reshape(1))[0].item())
            l = max(0, min(15, l))
            if ib % 2 == 0:
                packed_scales[ib // 2] = l
            else:
                packed_scales[ib // 2] |= l << 4

    return out


def dequantize_blocks_IQ2_S(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, rest = split_block_dims(blocks, 2)
    qs, rest = split_block_dims(rest, QK_K // 8)
    signs, rest = split_block_dims(rest, QK_K // 8)
    qh, scales = split_block_dims(rest, QK_K // 32)

    d = d.view(torch.float16).to(dtype)

    scales = scales.reshape((n_blocks, -1, 1)) >> torch.tensor(
        [0, 4], device=blocks.device, dtype=torch.uint8
    ).reshape((1, 1, 2))
    scales = (scales & 0x0F).reshape((n_blocks, -1))
    db = d * (0.5 + scales.to(d.dtype)) * 0.25
    db = db.reshape((n_blocks, -1, 1, 1))

    signs = signs.reshape((n_blocks, -1, 1)) >> torch.arange(
        8, device=blocks.device, dtype=torch.uint8
    ).reshape((1, 1, 8))
    signs = signs & 0x01
    signs = torch.where(signs == 0, 1.0, -1.0).to(d.dtype)
    signs = signs.reshape((n_blocks, -1, 2, 8))

    qh = qh.reshape((n_blocks, -1, 1)) >> torch.tensor(
        [0, 2, 4, 6], device=blocks.device, dtype=torch.uint8
    ).reshape((1, 1, 4))
    qs = qs.to(torch.int64) | ((qh & 0x03).to(torch.int64) << 8).reshape((n_blocks, -1))

    grid = _grid_tensor(IQ2_S, blocks.device).to(d.dtype)
    grid = grid[qs].reshape((n_blocks, -1, 2, 8))

    return (db * grid * signs).reshape((n_blocks, -1))


def quantize_blocks_IQ1_S(blocks, block_size, type_size):
    n_blocks = blocks.shape[0]
    out = torch.zeros((n_blocks, type_size), device=blocks.device, dtype=torch.uint8)
    kmap, neighbours, grid_l = _iq1_s_lookup_torch(blocks.device)
    quant_weights = torch.sum(blocks * blocks, dim=0, dtype=torch.float32)
    x_p = torch.tensor(
        [-1.0 + 0.125, 0.125, 1.0 + 0.125], device=blocks.device, dtype=torch.float32
    )
    x_m = torch.tensor(
        [-1.0 - 0.125, -0.125, 1.0 - 0.125], device=blocks.device, dtype=torch.float32
    )

    for i, x in enumerate(blocks):
        qs = out[i, 2 : 2 + QK_K // 8]
        qh = torch.zeros(QK_K // 32, device=blocks.device, dtype=torch.int32)
        scales = torch.zeros(QK_K // 32, device=blocks.device, dtype=torch.float32)
        shifts = torch.ones(QK_K // 32, device=blocks.device, dtype=torch.int8)
        max_scale = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
        sumx2 = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
        for value in x:
            sumx2 = sumx2 + value * value
        sigma2 = 2.0 * sumx2 / float(QK_K)

        for ib in range(QK_K // 32):
            xb = x[32 * ib : 32 * ib + 32]
            qw = quant_weights[32 * ib : 32 * ib + 32]
            weight = torch.empty(32, device=blocks.device, dtype=torch.float32)
            for j in range(32):
                weight[j] = qw[j] * torch.sqrt(sigma2 + xb[j] * xb[j])

            max_v = torch.max(torch.abs(xb))
            L = torch.ones(32, device=blocks.device, dtype=torch.int8)
            if bool(max_v < GROUP_MAX_EPS_IQ1_S):
                scales[ib] = 0.0
                shifts[ib] = 1
                continue

            order = torch.argsort(xb, stable=True)
            sumx = torch.zeros(33, device=blocks.device, dtype=torch.float32)
            sumw = torch.zeros(33, device=blocks.device, dtype=torch.float32)
            for j in range(32):
                index = int(order[j].item())
                sumx[j + 1] = sumx[j] + weight[index] * xb[index]
                sumw[j + 1] = sumw[j] + weight[index]

            best_score = torch.tensor(
                -torch.finfo(torch.float32).max,
                device=blocks.device,
                dtype=torch.float32,
            )
            scale = max_v
            besti1 = -1
            besti2 = -1
            best_shift = 0
            for i1 in range(33):
                for i2 in range(i1, 33):
                    for shift, values in ((1, x_p), (-1, x_m)):
                        sumqx = (sumx[i1] - sumx[0]) * values[0]
                        sumqx = sumqx + (sumx[i2] - sumx[i1]) * values[1]
                        sumqx = sumqx + (sumx[32] - sumx[i2]) * values[2]
                        w0 = sumw[i1] - sumw[0]
                        w1 = sumw[i2] - sumw[i1]
                        w2 = sumw[32] - sumw[i2]
                        q20 = (w0 * values[0]) * values[0]
                        q21 = (w1 * values[1]) * values[1]
                        q22 = (w2 * values[2]) * values[2]
                        sumq2 = (q20 + q21) + q22
                        lhs = sumqx * sumqx
                        rhs = torch.tensor(
                            float("-inf"), device=blocks.device, dtype=torch.float32
                        )
                        if bool(best_score >= 0):
                            rhs = best_score * sumq2
                        if bool((sumq2 > 0) & (lhs > rhs)):
                            scale = sumqx / sumq2
                            best_score = scale * sumqx
                            besti1 = i1
                            besti2 = i2
                            best_shift = shift

            if besti1 < 0 or besti2 < 0 or best_shift == 0:
                scales[ib] = 0.0
                shifts[ib] = 1
                continue

            for j in range(besti1):
                L[int(order[j].item())] = 0
            for j in range(besti1, besti2):
                L[int(order[j].item())] = 1
            for j in range(besti2, 32):
                L[int(order[j].item())] = 2

            if bool(scale < 0):
                for j in range(32):
                    L[j] = 2 - int(L[j].item())
                scale = -scale
                best_shift = -best_shift

            all_on_grid = True
            values = x_p if best_shift == 1 else x_m
            grid_indices = torch.zeros(4, device=blocks.device, dtype=torch.int32)
            for k in range(4):
                u = 0
                for j in range(8):
                    u |= int(L[8 * k + j].item()) << (2 * j)
                grid_index = int(kmap[u].item())
                if grid_index < 0:
                    all_on_grid = False
                    grid_index, L[8 * k : 8 * k + 8] = _best_iq1_neighbour_torch(
                        neighbours[u],
                        grid_l,
                        xb[8 * k : 8 * k + 8],
                        weight[8 * k : 8 * k + 8],
                        scale,
                        values,
                    )
                grid_indices[k] = grid_index

            if not all_on_grid:
                sumqx = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
                sumq2 = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
                for k in range(4):
                    levels = grid_l[int(grid_indices[k].item())].to(torch.int64)
                    for j in range(8):
                        index = 8 * k + j
                        w = weight[index]
                        q = values[int(levels[j].item())]
                        sumqx = sumqx + w * q * xb[index]
                        sumq2 = sumq2 + w * q * q
                if bool((sumqx > 0) & (sumq2 > 0)):
                    scale = sumqx / sumq2

            h = 0
            for k in range(4):
                grid_index = int(grid_indices[k].item())
                qs[4 * ib + k] = grid_index & 0xFF
                h |= (grid_index >> 8) << (3 * k)
            qh[ib] = h
            scales[ib] = scale
            shifts[ib] = best_shift
            if bool(scale > max_scale):
                max_scale = scale

        if bool(max_scale == 0):
            continue

        d = max_scale / 15.0
        out[i, :2] = (
            (d * 1.125).reshape(1).to(torch.float16).contiguous().view(torch.uint8)
        )
        id_ = 1.0 / d
        for ib in range(QK_K // 32):
            l = int(_nearest_int((0.5 * (id_ * scales[ib] - 1.0)).reshape(1))[0].item())
            l = max(0, min(7, l))
            if int(shifts[ib].item()) == -1:
                l |= 8
            qh[ib] |= l << 12

        qh_bytes = torch.empty(QK_K // 16, device=blocks.device, dtype=torch.uint8)
        qh_bytes[0::2] = (qh & 0xFF).to(torch.uint8)
        qh_bytes[1::2] = ((qh >> 8) & 0xFF).to(torch.uint8)
        out[i, 2 + QK_K // 8 :] = qh_bytes

    return out


def dequantize_blocks_IQ1_S(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, rest = split_block_dims(blocks, 2)
    qs, qh = split_block_dims(rest, QK_K // 8)

    d = d.view(torch.float16).to(dtype)
    qh = to_uint16(qh.reshape(-1, 2)).reshape((n_blocks, -1))

    dl = d * (2 * ((qh >> 12) & 7).to(d.dtype) + 1)
    dl = dl.reshape((n_blocks, -1, 1, 1))
    delta = torch.where((qh & 0x8000) == 0, 0.125, -0.125).to(d.dtype)
    delta = delta.reshape((n_blocks, -1, 1, 1))

    qh = qh.reshape((n_blocks, -1, 1)) >> torch.tensor(
        [0, 3, 6, 9], device=blocks.device, dtype=torch.int64
    ).reshape((1, 1, 4))
    qs = qs.to(torch.int64) | ((qh & 7) << 8).reshape((n_blocks, -1))

    grid = _grid_tensor(IQ1_S, blocks.device).to(d.dtype)
    grid = grid[qs].reshape((n_blocks, -1, 4, 8))

    return (dl * (grid + delta)).reshape((n_blocks, -1))


def quantize_blocks_IQ1_M(blocks, block_size, type_size):
    n_blocks = blocks.shape[0]
    out = torch.zeros((n_blocks, type_size), device=blocks.device, dtype=torch.uint8)
    kmap, neighbours, grid_l = _iq1_s_lookup_torch(blocks.device)
    x_p = torch.tensor(
        [-1.0 + 0.125, 0.125, 1.0 + 0.125], device=blocks.device, dtype=torch.float32
    )
    x_m = torch.tensor(
        [-1.0 - 0.125, -0.125, 1.0 - 0.125], device=blocks.device, dtype=torch.float32
    )
    masks = (0x00, 0x80, 0x08, 0x88)

    for i, x in enumerate(blocks):
        qs = out[i, : QK_K // 8]
        qh = out[i, QK_K // 8 : QK_K // 8 + QK_K // 16]
        scale_words = torch.zeros(QK_K // 64, device=blocks.device, dtype=torch.int32)
        scales = torch.zeros(QK_K // 16, device=blocks.device, dtype=torch.float32)
        shifts = torch.zeros(QK_K // 16, device=blocks.device, dtype=torch.int8)
        max_scale = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
        sumx2 = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
        for value in x:
            sumx2 = sumx2 + value * value
        sigma2 = 2.0 * sumx2 / float(QK_K)

        for ib in range(QK_K // 16):
            xb = x[16 * ib : 16 * ib + 16]
            weight = xb * xb
            max_v = torch.max(torch.abs(xb))
            L = torch.ones(16, device=blocks.device, dtype=torch.int8)
            if bool(max_v < GROUP_MAX_EPS_IQ1_M):
                scales[ib] = 0.0
                shifts[ib] = 0
                continue

            order = torch.argsort(xb, stable=True)
            best_score = torch.tensor(
                -torch.finfo(torch.float32).max,
                device=blocks.device,
                dtype=torch.float32,
            )
            scale = max_v
            besti1 = -1
            besti2 = -1
            best_k = -1
            for i1 in range(17):
                for i2 in range(i1, 17):
                    sumqx = torch.zeros(4, device=blocks.device, dtype=torch.float32)
                    sumq2 = torch.zeros(4, device=blocks.device, dtype=torch.float32)
                    for j in range(16):
                        index = int(order[j].item())
                        if j < i1:
                            level = 0
                        elif j < i2:
                            level = 1
                        else:
                            level = 2

                        value_sets = (
                            x_p,
                            x_p if index < 8 else x_m,
                            x_m if index < 8 else x_p,
                            x_m,
                        )
                        for k, values in enumerate(value_sets):
                            q = values[level]
                            w = weight[index]
                            sumqx[k] = sumqx[k] + (w * q) * xb[index]
                            sumq2[k] = sumq2[k] + (w * q) * q

                    for k in range(4):
                        lhs = sumqx[k] * sumqx[k]
                        rhs = torch.tensor(
                            float("-inf"), device=blocks.device, dtype=torch.float32
                        )
                        if bool(best_score >= 0):
                            rhs = best_score * sumq2[k]
                        if bool((sumq2[k] > 0) & (lhs > rhs)):
                            scale = sumqx[k] / sumq2[k]
                            best_score = scale * sumqx[k]
                            besti1 = i1
                            besti2 = i2
                            best_k = k

            if besti1 < 0 or besti2 < 0 or best_k < 0:
                scales[ib] = 0.0
                shifts[ib] = 0
                continue

            for j in range(besti1):
                L[int(order[j].item())] = 0
            for j in range(besti1, besti2):
                L[int(order[j].item())] = 1
            for j in range(besti2, 16):
                L[int(order[j].item())] = 2

            if bool(scale < 0):
                for j in range(16):
                    L[j] = 2 - int(L[j].item())
                scale = -scale
                best_k = (3, 2, 1, 0)[best_k]

            all_on_grid = True
            grid_indices = torch.zeros(2, device=blocks.device, dtype=torch.int32)
            for k in range(2):
                values = x_p if (best_k < 2 if k == 0 else best_k % 2 == 0) else x_m
                u = 0
                for j in range(8):
                    u |= int(L[8 * k + j].item()) << (2 * j)
                grid_index = int(kmap[u].item())
                if grid_index < 0:
                    all_on_grid = False
                    grid_index, L[8 * k : 8 * k + 8] = _best_iq1_neighbour_torch(
                        neighbours[u],
                        grid_l,
                        xb[8 * k : 8 * k + 8],
                        weight[8 * k : 8 * k + 8],
                        scale,
                        values,
                    )
                grid_indices[k] = grid_index

            if not all_on_grid:
                sumqx_f = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
                sumq2_f = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
                for k in range(2):
                    values = x_p if (best_k < 2 if k == 0 else best_k % 2 == 0) else x_m
                    levels = grid_l[int(grid_indices[k].item())].to(torch.int64)
                    for j in range(8):
                        index = 8 * k + j
                        w = weight[index]
                        q = values[int(levels[j].item())]
                        sumqx_f = sumqx_f + (w * q) * xb[index]
                        sumq2_f = sumq2_f + (w * q) * q
                if bool((sumqx_f > 0) & (sumq2_f > 0)):
                    scale = sumqx_f / sumq2_f

            qs[2 * ib] = int(grid_indices[0].item()) & 0xFF
            qs[2 * ib + 1] = int(grid_indices[1].item()) & 0xFF
            qh[ib] = (
                (int(grid_indices[0].item()) >> 8)
                | ((int(grid_indices[1].item()) >> 8) << 4)
            ) & 0xFF
            scales[ib] = scale
            shifts[ib] = best_k
            if bool(scale > max_scale):
                max_scale = scale

        if bool(max_scale == 0):
            continue

        d = max_scale / 15.0
        id_ = 1.0 / d
        sumqx_f = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
        sumq2_f = torch.tensor(0.0, device=blocks.device, dtype=torch.float32)
        for ib in range(QK_K // 16):
            l = int(_nearest_int((0.5 * (id_ * scales[ib] - 1.0)).reshape(1))[0].item())
            l = max(0, min(7, l))
            scale_words[ib // 4] |= l << (3 * (ib % 4))
            qh[ib] |= masks[int(shifts[ib].item())]
            xb = x[16 * ib : 16 * ib + 16]
            weight = xb * xb
            for k in range(2):
                values = (
                    x_p
                    if (
                        int(shifts[ib].item()) < 2
                        if k == 0
                        else int(shifts[ib].item()) % 2 == 0
                    )
                    else x_m
                )
                grid_index = int(qs[2 * ib + k].item()) | (
                    ((int(qh[ib].item()) >> (4 * k)) & 0x07) << 8
                )
                levels = grid_l[grid_index].to(torch.int64)
                for j in range(8):
                    index = 8 * k + j
                    w = weight[index]
                    q = values[int(levels[j].item())] * float(2 * l + 1)
                    sumqx_f = sumqx_f + (w * q) * xb[index]
                    sumq2_f = sumq2_f + (w * q) * q

        if bool(sumq2_f > 0):
            d = sumqx_f / sumq2_f
        scale_bytes = (
            (d * 1.1125).reshape(1).to(torch.float16).contiguous().view(torch.uint8)
        )
        scale_bits = int(scale_bytes[0].item()) | (int(scale_bytes[1].item()) << 8)
        scale_words[0] |= (scale_bits & 0x000F) << 12
        scale_words[1] |= (scale_bits & 0x00F0) << 8
        scale_words[2] |= (scale_bits & 0x0F00) << 4
        scale_words[3] |= scale_bits & 0xF000

        scale_out = out[i, QK_K // 8 + QK_K // 16 :]
        scale_out[0::2] = (scale_words & 0xFF).to(torch.uint8)
        scale_out[1::2] = ((scale_words >> 8) & 0xFF).to(torch.uint8)

    return out


def dequantize_blocks_IQ1_M(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    qs, rest = split_block_dims(blocks, QK_K // 8)
    qh, scales = split_block_dims(rest, QK_K // 16)

    scales = to_uint16(scales.reshape(-1, 2)).reshape((n_blocks, -1))
    d_bits = (scales.reshape((n_blocks, 4)) & 0xF000) >> torch.tensor(
        [12, 8, 4, 0], device=blocks.device, dtype=torch.int32
    ).reshape((1, 4))
    d = (d_bits[:, 0] | d_bits[:, 1] | d_bits[:, 2] | d_bits[:, 3]).to(torch.int16)
    d = d.view(torch.float16).to(dtype).reshape((n_blocks, 1))

    scales = scales.reshape((n_blocks, -1, 1)) >> torch.tensor(
        [0, 3, 6, 9], device=blocks.device, dtype=torch.int32
    ).reshape((1, 1, 4))
    scales = (scales & 0x07).reshape((n_blocks, -1))
    dl = d * (2 * scales.to(d.dtype) + 1)
    dl = dl.reshape((n_blocks, -1, 2, 1, 1))

    qh = qh.reshape((n_blocks, -1, 1)) >> torch.tensor(
        [0, 4], device=blocks.device, dtype=torch.uint8
    ).reshape((1, 1, 2))
    qs = qs.to(torch.int64) | ((qh & 0x07).to(torch.int64) << 8).reshape((n_blocks, -1))

    delta = torch.where((qh & 0x08) == 0, 0.125, -0.125).to(d.dtype)
    delta = delta.reshape((n_blocks, -1, 2, 2, 1))

    grid = _grid_tensor(IQ1_M, blocks.device).to(d.dtype)
    grid = grid[qs].reshape((n_blocks, -1, 2, 2, 8))

    return (dl * (grid + delta)).reshape((n_blocks, -1))


def _best_index_int8(values, x):
    if x <= values[0]:
        return 0
    if x >= values[-1]:
        return len(values) - 1
    ml = 0
    mu = len(values) - 1
    while mu - ml > 1:
        mav = (ml + mu) // 2
        if x < values[mav]:
            mu = mav
        else:
            ml = mav
    return mu - 1 if x - values[mu - 1] < values[mu] - x else mu


def _quantize_iq4_nl_impl(x, super_block_size, block_size, values, ntry):
    device = x.device
    l_vals = torch.zeros(super_block_size, device=device, dtype=torch.uint8)
    q4 = torch.zeros(super_block_size // 2, device=device, dtype=torch.uint8)
    scales = torch.zeros(
        super_block_size // block_size, device=device, dtype=torch.float32
    )

    max_scale = torch.tensor(0.0, device=device, dtype=torch.float32)
    amax_scale = torch.tensor(0.0, device=device, dtype=torch.float32)
    value_tensor = torch.tensor(values, device=device, dtype=torch.float32)

    for ib in range(super_block_size // block_size):
        start = ib * block_size
        xb = x[start : start + block_size]
        weight = xb * xb

        ax = torch.abs(xb)
        imax = int(torch.argmax(ax).item())
        amax = ax[imax]
        max_v = xb[imax]
        if bool(amax < GROUP_MAX_EPS):
            scales[ib] = 0
            continue

        d = (-max_v / float(values[0])) if ntry > 0 else (max_v / float(values[0]))
        id_ = 1.0 / d
        sumqx = torch.tensor(0.0, device=device, dtype=torch.float32)
        sumq2 = torch.tensor(0.0, device=device, dtype=torch.float32)
        for j in range(block_size):
            l = _best_index_int8(values, float((id_ * xb[j]).item()))
            l_vals[start + j] = l
            q = value_tensor[l]
            w = weight[j]
            sumqx = sumqx + w * q * xb[j]
            sumq2 = sumq2 + w * q * q
        d = (
            sumqx / sumq2
            if bool(sumq2 > 0)
            else torch.tensor(0.0, device=device, dtype=torch.float32)
        )
        best = d * sumqx

        for itry in range(-ntry, ntry + 1):
            id_ = (itry + values[0]) / max_v
            sumqx = torch.tensor(0.0, device=device, dtype=torch.float32)
            sumq2 = torch.tensor(0.0, device=device, dtype=torch.float32)
            for j in range(block_size):
                l = _best_index_int8(values, float((id_ * xb[j]).item()))
                q = value_tensor[l]
                w = weight[j]
                sumqx = sumqx + w * q * xb[j]
                sumq2 = sumq2 + w * q * q
            if bool((sumq2 > 0) & (sumqx * sumqx > best * sumq2)):
                d = sumqx / sumq2
                best = d * sumqx

        scales[ib] = d
        abs_d = torch.abs(d)
        if bool(abs_d > amax_scale):
            amax_scale = abs_d
            max_scale = d

    scales_h = 0
    scales_l = None
    if super_block_size // block_size > 1:
        nb = super_block_size // block_size
        scales_l = torch.zeros((nb + 1) // 2, device=device, dtype=torch.uint8)
        d = -max_scale / 32.0
        dh = d.reshape(1).to(torch.float16)[0]
        id_ = (
            1.0 / d
            if bool(d != 0)
            else torch.tensor(0.0, device=device, dtype=torch.float32)
        )
        for ib in range(nb):
            l = int(_nearest_int((id_ * scales[ib]).reshape(1))[0].item())
            l = max(-32, min(31, l))
            dl = d * float(l)
            idl = (
                1.0 / dl
                if bool(dl != 0)
                else torch.tensor(0.0, device=device, dtype=torch.float32)
            )
            start = ib * block_size
            xb = x[start : start + block_size]
            for j in range(block_size):
                l_vals[start + j] = _best_index_int8(
                    values, float((idl * xb[j]).item())
                )
            l += 32
            l_l = l & 0x0F
            l_h = l >> 4
            if ib % 2 == 0:
                scales_l[ib // 2] = l_l
            else:
                scales_l[ib // 2] |= l_l << 4
            scales_h |= l_h << (2 * (ib % 8))
    else:
        dh = scales[0].reshape(1).to(torch.float16)[0]
        if ntry > 0:
            id_ = (
                1.0 / dh.to(torch.float32)
                if bool(dh != 0)
                else torch.tensor(0.0, device=device, dtype=torch.float32)
            )
            for j in range(super_block_size):
                l_vals[j] = _best_index_int8(values, float((id_ * x[j]).item()))

    for i in range(super_block_size // 32):
        for j in range(16):
            q4[16 * i + j] = l_vals[32 * i + j] | (l_vals[32 * i + 16 + j] << 4)

    return dh, q4, scales_h, scales_l


def quantize_blocks_IQ4_NL(blocks, block_size, type_size):
    n_blocks = blocks.shape[0]
    out = torch.zeros((n_blocks, type_size), device=blocks.device, dtype=torch.uint8)
    values = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)
    for i, x in enumerate(blocks):
        d, qs, _, _ = _quantize_iq4_nl_impl(x, 32, 32, values, 7)
        out[i, :2] = d.reshape(1).contiguous().view(torch.uint8)
        out[i, 2:] = qs
    return out


def dequantize_blocks_IQ4_NL(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)

    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 1)).to(torch.int64)

    kvalues = KVALUES.to(qs.device).expand(*qs.shape[:-1], 16)
    qs = torch.gather(kvalues, dim=-1, index=qs).reshape((n_blocks, -1))
    del kvalues  # should still be view, but just to be safe

    return d * qs


def quantize_blocks_IQ4_XS(blocks, block_size, type_size):
    n_blocks = blocks.shape[0]
    out = torch.zeros((n_blocks, type_size), device=blocks.device, dtype=torch.uint8)
    values = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)
    for i, x in enumerate(blocks):
        d, qs, scales_h, scales_l = _quantize_iq4_nl_impl(x, QK_K, 32, values, 7)
        assert scales_l is not None
        out[i, :2] = d.reshape(1).contiguous().view(torch.uint8)
        out[i, 2:4] = torch.tensor(
            [scales_h], device=blocks.device, dtype=torch.uint16
        ).view(torch.uint8)
        out[i, 4 : 4 + QK_K // 64] = scales_l
        out[i, 4 + QK_K // 64 :] = qs
    return out


def dequantize_blocks_IQ4_XS(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, scales_h, scales_l, qs = split_block_dims(blocks, 2, 2, QK_K // 64)
    d = d.view(torch.float16).to(dtype)
    scales_h = to_uint16(scales_h)

    shift_a = torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 2)
    )
    shift_b = torch.tensor(
        [2 * i for i in range(QK_K // 32)], device=d.device, dtype=torch.uint8
    ).reshape((1, -1, 1))

    scales_l = scales_l.reshape((n_blocks, -1, 1)) >> shift_a.reshape((1, 1, 2))
    scales_h = scales_h.reshape((n_blocks, -1, 1)) >> shift_b.reshape((1, -1, 1))

    scales_l = scales_l.reshape((n_blocks, -1)) & 0x0F
    scales_h = scales_h.reshape((n_blocks, -1)).to(torch.uint8) & 0x03

    scales = (scales_l | (scales_h << 4)).to(torch.int8) - 32
    dl = (d * scales.to(dtype)).reshape((n_blocks, -1, 1))

    qs = qs.reshape((n_blocks, -1, 1, 16)) >> shift_a.reshape((1, 1, 2, 1))
    qs = qs.reshape((n_blocks, -1, 32, 1)) & 0x0F

    kvalues = KVALUES.to(qs.device).expand(*qs.shape[:-1], 16)
    qs = torch.gather(kvalues, dim=-1, index=qs.to(torch.int64)).reshape(
        (n_blocks, -1, 32)
    )
    del kvalues  # see IQ4_NL
    del shift_a
    del shift_b

    return (dl * qs).reshape((n_blocks, -1))


def dequantize_blocks_NVFP4(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]

    d_bytes, qs = split_block_dims(blocks, 4)

    x = d_bytes.to(torch.int32)
    exp = (x >> 3) & 0x0F
    man = (x & 0x07).to(torch.float32)
    two = torch.tensor(2.0, device=blocks.device, dtype=torch.float32)
    raw = torch.where(
        exp == 0,
        man * (2.0**-9),
        (1.0 + man / 8.0) * torch.pow(two, exp.to(torch.float32) - 7.0),
    )
    d = torch.where((x == 0) | (x == 0x7F), 0.0, raw * 0.5)
    d = d.to(dtype if dtype is not None else torch.float32).reshape((n_blocks, 4, 1))

    qs = qs.reshape((n_blocks, 4, 8))
    lo = (qs & 0x0F).to(torch.int64)
    hi = (qs >> 4).to(torch.int64)
    idx = torch.cat([lo, hi], dim=-1)

    kvalues = torch.tensor(
        [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
        device=blocks.device,
        dtype=d.dtype,
    )
    vals = kvalues[idx]

    return (d * vals).reshape((n_blocks, block_size))


def quantize_blocks_NVFP4(blocks, block_size, type_size):
    n_super = blocks.shape[0]

    blocks = blocks.reshape((n_super, 4, 16))
    d = torch.max(torch.abs(blocks), dim=-1).values / 6.0
    d_bytes = fp32_to_ue4m3(d)
    d_fp = ue4m3_to_fp32(d_bytes).reshape((n_super, 4, 1))

    kvalues = torch.tensor(
        [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
        device=blocks.device,
        dtype=torch.float32,
    ).reshape((1, 1, 1, 16))
    errs = torch.abs(
        d_fp.reshape((n_super, 4, 1, 1)) * kvalues - blocks.reshape((n_super, 4, 16, 1))
    )
    best = torch.argmin(errs, dim=-1).to(torch.uint8)

    lo = best[:, :, :8]
    hi = best[:, :, 8:] << 4
    qs = (lo | hi).reshape((n_super, 32))

    return torch.cat([d_bytes, qs], dim=-1)


quantize_functions = {
    GGMLQuantizationType.BF16: quantize_blocks_BF16,
    GGMLQuantizationType.Q1_0: quantize_blocks_Q1_0,
    GGMLQuantizationType.Q8_0: quantize_blocks_Q8_0,
    GGMLQuantizationType.Q5_1: quantize_blocks_Q5_1,
    GGMLQuantizationType.Q5_0: quantize_blocks_Q5_0,
    GGMLQuantizationType.Q4_1: quantize_blocks_Q4_1,
    GGMLQuantizationType.Q4_0: quantize_blocks_Q4_0,
    GGMLQuantizationType.Q6_K: quantize_blocks_Q6_K,
    GGMLQuantizationType.Q5_K: quantize_blocks_Q5_K,
    GGMLQuantizationType.Q4_K: quantize_blocks_Q4_K,
    GGMLQuantizationType.Q3_K: quantize_blocks_Q3_K,
    GGMLQuantizationType.Q2_K: quantize_blocks_Q2_K,
    GGMLQuantizationType.TQ1_0: quantize_blocks_TQ1_0,
    GGMLQuantizationType.TQ2_0: quantize_blocks_TQ2_0,
    GGMLQuantizationType.MXFP4: quantize_blocks_MXFP4,
    GGMLQuantizationType.NVFP4: quantize_blocks_NVFP4,
    GGMLQuantizationType.IQ4_NL: quantize_blocks_IQ4_NL,
    GGMLQuantizationType.IQ4_XS: quantize_blocks_IQ4_XS,
    GGMLQuantizationType.IQ3_XXS: quantize_blocks_IQ3_XXS,
    GGMLQuantizationType.IQ3_S: quantize_blocks_IQ3_S,
    GGMLQuantizationType.IQ2_XXS: quantize_blocks_IQ2_XXS,
    GGMLQuantizationType.IQ2_XS: quantize_blocks_IQ2_XS,
    GGMLQuantizationType.IQ2_S: quantize_blocks_IQ2_S,
    GGMLQuantizationType.IQ1_S: quantize_blocks_IQ1_S,
    GGMLQuantizationType.IQ1_M: quantize_blocks_IQ1_M,
}

dequantize_functions = {
    GGMLQuantizationType.BF16: dequantize_blocks_BF16,
    GGMLQuantizationType.Q1_0: dequantize_blocks_Q1_0,
    GGMLQuantizationType.Q8_0: dequantize_blocks_Q8_0,
    GGMLQuantizationType.Q5_1: dequantize_blocks_Q5_1,
    GGMLQuantizationType.Q5_0: dequantize_blocks_Q5_0,
    GGMLQuantizationType.Q4_1: dequantize_blocks_Q4_1,
    GGMLQuantizationType.Q4_0: dequantize_blocks_Q4_0,
    GGMLQuantizationType.Q6_K: dequantize_blocks_Q6_K,
    GGMLQuantizationType.Q5_K: dequantize_blocks_Q5_K,
    GGMLQuantizationType.Q4_K: dequantize_blocks_Q4_K,
    GGMLQuantizationType.Q3_K: dequantize_blocks_Q3_K,
    GGMLQuantizationType.Q2_K: dequantize_blocks_Q2_K,
    GGMLQuantizationType.IQ2_XXS: dequantize_blocks_IQ2_XXS,
    GGMLQuantizationType.IQ2_XS: dequantize_blocks_IQ2_XS,
    GGMLQuantizationType.IQ3_XXS: dequantize_blocks_IQ3_XXS,
    GGMLQuantizationType.IQ1_S: dequantize_blocks_IQ1_S,
    GGMLQuantizationType.IQ3_S: dequantize_blocks_IQ3_S,
    GGMLQuantizationType.IQ2_S: dequantize_blocks_IQ2_S,
    GGMLQuantizationType.IQ1_M: dequantize_blocks_IQ1_M,
    GGMLQuantizationType.IQ4_NL: dequantize_blocks_IQ4_NL,
    GGMLQuantizationType.IQ4_XS: dequantize_blocks_IQ4_XS,
    GGMLQuantizationType.TQ1_0: dequantize_blocks_TQ1_0,
    GGMLQuantizationType.TQ2_0: dequantize_blocks_TQ2_0,
    GGMLQuantizationType.MXFP4: dequantize_blocks_MXFP4,
    GGMLQuantizationType.NVFP4: dequantize_blocks_NVFP4,
}
