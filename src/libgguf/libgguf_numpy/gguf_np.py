from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence
from math import log2, ceil

from numpy.typing import DTypeLike

import numpy as np
from libgguf._metadata import (
    GGML_QUANT_SIZES,
    QK_K,
    GGMLQuantizationType,
    quant_shape_from_byte_shape,
    quant_shape_to_byte_shape,
)


# This is faster than np.vectorize and np.apply_along_axis because it works on more than one row at a time
def _apply_over_grouped_rows(func: Callable[[np.ndarray], np.ndarray], arr: np.ndarray, otype: DTypeLike, oshape: tuple[int, ...]) -> np.ndarray:
    rows = arr.reshape((-1, arr.shape[-1]))
    osize = 1
    for dim in oshape:
        osize *= dim
    out = np.empty(shape=osize, dtype=otype)
    # compute over groups of 16 rows (arbitrary, but seems good for performance)
    n_groups = (rows.shape[0] // 16) or 1
    np.concatenate([func(group).ravel() for group in np.array_split(rows, n_groups)], axis=0, out=out)
    return out.reshape(oshape)


# round away from zero
# ref: https://stackoverflow.com/a/59143326/22827863
def np_roundf(n: np.ndarray) -> np.ndarray:
    a = abs(n)
    floored = np.floor(a)
    b = floored + np.floor(2 * (a - floored))
    return np.sign(n) * b


class QuantError(Exception): ...


_type_traits: dict[GGMLQuantizationType, type[__Quant]] = {}
GROUP_MAX_EPS = np.float32(1.0e-15)
GROUP_MAX_EPS_IQ3_XXS = np.float32(1.0e-8)
GROUP_MAX_EPS_IQ2_S = np.float32(1.0e-8)
GROUP_MAX_EPS_IQ1_S = np.float32(1.0e-12)
GROUP_MAX_EPS_IQ1_M = np.float32(1.0e-7)


def _quantize_blocks_with_libgguf(blocks: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
    try:
        import libgguf
    except ImportError as exc:
        raise RuntimeError(
            f"Quantization for {qtype.name} requires the separate `libgguf` package. "
            "Install it with `python -m pip install -e libgguf --no-build-isolation`."
        ) from exc

    return libgguf.quantize_rows(blocks, qtype)


def _nearest_int(values: np.ndarray | np.float32 | float) -> np.ndarray:
    fvals = np.asarray(values, dtype=np.float32)
    biased = fvals + np.float32(12_582_912.0)
    ints = np.ascontiguousarray(biased).view(np.int32)
    return (ints & 0x007FFFFF) - 0x00400000


def _make_qx_quants(
    x: np.ndarray, nmax: int, quant_weights: np.ndarray | None = None
) -> tuple[np.float32, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    ax = np.abs(x)
    imax = int(np.argmax(ax))
    amax = ax[imax]
    max_v = x[imax]
    if amax < GROUP_MAX_EPS:
        return np.float32(0.0), np.zeros(x.shape, dtype=np.int8)

    iscale = np.float32(-nmax) / max_v
    l = _nearest_int(iscale * x).clip(-nmax, nmax - 1).astype(np.int8)
    w = x * x if quant_weights is None else np.asarray(quant_weights, dtype=np.float32)
    sumlx = np.float32(0.0)
    suml2 = np.float32(0.0)
    for i in range(x.size):
        lf = np.float32(l[i])
        sumlx = np.float32(sumlx + np.float32(w[i] * x[i] * lf))
        suml2 = np.float32(suml2 + np.float32(w[i] * lf * lf))
    scale = np.float32(sumlx / suml2) if suml2 != 0 else np.float32(0.0)
    best = np.float32(scale * sumlx)
    best_l = l.copy()

    for is_ in range(-9, 10):
        if is_ == 0:
            continue
        iscale = -(np.float32(nmax) + np.float32(0.1) * np.float32(is_)) / max_v
        l_try = _nearest_int(iscale * x).clip(-nmax, nmax - 1).astype(np.int8)
        sumlx = np.float32(0.0)
        suml2 = np.float32(0.0)
        for i in range(x.size):
            lf = np.float32(l_try[i])
            sumlx = np.float32(sumlx + np.float32(w[i] * x[i] * lf))
            suml2 = np.float32(suml2 + np.float32(w[i] * lf * lf))
        if suml2 > 0 and sumlx * sumlx > best * suml2:
            best_l = l_try
            scale = np.float32(sumlx / suml2)
            best = np.float32(scale * sumlx)

    return scale, (best_l + np.int8(nmax)).astype(np.int8)


def _make_q3_quants(x: np.ndarray, nmax: int) -> tuple[np.float32, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    ax = np.abs(x)
    imax = int(np.argmax(ax))
    amax = ax[imax]
    max_v = x[imax]
    if amax < GROUP_MAX_EPS:
        return np.float32(0.0), np.zeros(x.shape, dtype=np.int8)

    iscale = np.float32(-nmax) / max_v
    L = _nearest_int(iscale * x).clip(-nmax, nmax - 1).astype(np.int8)
    w = x * x
    sumlx = np.float32(0.0)
    suml2 = np.float32(0.0)
    for i in range(x.size):
        lf = np.float32(L[i])
        sumlx = np.float32(sumlx + np.float32(w[i] * x[i] * lf))
        suml2 = np.float32(suml2 + np.float32(w[i] * lf * lf))

    for _ in range(5):
        n_changed = 0
        for i in range(x.size):
            lf = np.float32(L[i])
            wi = np.float32(w[i])
            slx = np.float32(sumlx - np.float32(wi * x[i] * lf))
            if slx > 0:
                sl2 = np.float32(suml2 - np.float32(wi * lf * lf))
                new_l = int(_nearest_int(x[i] * sl2 / slx).item())
                new_l = max(-nmax, min(nmax - 1, new_l))
                if new_l != int(L[i]):
                    new_l_f = np.float32(new_l)
                    slx = np.float32(slx + np.float32(wi * x[i] * new_l_f))
                    sl2 = np.float32(sl2 + np.float32(wi * new_l_f * new_l_f))
                    if sl2 > 0 and slx * slx * suml2 > sumlx * sumlx * sl2:
                        L[i] = new_l
                        sumlx = slx
                        suml2 = sl2
                        n_changed += 1
        if n_changed == 0:
            break

    return (
        np.float32(sumlx / suml2) if suml2 > 0 else np.float32(0.0),
        (L + np.int8(nmax)).astype(np.int8),
    )


def _make_qp_quants(
    x: np.ndarray,
    nmax: int,
    quant_weights: np.ndarray,
) -> tuple[np.float32, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    quant_weights = np.asarray(quant_weights, dtype=np.float32)
    max_v = np.float32(np.max(x))
    L = np.zeros(x.shape, dtype=np.uint8)
    if max_v < GROUP_MAX_EPS:
        return np.float32(0.0), L

    iscale = np.float32(nmax) / max_v
    for i in range(x.size):
        L[i] = int(_nearest_int(np.float32(iscale * x[i])).item()) & 0xFF
    scale = np.float32(1.0) / iscale
    best_mse = np.float32(0.0)
    for i in range(x.size):
        diff = np.float32(x[i] - np.float32(scale * L[i]))
        best_mse = np.float32(best_mse + np.float32(quant_weights[i] * diff * diff))

    for is_ in range(-4, 5):
        if is_ == 0:
            continue
        iscale_is = (np.float32(0.1) * np.float32(is_) + np.float32(nmax)) / max_v
        scale_is = np.float32(1.0) / iscale_is
        mse = np.float32(0.0)
        for i in range(x.size):
            l = int(_nearest_int(np.float32(iscale_is * x[i])).item())
            l = min(nmax, l)
            diff = np.float32(x[i] - np.float32(scale_is * l))
            mse = np.float32(mse + np.float32(quant_weights[i] * diff * diff))
        if mse < best_mse:
            best_mse = mse
            iscale = iscale_is

    sumlx = np.float32(0.0)
    suml2 = np.float32(0.0)
    for i in range(x.size):
        l = int(_nearest_int(np.float32(iscale * x[i])).item())
        l = min(nmax, l)
        L[i] = l & 0xFF
        lf = np.float32(L[i])
        w = np.float32(quant_weights[i])
        sumlx = np.float32(sumlx + np.float32(w * x[i] * lf))
        suml2 = np.float32(suml2 + np.float32(w * lf * lf))

    for _ in range(5):
        n_changed = 0
        for i in range(x.size):
            w = np.float32(quant_weights[i])
            lf = np.float32(L[i])
            slx = np.float32(sumlx - np.float32(w * x[i] * lf))
            sl2 = np.float32(suml2 - np.float32(w * lf * lf))
            if slx > 0 and sl2 > 0:
                new_l = int(_nearest_int(np.float32(x[i] * sl2 / slx)).item())
                new_l = min(nmax, new_l)
                if new_l != int(L[i]):
                    new_lf = np.float32(new_l)
                    slx = np.float32(slx + np.float32(w * x[i] * new_lf))
                    sl2 = np.float32(sl2 + np.float32(w * new_lf * new_lf))
                    if slx * slx * suml2 > sumlx * sumlx * sl2:
                        L[i] = new_l & 0xFF
                        sumlx = slx
                        suml2 = sl2
                        n_changed += 1
        if n_changed == 0:
            break

    return (np.float32(sumlx / suml2) if suml2 > 0 else np.float32(0.0)), L


def _make_qkx2_quants(
    x: np.ndarray,
    weights: np.ndarray,
    nmax: int,
    rmin: float,
    rdelta: float,
    nstep: int,
    use_mad: bool = False,
) -> tuple[np.float32, np.float32, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    min_v = np.float32(x[0])
    max_v = np.float32(x[0])
    sum_w = np.float32(weights[0])
    sum_x = np.float32(sum_w * x[0])
    for i in range(1, x.size):
        if x[i] < min_v:
            min_v = np.float32(x[i])
        if x[i] > max_v:
            max_v = np.float32(x[i])
        w = np.float32(weights[i])
        sum_w = np.float32(sum_w + w)
        sum_x = np.float32(sum_x + np.float32(w * x[i]))

    if min_v > 0:
        min_v = np.float32(0.0)
    if max_v == min_v:
        return np.float32(0.0), np.float32(-min_v), np.zeros(x.shape, dtype=np.uint8)

    iscale = np.float32(nmax) / np.float32(max_v - min_v)
    scale = np.float32(1.0) / iscale
    L = np.zeros(x.shape, dtype=np.uint8)
    best_error = np.float32(0.0)
    for i in range(x.size):
        l = int(_nearest_int(iscale * np.float32(x[i] - min_v)).item())
        l = max(0, min(nmax, l))
        L[i] = l
        diff = np.float32(scale * l + min_v - x[i])
        if use_mad:
            diff = np.abs(diff)
        else:
            diff = np.float32(diff * diff)
        best_error = np.float32(best_error + np.float32(weights[i] * diff))

    for is_ in range(nstep + 1):
        iscale = (
            np.float32(rmin)
            + np.float32(rdelta) * np.float32(is_)
            + np.float32(nmax)
        ) / np.float32(max_v - min_v)
        Laux = np.zeros(x.shape, dtype=np.uint8)
        sum_l = np.float32(0.0)
        sum_l2 = np.float32(0.0)
        sum_xl = np.float32(0.0)
        for i in range(x.size):
            l = int(_nearest_int(iscale * np.float32(x[i] - min_v)).item())
            l = max(0, min(nmax, l))
            Laux[i] = l
            w = np.float32(weights[i])
            lf = np.float32(l)
            sum_l = np.float32(sum_l + np.float32(w * lf))
            sum_l2 = np.float32(sum_l2 + np.float32(w * lf * lf))
            sum_xl = np.float32(sum_xl + np.float32(w * lf * x[i]))

        D = np.float32(sum_w * sum_l2 - sum_l * sum_l)
        if D > 0:
            this_scale = np.float32((sum_w * sum_xl - sum_x * sum_l) / D)
            this_min = np.float32((sum_l2 * sum_x - sum_l * sum_xl) / D)
            if this_min > 0:
                this_min = np.float32(0.0)
                this_scale = np.float32(sum_xl / sum_l2)

            cur_error = np.float32(0.0)
            for i in range(x.size):
                diff = np.float32(this_scale * Laux[i] + this_min - x[i])
                if use_mad:
                    diff = np.abs(diff)
                else:
                    diff = np.float32(diff * diff)
                cur_error = np.float32(cur_error + np.float32(weights[i] * diff))

            if cur_error < best_error:
                L = Laux
                best_error = cur_error
                scale = this_scale
                min_v = this_min

    return scale, np.float32(-min_v), L


def _quantize_q4_k_blocks(blocks: np.ndarray, type_size: int) -> np.ndarray:
    n_blocks = blocks.shape[0]
    blocks = blocks.astype(np.float32, copy=False)
    out = np.zeros((n_blocks, type_size), dtype=np.uint8)

    for i, x in enumerate(blocks):
        L = np.zeros(QK_K, dtype=np.uint8)
        scales = np.zeros(QK_K // 32, dtype=np.float32)
        mins = np.zeros(QK_K // 32, dtype=np.float32)
        max_scale = np.float32(0.0)
        max_min = np.float32(0.0)

        for j in range(QK_K // 32):
            start = 32 * j
            sub = x[start:start + 32]
            sum_x2 = np.float32(0.0)
            for value in sub:
                sum_x2 = np.float32(sum_x2 + np.float32(value * value))
            av_x = np.float32(np.sqrt(np.float32(sum_x2 / np.float32(32.0))))
            weights = av_x + np.abs(sub)
            scales[j], mins[j], L[start:start + 32] = _make_qkx2_quants(
                sub, weights, 15, -1.0, 0.1, 20
            )
            if scales[j] > max_scale:
                max_scale = scales[j]
            if mins[j] > max_min:
                max_min = mins[j]

        inv_scale = np.float32(63.0) / max_scale if max_scale > 0 else np.float32(0.0)
        inv_min = np.float32(63.0) / max_min if max_min > 0 else np.float32(0.0)
        ls = np.zeros(QK_K // 32, dtype=np.uint8)
        lm = np.zeros(QK_K // 32, dtype=np.uint8)
        scales_packed = out[i, 4:16]
        for j in range(QK_K // 32):
            ls_j = min(63, int(_nearest_int(inv_scale * scales[j]).item()))
            lm_j = min(63, int(_nearest_int(inv_min * mins[j]).item()))
            ls[j] = ls_j
            lm[j] = lm_j
            if j < 4:
                scales_packed[j] = ls_j
                scales_packed[j + 4] = lm_j
            else:
                scales_packed[j + 4] = (ls_j & 0x0F) | ((lm_j & 0x0F) << 4)
                scales_packed[j - 4] |= (ls_j >> 4) << 6
                scales_packed[j] |= (lm_j >> 4) << 6

        d = np.array([max_scale / np.float32(63.0)], dtype=np.float16)
        dmin = np.array([max_min / np.float32(63.0)], dtype=np.float16)
        out[i, :2] = d.view(np.uint8)
        out[i, 2:4] = dmin.view(np.uint8)
        d_f32 = d.astype(np.float32)[0]
        dmin_f32 = dmin.astype(np.float32)[0]

        for j in range(QK_K // 32):
            d_sub = np.float32(d_f32 * ls[j])
            if d_sub == 0:
                continue
            dm_sub = np.float32(dmin_f32 * lm[j])
            start = 32 * j
            l = _nearest_int((x[start:start + 32] + dm_sub) / d_sub).clip(0, 15)
            L[start:start + 32] = l.astype(np.uint8)

        qs = out[i, 16:]
        for j in range(0, QK_K, 64):
            for l in range(32):
                qs[j // 2 + l] = L[j + l] | (L[j + l + 32] << np.uint8(4))

    return out


def _best_index_int8(values: Sequence[int], x: np.float32 | float) -> int:
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


def _quantize_iq4_nl_impl(
    x: np.ndarray,
    super_block_size: int,
    block_size: int,
    values: Sequence[int],
    ntry: int,
) -> tuple[np.float16, np.ndarray, int, np.ndarray | None]:
    L = np.zeros(super_block_size, dtype=np.uint8)
    q4 = np.zeros(super_block_size // 2, dtype=np.uint8)
    scales = np.zeros(super_block_size // block_size, dtype=np.float32)
    weight = np.zeros(block_size, dtype=np.float32)

    max_scale = np.float32(0.0)
    amax_scale = np.float32(0.0)
    for ib in range(super_block_size // block_size):
        start = ib * block_size
        xb = x[start:start + block_size]
        Lb = L[start:start + block_size]
        for j in range(block_size):
            weight[j] = np.float32(xb[j] * xb[j])

        amax = np.float32(0.0)
        max_v = np.float32(0.0)
        for value in xb:
            ax = np.abs(value)
            if ax > amax:
                amax = ax
                max_v = np.float32(value)
        if amax < GROUP_MAX_EPS:
            scales[ib] = np.float32(0.0)
            continue

        d = (-max_v / np.float32(values[0])) if ntry > 0 else (max_v / np.float32(values[0]))
        id_ = np.float32(1.0) / d
        sumqx = np.float32(0.0)
        sumq2 = np.float32(0.0)
        for j in range(block_size):
            l = _best_index_int8(values, np.float32(id_ * xb[j]))
            Lb[j] = l
            q = np.float32(values[l])
            w = weight[j]
            sumqx = np.float32(sumqx + np.float32(w * q * xb[j]))
            sumq2 = np.float32(sumq2 + np.float32(w * q * q))
        d = np.float32(sumqx / sumq2) if sumq2 > 0 else np.float32(0.0)
        best = np.float32(d * sumqx)

        for itry in range(-ntry, ntry + 1):
            id_ = np.float32(itry + values[0]) / max_v
            sumqx = np.float32(0.0)
            sumq2 = np.float32(0.0)
            for j in range(block_size):
                l = _best_index_int8(values, np.float32(id_ * xb[j]))
                q = np.float32(values[l])
                w = weight[j]
                sumqx = np.float32(sumqx + np.float32(w * q * xb[j]))
                sumq2 = np.float32(sumq2 + np.float32(w * q * q))
            if sumq2 > 0 and sumqx * sumqx > best * sumq2:
                d = np.float32(sumqx / sumq2)
                best = np.float32(d * sumqx)

        scales[ib] = d
        abs_d = np.abs(d)
        if abs_d > amax_scale:
            amax_scale = abs_d
            max_scale = d

    scales_h = 0
    scales_l: np.ndarray | None = None
    if super_block_size // block_size > 1:
        nb = super_block_size // block_size
        scales_l = np.zeros((nb + 1) // 2, dtype=np.uint8)
        d = np.float32(-max_scale / np.float32(32.0))
        dh = np.array([d], dtype=np.float16)[0]
        id_ = np.float32(1.0) / d if d != 0 else np.float32(0.0)
        for ib in range(nb):
            l = int(_nearest_int(id_ * scales[ib]).item())
            l = max(-32, min(31, l))
            dl = np.float32(d * l)
            idl = np.float32(1.0) / dl if dl != 0 else np.float32(0.0)
            start = ib * block_size
            xb = x[start:start + block_size]
            for j in range(block_size):
                L[start + j] = _best_index_int8(values, np.float32(idl * xb[j]))
            l += 32
            l_l = l & 0x0F
            l_h = l >> 4
            if ib % 2 == 0:
                scales_l[ib // 2] = l_l
            else:
                scales_l[ib // 2] |= l_l << 4
            scales_h |= l_h << (2 * (ib % 8))
    else:
        dh = np.array([scales[0]], dtype=np.float16)[0]
        if ntry > 0:
            id_ = np.float32(1.0) / dh.astype(np.float32) if dh != 0 else np.float32(0.0)
            for j in range(super_block_size):
                L[j] = _best_index_int8(values, np.float32(id_ * x[j]))

    for i in range(super_block_size // 32):
        for j in range(16):
            q4[16 * i + j] = L[32 * i + j] | (L[32 * i + 16 + j] << np.uint8(4))

    return dh, q4, scales_h, scales_l


def _decode_lattice_indices(cls: type["__Quant"]) -> np.ndarray:
    cls.init_grid()
    assert cls.grid is not None
    grid = cls.grid.reshape(cls.grid_shape)
    grid = grid.astype(np.float32)
    if cls.grid_map and cls.grid_map[0] == 0x04:
        grid = grid / np.float32(4.0)
    return np.rint((grid - np.float32(1.0)) / np.float32(2.0)).astype(np.int16)


def _decode_packed_grid_indices(cls: type["__Quant"]) -> np.ndarray:
    assert cls.grid_hex is not None
    bits_per_elem = ceil(log2(len(cls.grid_map)))
    elems_per_byte = 8 // bits_per_elem
    grid = np.frombuffer(cls.grid_hex, dtype=np.uint8)
    grid = grid.reshape((-1, 2))
    grid = (np.where(grid > 0x40, grid + 9, grid) & 0x0F) << np.array([4, 0], dtype=np.uint8).reshape((1, 2))
    grid = grid[..., 0] | grid[..., 1]
    grid = grid.reshape((-1, 1)) >> np.array(
        [i for i in range(0, 8, 8 // elems_per_byte)], dtype=np.uint8
    ).reshape((1, elems_per_byte))
    grid = (grid & ((1 << bits_per_elem) - 1)).reshape(cls.grid_shape)
    return grid.astype(np.int16)


def _build_lattice_lookup(
    grid_l: np.ndarray,
    bits_per_elem: int,
    grid_size: int,
    nwant: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    elems_per_grid = grid_l.shape[1]
    map_size = 1 << (bits_per_elem * elems_per_grid)
    kmap = np.full(map_size, -1, dtype=np.int32)

    shifts = np.arange(elems_per_grid, dtype=np.uint16) * np.uint16(bits_per_elem)
    for i in range(grid_size):
        index = int(np.sum(grid_l[i].astype(np.uint16) << shifts, dtype=np.uint16))
        kmap[index] = i

    neighbours: list[np.ndarray] = [np.empty(0, dtype=np.uint16) for _ in range(map_size)]
    grid_pos = (2 * grid_l.astype(np.int16)) + 1
    for i in range(map_size):
        if kmap[i] >= 0:
            continue
        pos = np.array(
            [(i >> (bits_per_elem * k)) & ((1 << bits_per_elem) - 1) for k in range(elems_per_grid)],
            dtype=np.int16,
        )
        pos = 2 * pos + 1
        dist2 = np.sum((grid_pos - pos.reshape(1, elems_per_grid)) ** 2, axis=1)
        order = np.lexsort((np.arange(grid_size, dtype=np.int32), dist2))
        distinct = 1
        cutoff = dist2[order[0]]
        count = 0
        for j in order:
            if dist2[j] > cutoff:
                if distinct == nwant:
                    break
                cutoff = dist2[j]
                distinct += 1
            count += 1
        neighbours[i] = order[:count].astype(np.uint16)
    return kmap, neighbours


def _best_lattice_neighbour(
    neighbours: np.ndarray,
    grid_l: np.ndarray,
    xval: np.ndarray,
    weight: np.ndarray,
    scale: np.float32,
    tie_eps: np.float32 = np.float32(0.0),
) -> tuple[int, np.ndarray]:
    best_d2 = np.float32(np.inf)
    best_index = -1
    for index in neighbours:
        pos = (2 * grid_l[int(index)].astype(np.float32)) + np.float32(1.0)
        d2 = np.float32(0.0)
        for i in range(xval.size):
            diff = np.float32(scale * pos[i] - xval[i])
            d2 = np.float32(d2 + np.float32(weight[i] * diff * diff))
        if d2 < best_d2 - tie_eps or (tie_eps > 0 and np.abs(d2 - best_d2) <= tie_eps and int(index) < best_index):
            best_d2 = d2
            best_index = int(index)
    return best_index, grid_l[best_index].astype(np.int8)


def _best_iq1_neighbour(
    neighbours: np.ndarray,
    grid_l: np.ndarray,
    xval: np.ndarray,
    weight: np.ndarray,
    scale: np.float32,
    values: np.ndarray,
) -> tuple[int, np.ndarray]:
    best_d2 = np.float32(np.inf)
    best_index = -1
    for index in neighbours:
        levels = grid_l[int(index)].astype(np.int8)
        d2 = np.float32(0.0)
        for i in range(xval.size):
            q = np.float32(values[int(levels[i])])
            diff = np.float32(np.float32(scale * q) - xval[i])
            d2 = np.float32(d2 + np.float32(weight[i] * diff * diff))
        if d2 < best_d2:
            best_d2 = d2
            best_index = int(index)
    return best_index, grid_l[best_index].astype(np.int8)


_iq3_xxs_lookup: tuple[np.ndarray, list[np.ndarray], np.ndarray] | None = None
_iq3_s_lookup: tuple[np.ndarray, list[np.ndarray], np.ndarray] | None = None
_iq2_xxs_lookup: tuple[np.ndarray, list[np.ndarray], np.ndarray] | None = None
_iq2_xs_lookup: tuple[np.ndarray, list[np.ndarray], np.ndarray] | None = None
_iq2_s_lookup: tuple[np.ndarray, list[np.ndarray], np.ndarray] | None = None
_iq1_s_lookup: tuple[np.ndarray, list[np.ndarray], np.ndarray] | None = None


def _get_iq3_xxs_lookup() -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    global _iq3_xxs_lookup
    if _iq3_xxs_lookup is None:
        grid_l = _decode_lattice_indices(IQ3_XXS)
        kmap, neighbours = _build_lattice_lookup(grid_l, 3, 256, 2)
        _iq3_xxs_lookup = (kmap, neighbours, grid_l)
    return _iq3_xxs_lookup


def _get_iq3_s_lookup() -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    global _iq3_s_lookup
    if _iq3_s_lookup is None:
        grid_l = _decode_lattice_indices(IQ3_S)
        kmap, neighbours = _build_lattice_lookup(grid_l, 3, 512, 3)
        _iq3_s_lookup = (kmap, neighbours, grid_l)
    return _iq3_s_lookup


def _get_iq2_xxs_lookup() -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    global _iq2_xxs_lookup
    if _iq2_xxs_lookup is None:
        grid_l = _decode_packed_grid_indices(IQ2_XXS)
        kmap, neighbours = _build_lattice_lookup(grid_l, 2, 256, 2)
        _iq2_xxs_lookup = (kmap, neighbours, grid_l)
    return _iq2_xxs_lookup


def _get_iq2_xs_lookup() -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    global _iq2_xs_lookup
    if _iq2_xs_lookup is None:
        grid_l = _decode_packed_grid_indices(IQ2_XS)
        kmap, neighbours = _build_lattice_lookup(grid_l, 2, 512, 2)
        _iq2_xs_lookup = (kmap, neighbours, grid_l)
    return _iq2_xs_lookup


def _get_iq2_s_lookup() -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    global _iq2_s_lookup
    if _iq2_s_lookup is None:
        grid_l = _decode_packed_grid_indices(IQ2_S)
        kmap, neighbours = _build_lattice_lookup(grid_l, 2, 1024, 1)
        _iq2_s_lookup = (kmap, neighbours, grid_l)
    return _iq2_s_lookup


def _get_iq1_s_lookup() -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    global _iq1_s_lookup
    if _iq1_s_lookup is None:
        grid_l = _decode_packed_grid_indices(IQ1_S)
        kmap, neighbours = _build_lattice_lookup(grid_l, 2, 2048, 3)
        _iq1_s_lookup = (kmap, neighbours, grid_l)
    return _iq1_s_lookup


def quantize(data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
    if qtype == GGMLQuantizationType.F32:
        return data.astype(np.float32, copy=False)
    elif qtype == GGMLQuantizationType.F16:
        return data.astype(np.float16, copy=False)
    elif (q := _type_traits.get(qtype)) is not None:
        return q.quantize(data)
    else:
        raise NotImplementedError(f"Quantization for {qtype.name} is not yet implemented")


def dequantize(data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
    if qtype == GGMLQuantizationType.F32:
        return data.view(np.float32)
    elif qtype == GGMLQuantizationType.F16:
        return data.view(np.float16).astype(np.float32)
    elif (q := _type_traits.get(qtype)) is not None:
        return q.dequantize(data)
    else:
        raise NotImplementedError(f"Dequantization for {qtype.name} is not yet implemented")


class __Quant(ABC):
    qtype: GGMLQuantizationType
    block_size: int
    type_size: int

    grid: np.ndarray[Any, np.dtype[np.float32]] | None = None
    grid_shape: tuple[int, int] = (0, 0)
    grid_map: tuple[int | float, ...] = ()
    grid_hex: bytes | None = None

    def __init__(self):
        return TypeError("Quant conversion classes can't have instances")

    def __init_subclass__(cls, qtype: GGMLQuantizationType) -> None:
        cls.qtype = qtype
        cls.block_size, cls.type_size = GGML_QUANT_SIZES[qtype]
        assert qtype not in _type_traits
        _type_traits[qtype] = cls

    @classmethod
    def init_grid(cls):
        if cls.grid is not None or cls.grid_hex is None:
            return

        bits_per_elem = ceil(log2(len(cls.grid_map)))
        assert bits_per_elem != 0, cls.qtype.name
        elems_per_byte = 8 // bits_per_elem

        grid = np.frombuffer(cls.grid_hex, dtype=np.uint8)
        # decode hexadecimal chars from grid
        grid = grid.reshape((-1, 2))
        grid = (np.where(grid > 0x40, grid + 9, grid) & 0x0F) << np.array([4, 0], dtype=np.uint8).reshape((1, 2))
        grid = grid[..., 0] | grid[..., 1]
        # unpack the grid values
        grid = grid.reshape((-1, 1)) >> np.array([i for i in range(0, 8, 8 // elems_per_byte)], dtype=np.uint8).reshape((1, elems_per_byte))
        grid = (grid & ((1 << bits_per_elem) - 1)).reshape((-1, 1))
        grid_map = np.array(cls.grid_map, dtype=np.float32).reshape((1, -1))
        grid = np.take_along_axis(grid_map, grid, axis=-1)
        cls.grid = grid.reshape((1, 1, *cls.grid_shape))

    @classmethod
    @abstractmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def quantize_rows(cls, rows: np.ndarray) -> np.ndarray:
        rows = rows.astype(np.float32, copy=False)
        shape = rows.shape
        n_blocks = rows.size // cls.block_size
        blocks = rows.reshape((n_blocks, cls.block_size))
        blocks = cls.quantize_blocks(blocks)
        assert blocks.dtype == np.uint8
        assert blocks.shape[-1] == cls.type_size
        return blocks.reshape(cls.__shape_to_bytes(shape))

    @classmethod
    def dequantize_rows(cls, rows: np.ndarray) -> np.ndarray:
        rows = rows.view(np.uint8)
        shape = rows.shape
        n_blocks = rows.size // cls.type_size
        blocks = rows.reshape((n_blocks, cls.type_size))
        blocks = cls.dequantize_blocks(blocks)
        assert blocks.dtype == np.float32
        assert blocks.shape[-1] == cls.block_size
        return blocks.reshape(cls.__shape_from_bytes(shape))

    @classmethod
    def __shape_to_bytes(cls, shape: Sequence[int]):
        return quant_shape_to_byte_shape(shape, cls.qtype)

    @classmethod
    def __shape_from_bytes(cls, shape: Sequence[int]):
        return quant_shape_from_byte_shape(shape, cls.qtype)

    @classmethod
    def __quantize_array(cls, array: np.ndarray) -> np.ndarray:
        return _apply_over_grouped_rows(cls.quantize_rows, arr=array, otype=np.uint8, oshape=cls.__shape_to_bytes(array.shape))

    @classmethod
    def __dequantize_array(cls, array: np.ndarray) -> np.ndarray:
        cls.init_grid()
        return _apply_over_grouped_rows(cls.dequantize_rows, arr=array, otype=np.float32, oshape=cls.__shape_from_bytes(array.shape))

    @classmethod
    def can_quantize(cls, tensor: np.ndarray) -> bool:
        return tensor.shape[-1] % cls.block_size == 0

    @classmethod
    def quantize(cls, tensor: np.ndarray) -> np.ndarray:
        if not cls.can_quantize(tensor):
            raise QuantError(f"Can't quantize tensor with shape {tensor.shape} to {cls.qtype.name}")
        return cls.__quantize_array(tensor)

    @classmethod
    def dequantize(cls, tensor: np.ndarray) -> np.ndarray:
        return cls.__dequantize_array(tensor)


class BF16(__Quant, qtype=GGMLQuantizationType.BF16):
    @classmethod
    # same as ggml_compute_fp32_to_bf16 in ggml-impl.h
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n = blocks.view(np.uint32)
        # force nan to quiet
        n = np.where((n & 0x7fffffff) > 0x7f800000, (n & np.uint32(0xffff0000)) | np.uint32(64 << 16), n)
        # round to nearest even
        n = (np.uint64(n) + (0x7fff + ((n >> 16) & 1))) >> 16
        return n.astype(np.uint16).view(np.uint8)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        return (blocks.view(np.int16).astype(np.int32) << 16).view(np.float32)


class Q1_0(__Quant, qtype=GGMLQuantizationType.Q1_0):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d = np.mean(np.abs(blocks), axis=-1, keepdims=True).astype(np.float16).view(np.uint8)

        signs = (blocks >= 0).astype(np.uint8).reshape((n_blocks, cls.block_size // 8, 8))
        shifts = np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 8))
        qs = np.sum(signs << shifts, axis=-1, dtype=np.uint8)

        return np.concatenate([d, qs], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, qs = np.hsplit(blocks, [2])
        d = d.view(np.float16).astype(np.float32)

        bits = qs.reshape((n_blocks, cls.block_size // 8, 1)) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 8))
        bits = (bits & np.uint8(0x01)).reshape((n_blocks, cls.block_size))
        values = np.where(bits != 0, d, -d)

        return values.astype(np.float32)


class Q4_0(__Quant, qtype=GGMLQuantizationType.Q4_0):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        imax = abs(blocks).argmax(axis=-1, keepdims=True)
        max = np.take_along_axis(blocks, imax, axis=-1)

        d = max / -8
        with np.errstate(divide="ignore"):
            id = np.where(d == 0, 0, 1 / d)
        qs = np.trunc((blocks * id) + np.float32(8.5), dtype=np.float32).astype(np.uint8).clip(0, 15)

        qs = qs.reshape((n_blocks, 2, cls.block_size // 2))
        qs = qs[..., 0, :] | (qs[..., 1, :] << np.uint8(4))

        d = d.astype(np.float16).view(np.uint8)

        return np.concatenate([d, qs], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, qs = np.hsplit(blocks, [2])

        d = d.view(np.float16).astype(np.float32)

        qs = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1)).astype(np.int8) - np.int8(8)

        return (d * qs.astype(np.float32))


class Q4_1(__Quant, qtype=GGMLQuantizationType.Q4_1):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        max = blocks.max(axis=-1, keepdims=True)
        min = blocks.min(axis=-1, keepdims=True)

        d = (max - min) / 15
        with np.errstate(divide="ignore"):
            id = np.where(d == 0, 0, 1 / d)
        qs = np.trunc((blocks - min) * id + np.float32(0.5), dtype=np.float32).astype(np.uint8).clip(0, 15)

        qs = qs.reshape((n_blocks, 2, cls.block_size // 2))
        qs = qs[..., 0, :] | (qs[..., 1, :] << np.uint8(4))

        d = d.astype(np.float16).view(np.uint8)
        m = min.astype(np.float16).view(np.uint8)

        return np.concatenate([d, m, qs], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        m, qs = np.hsplit(rest, [2])

        d = d.view(np.float16).astype(np.float32)
        m = m.view(np.float16).astype(np.float32)

        qs = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1)).astype(np.float32)

        return (d * qs) + m


class Q5_0(__Quant, qtype=GGMLQuantizationType.Q5_0):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        imax = abs(blocks).argmax(axis=-1, keepdims=True)
        max = np.take_along_axis(blocks, imax, axis=-1)

        d = max / -16
        with np.errstate(divide="ignore"):
            id = np.where(d == 0, 0, 1 / d)
        q = np.trunc((blocks * id) + np.float32(16.5), dtype=np.float32).astype(np.uint8).clip(0, 31)

        qs = q.reshape((n_blocks, 2, cls.block_size // 2))
        qs = (qs[..., 0, :] & np.uint8(0x0F)) | (qs[..., 1, :] << np.uint8(4))

        qh = np.packbits(q.reshape((n_blocks, 1, 32)) >> np.uint8(4), axis=-1, bitorder="little").reshape(n_blocks, 4)

        d = d.astype(np.float16).view(np.uint8)

        return np.concatenate([d, qh, qs], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        qh, qs = np.hsplit(rest, [4])

        d = d.view(np.float16).astype(np.float32)
        qh = qh.view(np.uint32)

        qh = qh.reshape((n_blocks, 1)) >> np.array([i for i in range(32)], dtype=np.uint32).reshape((1, 32))
        ql = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qh = (qh & np.uint32(0x01)).astype(np.uint8)
        ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1))

        qs = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(16)

        return (d * qs.astype(np.float32))


class Q5_1(__Quant, qtype=GGMLQuantizationType.Q5_1):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        max = blocks.max(axis=-1, keepdims=True)
        min = blocks.min(axis=-1, keepdims=True)

        d = (max - min) / 31
        with np.errstate(divide="ignore"):
            id = np.where(d == 0, 0, 1 / d)
        q = np.trunc((blocks - min) * id + np.float32(0.5), dtype=np.float32).astype(np.uint8).clip(0, 31)

        qs = q.reshape((n_blocks, 2, cls.block_size // 2))
        qs = (qs[..., 0, :] & np.uint8(0x0F)) | (qs[..., 1, :] << np.uint8(4))

        qh = np.packbits(q.reshape((n_blocks, 1, 32)) >> np.uint8(4), axis=-1, bitorder="little").reshape(n_blocks, 4)

        d = d.astype(np.float16).view(np.uint8)
        m = min.astype(np.float16).view(np.uint8)

        return np.concatenate([d, m, qh, qs], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        m, rest = np.hsplit(rest, [2])
        qh, qs = np.hsplit(rest, [4])

        d = d.view(np.float16).astype(np.float32)
        m = m.view(np.float16).astype(np.float32)
        qh = qh.view(np.uint32)

        qh = qh.reshape((n_blocks, 1)) >> np.array([i for i in range(32)], dtype=np.uint32).reshape((1, 32))
        ql = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qh = (qh & np.uint32(0x01)).astype(np.uint8)
        ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1))

        qs = (ql | (qh << np.uint8(4))).astype(np.float32)

        return (d * qs) + m


class Q8_0(__Quant, qtype=GGMLQuantizationType.Q8_0):
    @classmethod
    # Implementation of Q8_0 with bit-exact same results as reference implementation in ggml-quants.c
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:

        d = abs(blocks).max(axis=1, keepdims=True) / 127
        with np.errstate(divide="ignore"):
            id = np.where(d == 0, 0, 1 / d)
        qs = np_roundf(blocks * id)

        # (n_blocks, 2)
        d = d.astype(np.float16).view(np.uint8)
        # (n_blocks, block_size)
        qs = qs.astype(np.int8).view(np.uint8)

        return np.concatenate([d, qs], axis=1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        d, x = np.split(blocks, [2], axis=1)
        d = d.view(np.float16).astype(np.float32)
        x = x.view(np.int8).astype(np.float32)

        return (x * d)


class Q2_K(__Quant, qtype=GGMLQuantizationType.Q2_K):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        blocks = blocks.astype(np.float32, copy=False)
        out = np.zeros((n_blocks, cls.type_size), dtype=np.uint8)

        for i, x in enumerate(blocks):
            L = np.zeros(QK_K, dtype=np.uint8)
            scales = np.zeros(QK_K // 16, dtype=np.float32)
            mins = np.zeros(QK_K // 16, dtype=np.float32)
            max_scale = np.float32(0.0)
            max_min = np.float32(0.0)

            for j in range(QK_K // 16):
                start = 16 * j
                sub = x[start:start + 16]
                weights = np.abs(sub)
                scales[j], mins[j], L[start:start + 16] = _make_qkx2_quants(
                    sub, weights, 3, -0.5, 0.1, 15, use_mad=True
                )
                if scales[j] > max_scale:
                    max_scale = scales[j]
                if mins[j] > max_min:
                    max_min = mins[j]

            q4scale = np.float32(15.0)
            packed_scales = out[i, : QK_K // 16]
            if max_scale > 0:
                iscale = q4scale / max_scale
                for j in range(QK_K // 16):
                    packed_scales[j] = int(_nearest_int(iscale * scales[j]).item())
                d = np.array([max_scale / q4scale], dtype=np.float16)
            else:
                d = np.array([0.0], dtype=np.float16)

            if max_min > 0:
                iscale = q4scale / max_min
                for j in range(QK_K // 16):
                    packed_scales[j] |= int(_nearest_int(iscale * mins[j]).item()) << 4
                dmin = np.array([max_min / q4scale], dtype=np.float16)
            else:
                dmin = np.array([0.0], dtype=np.float16)

            d_f32 = d.astype(np.float32)[0]
            dmin_f32 = dmin.astype(np.float32)[0]
            out[i, -4:-2] = d.view(np.uint8)
            out[i, -2:] = dmin.view(np.uint8)

            for j in range(QK_K // 16):
                d_sub = np.float32(d_f32 * (packed_scales[j] & np.uint8(0x0F)))
                if d_sub == 0:
                    continue
                dm_sub = np.float32(dmin_f32 * (packed_scales[j] >> np.uint8(4)))
                start = 16 * j
                l = _nearest_int((x[start:start + 16] + dm_sub) / d_sub).clip(0, 3)
                L[start:start + 16] = l.astype(np.uint8)

            qs = out[i, QK_K // 16 : QK_K // 16 + QK_K // 4]
            for j in range(0, QK_K, 128):
                for l in range(32):
                    qs[j // 4 + l] = (
                        L[j + l]
                        | (L[j + l + 32] << np.uint8(2))
                        | (L[j + l + 64] << np.uint8(4))
                        | (L[j + l + 96] << np.uint8(6))
                    )

        return out

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        scales, rest = np.hsplit(blocks, [QK_K // 16])
        qs, rest = np.hsplit(rest, [QK_K // 4])
        d, dmin = np.hsplit(rest, [2])

        d = d.view(np.float16).astype(np.float32)
        dmin = dmin.view(np.float16).astype(np.float32)

        # (n_blocks, 16, 1)
        dl = (d * (scales & 0xF).astype(np.float32)).reshape((n_blocks, QK_K // 16, 1))
        ml = (dmin * (scales >> 4).astype(np.float32)).reshape((n_blocks, QK_K // 16, 1))

        shift = np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))

        qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & np.uint8(3)

        qs = qs.reshape((n_blocks, QK_K // 16, 16)).astype(np.float32)

        qs = dl * qs - ml

        return qs.reshape((n_blocks, -1))


class Q3_K(__Quant, qtype=GGMLQuantizationType.Q3_K):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        blocks = blocks.astype(np.float32, copy=False)
        out = np.zeros((n_blocks, cls.type_size), dtype=np.uint8)

        for i, x in enumerate(blocks):
            L = np.zeros(QK_K, dtype=np.int8)
            scales = np.zeros(QK_K // 16, dtype=np.float32)
            max_scale = np.float32(0.0)
            amax = np.float32(0.0)

            for j in range(QK_K // 16):
                start = 16 * j
                sub = x[start:start + 16]
                scales[j], L[start:start + 16] = _make_q3_quants(sub, 4)
                scale_abs = np.abs(scales[j])
                if scale_abs > amax:
                    amax = scale_abs
                    max_scale = scales[j]

            if max_scale != 0:
                iscale = np.float32(-32.0) / max_scale
                Ls = (_nearest_int(iscale * scales).clip(-32, 31) + 32).astype(np.int8)
                d = np.array([np.float32(1.0) / iscale], dtype=np.float16)
            else:
                Ls = np.zeros(QK_K // 16, dtype=np.int8)
                d = np.array([0.0], dtype=np.float16)

            scales_packed = out[i, QK_K // 8 + QK_K // 4 : QK_K // 8 + QK_K // 4 + 12]
            for j in range(QK_K // 16):
                l = int(Ls[j])
                if j < 8:
                    scales_packed[j] = l & 0x0F
                else:
                    scales_packed[j - 8] |= (l & 0x0F) << 4
                l >>= 4
                scales_packed[j % 4 + 8] |= l << (2 * (j // 4))

            d_f32 = d.astype(np.float32)[0]
            out[i, -2:] = d.view(np.uint8)

            for j in range(QK_K // 16):
                sc = int(Ls[j]) - 32
                d_sub = np.float32(d_f32 * sc)
                if d_sub == 0:
                    continue
                start = 16 * j
                l = _nearest_int(x[start:start + 16] / d_sub).clip(-4, 3).astype(np.int8)
                L[start:start + 16] = l + np.int8(4)

            hmask = out[i, : QK_K // 8]
            m = 0
            hm = 1
            for j in range(QK_K):
                if L[j] > 3:
                    hmask[m] |= hm
                    L[j] -= np.int8(4)
                m += 1
                if m == QK_K // 8:
                    m = 0
                    hm <<= 1

            qs = out[i, QK_K // 8 : QK_K // 8 + QK_K // 4]
            for j in range(0, QK_K, 128):
                for l in range(32):
                    qs[j // 4 + l] = (
                        (int(L[j + l]) & 0x03)
                        | ((int(L[j + l + 32]) & 0x03) << 2)
                        | ((int(L[j + l + 64]) & 0x03) << 4)
                        | ((int(L[j + l + 96]) & 0x03) << 6)
                    )

        return out

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        hmask, rest = np.hsplit(blocks, [QK_K // 8])
        qs, rest = np.hsplit(rest, [QK_K // 4])
        scales, d = np.hsplit(rest, [12])

        d = d.view(np.float16).astype(np.float32)

        # The scales are packed at 6-bit each in this pattern:
        #  0: IIIIAAAA
        #  1: JJJJBBBB
        #  2: KKKKCCCC
        #  3: LLLLDDDD
        #  4: MMMMEEEE
        #  5: NNNNFFFF
        #  6: OOOOGGGG
        #  7: PPPPHHHH
        #  8: MMIIEEAA
        #  9: NNJJFFBB
        # 10: OOKKGGCC
        # 11: PPLLHHDD
        lscales, hscales = np.hsplit(scales, [8])
        lscales = lscales.reshape((n_blocks, 1, 8)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 2, 1))
        lscales = lscales.reshape((n_blocks, 16))
        hscales = hscales.reshape((n_blocks, 1, 4)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 4, 1))
        hscales = hscales.reshape((n_blocks, 16))
        scales = (lscales & np.uint8(0x0F)) | ((hscales & np.uint8(0x03)) << np.uint8(4))
        scales = (scales.astype(np.int8) - np.int8(32)).astype(np.float32)

        dl = (d * scales).reshape((n_blocks, 16, 1))

        ql = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
        qh = hmask.reshape(n_blocks, -1, 1, 32) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 8, 1))
        ql = ql.reshape((n_blocks, 16, QK_K // 16)) & np.uint8(3)
        qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & np.uint8(1))
        qh = qh ^ np.uint8(1)  # strangely, the offset is zero when the bitmask is 1
        q = (ql.astype(np.int8) - (qh << np.uint8(2)).astype(np.int8)).astype(np.float32)

        return (dl * q).reshape((n_blocks, QK_K))


class Q4_K(__Quant, qtype=GGMLQuantizationType.Q4_K):
    K_SCALE_SIZE = 12

    @staticmethod
    def get_scale_min(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_blocks = scales.shape[0]
        scales = scales.view(np.uint8)
        ### Unpacking the following: ###
        #  0 EEAAAAAA
        #  1 FFBBBBBB
        #  2 GGCCCCCC
        #  3 HHDDDDDD
        #  4 eeaaaaaa
        #  5 ffbbbbbb
        #  6 ggcccccc
        #  7 hhdddddd
        #  8 eeeeEEEE
        #  9 ffffFFFF
        # 10 ggggGGGG
        # 11 hhhhHHHH
        scales = scales.reshape((n_blocks, 3, 4))
        d, m, m_d = np.split(scales, 3, axis=-2)

        sc = np.concatenate([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
        min = np.concatenate([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], axis=-1)

        return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))

    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        return _quantize_q4_k_blocks(blocks, cls.type_size)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        dmin, rest = np.hsplit(rest, [2])
        scales, qs = np.hsplit(rest, [cls.K_SCALE_SIZE])

        d = d.view(np.float16).astype(np.float32)
        dmin = dmin.view(np.float16).astype(np.float32)

        sc, m = Q4_K.get_scale_min(scales)

        d = (d * sc.astype(np.float32)).reshape((n_blocks, -1, 1))
        dm = (dmin * m.astype(np.float32)).reshape((n_blocks, -1, 1))

        qs = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1, 32)).astype(np.float32)

        return (d * qs - dm).reshape((n_blocks, QK_K))


class Q5_K(__Quant, qtype=GGMLQuantizationType.Q5_K):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        blocks = blocks.astype(np.float32, copy=False)
        out = np.zeros((n_blocks, cls.type_size), dtype=np.uint8)

        for i, x in enumerate(blocks):
            L = np.zeros(QK_K, dtype=np.uint8)
            scales = np.zeros(QK_K // 32, dtype=np.float32)
            mins = np.zeros(QK_K // 32, dtype=np.float32)
            max_scale = np.float32(0.0)
            max_min = np.float32(0.0)

            for j in range(QK_K // 32):
                start = 32 * j
                sub = x[start:start + 32]
                sum_x2 = np.float32(0.0)
                for value in sub:
                    sum_x2 = np.float32(sum_x2 + np.float32(value * value))
                av_x = np.float32(np.sqrt(np.float32(sum_x2 / np.float32(32.0))))
                weights = av_x + np.abs(sub)
                scales[j], mins[j], L[start:start + 32] = _make_qkx2_quants(
                    sub, weights, 31, -0.5, 0.1, 15
                )
                if scales[j] > max_scale:
                    max_scale = scales[j]
                if mins[j] > max_min:
                    max_min = mins[j]

            inv_scale = np.float32(63.0) / max_scale if max_scale > 0 else np.float32(0.0)
            inv_min = np.float32(63.0) / max_min if max_min > 0 else np.float32(0.0)
            ls = np.zeros(QK_K // 32, dtype=np.uint8)
            lm = np.zeros(QK_K // 32, dtype=np.uint8)
            scales_packed = out[i, 4:16]
            for j in range(QK_K // 32):
                ls_j = min(63, int(_nearest_int(inv_scale * scales[j]).item()))
                lm_j = min(63, int(_nearest_int(inv_min * mins[j]).item()))
                ls[j] = ls_j
                lm[j] = lm_j
                if j < 4:
                    scales_packed[j] = ls_j
                    scales_packed[j + 4] = lm_j
                else:
                    scales_packed[j + 4] = (ls_j & 0x0F) | ((lm_j & 0x0F) << 4)
                    scales_packed[j - 4] |= (ls_j >> 4) << 6
                    scales_packed[j] |= (lm_j >> 4) << 6

            d = np.array([max_scale / np.float32(63.0)], dtype=np.float16)
            dmin = np.array([max_min / np.float32(63.0)], dtype=np.float16)
            out[i, :2] = d.view(np.uint8)
            out[i, 2:4] = dmin.view(np.uint8)
            d_f32 = d.astype(np.float32)[0]
            dmin_f32 = dmin.astype(np.float32)[0]

            for j in range(QK_K // 32):
                d_sub = np.float32(d_f32 * ls[j])
                if d_sub == 0:
                    continue
                dm_sub = np.float32(dmin_f32 * lm[j])
                start = 32 * j
                l = _nearest_int((x[start:start + 32] + dm_sub) / d_sub).clip(0, 31)
                L[start:start + 32] = l.astype(np.uint8)

            qh = out[i, 16 : 16 + QK_K // 8]
            ql = out[i, 16 + QK_K // 8 :]
            m1 = 1
            m2 = 2
            for n in range(0, QK_K, 64):
                ql_base = n // 2
                for j in range(32):
                    l1 = int(L[n + j])
                    if l1 > 15:
                        l1 -= 16
                        qh[j] |= m1
                    l2 = int(L[n + j + 32])
                    if l2 > 15:
                        l2 -= 16
                        qh[j] |= m2
                    ql[ql_base + j] = l1 | (l2 << 4)
                m1 <<= 2
                m2 <<= 2

        return out

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        dmin, rest = np.hsplit(rest, [2])
        scales, rest = np.hsplit(rest, [Q4_K.K_SCALE_SIZE])
        qh, qs = np.hsplit(rest, [QK_K // 8])

        d = d.view(np.float16).astype(np.float32)
        dmin = dmin.view(np.float16).astype(np.float32)

        sc, m = Q4_K.get_scale_min(scales)

        d = (d * sc.astype(np.float32)).reshape((n_blocks, -1, 1))
        dm = (dmin * m.astype(np.float32)).reshape((n_blocks, -1, 1))

        ql = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qh = qh.reshape((n_blocks, -1, 1, 32)) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 8, 1))
        ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1, 32))
        qh = (qh & np.uint8(0x01)).reshape((n_blocks, -1, 32))
        q = (ql | (qh << np.uint8(4))).astype(np.float32)

        return (d * q - dm).reshape((n_blocks, QK_K))


class Q6_K(__Quant, qtype=GGMLQuantizationType.Q6_K):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        blocks = blocks.astype(np.float32, copy=False)
        out = np.zeros((n_blocks, cls.type_size), dtype=np.uint8)

        for i, x in enumerate(blocks):
            L = np.zeros(QK_K, dtype=np.int8)
            scales = np.zeros(QK_K // 16, dtype=np.float32)
            max_scale = np.float32(0.0)
            max_abs_scale = np.float32(0.0)

            for ib in range(QK_K // 16):
                start = 16 * ib
                scale, quants = _make_qx_quants(x[start:start + 16], 32)
                L[start:start + 16] = quants
                scales[ib] = scale

                abs_scale = np.abs(scale)
                if abs_scale > max_abs_scale:
                    max_abs_scale = abs_scale
                    max_scale = scale

            if max_abs_scale < GROUP_MAX_EPS:
                continue

            iscale = np.float32(-128.0) / max_scale
            d = (np.float32(1.0) / iscale).astype(np.float16)
            d_f32 = d.astype(np.float32)
            q_scales = np.minimum(127, _nearest_int(iscale * scales)).astype(np.int8)

            for j in range(QK_K // 16):
                d_sub = np.float32(d_f32 * q_scales[j])
                if d_sub == 0:
                    continue
                start = 16 * j
                l = _nearest_int(x[start:start + 16] / d_sub).clip(-32, 31).astype(np.int8)
                L[start:start + 16] = l + np.int8(32)

            ql = out[i, : QK_K // 2]
            qh = out[i, QK_K // 2 : QK_K // 2 + QK_K // 4]
            for j in range(0, QK_K, 128):
                ql_base = j // 2
                qh_base = j // 4
                for l in range(32):
                    q1 = L[j + l + 0] & np.int8(0x0F)
                    q2 = L[j + l + 32] & np.int8(0x0F)
                    q3 = L[j + l + 64] & np.int8(0x0F)
                    q4 = L[j + l + 96] & np.int8(0x0F)
                    ql[ql_base + l + 0] = q1 | (q3 << np.int8(4))
                    ql[ql_base + l + 32] = q2 | (q4 << np.int8(4))
                    qh[qh_base + l] = (
                        (L[j + l] >> np.int8(4))
                        | ((L[j + l + 32] >> np.int8(4)) << np.int8(2))
                        | ((L[j + l + 64] >> np.int8(4)) << np.int8(4))
                        | ((L[j + l + 96] >> np.int8(4)) << np.int8(6))
                    )

            scales_start = QK_K // 2 + QK_K // 4
            out[i, scales_start : scales_start + QK_K // 16] = q_scales.view(np.uint8)
            out[i, -2:] = np.array([d], dtype=np.float16).view(np.uint8)

        return out

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        ql, rest = np.hsplit(blocks, [QK_K // 2])
        qh, rest = np.hsplit(rest, [QK_K // 4])
        scales, d = np.hsplit(rest, [QK_K // 16])

        scales = scales.view(np.int8).astype(np.float32)
        d = d.view(np.float16).astype(np.float32)
        d = (d * scales).reshape((n_blocks, QK_K // 16, 1))

        ql = ql.reshape((n_blocks, -1, 1, 64)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1, 32))
        qh = qh.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
        qh = (qh & np.uint8(0x03)).reshape((n_blocks, -1, 32))
        q = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(32)
        q = q.reshape((n_blocks, QK_K // 16, -1)).astype(np.float32)

        return (d * q).reshape((n_blocks, QK_K))


class TQ1_0(__Quant, qtype=GGMLQuantizationType.TQ1_0):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d = abs(blocks).max(axis=-1, keepdims=True)
        with np.errstate(divide="ignore"):
            id = np.where(d == 0, 0, 1 / d)
        qs = np_roundf(blocks * id)
        qs = (qs.astype(np.int8) + np.int8(1)).astype(np.uint8)

        qs0, qs1, qh = qs[..., :(32 * 5)], qs[..., (32 * 5):(48 * 5)], qs[..., (48 * 5):]
        qs0 = qs0.reshape((n_blocks, -1, 5, 32)) * np.array([81, 27, 9, 3, 1], dtype=np.uint8).reshape((1, 1, 5, 1))
        qs0 = np.sum(qs0, axis=-2).reshape((n_blocks, -1))
        qs1 = qs1.reshape((n_blocks, -1, 5, 16)) * np.array([81, 27, 9, 3, 1], dtype=np.uint8).reshape((1, 1, 5, 1))
        qs1 = np.sum(qs1, axis=-2).reshape((n_blocks, -1))
        qh = qh.reshape((n_blocks, -1, 4, 4)) * np.array([81, 27, 9, 3], dtype=np.uint8).reshape((1, 1, 4, 1))
        qh = np.sum(qh, axis=-2).reshape((n_blocks, -1))
        qs = np.concatenate([qs0, qs1, qh], axis=-1)
        qs = (qs.astype(np.uint16) * 256 + (243 - 1)) // 243

        qs = qs.astype(np.uint8)
        d = d.astype(np.float16).view(np.uint8)

        return np.concatenate([qs, d], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        qs, rest = np.hsplit(blocks, [(QK_K - 4 * QK_K // 64) // 5])
        qh, d = np.hsplit(rest, [QK_K // 64])

        d = d.view(np.float16).astype(np.float32)

        qs0, qs1 = qs[..., :32], qs[..., 32:]
        qs0 = qs0.reshape((n_blocks, -1, 1, 32)) * np.array([1, 3, 9, 27, 81], dtype=np.uint8).reshape((1, 1, 5, 1))
        qs0 = qs0.reshape((n_blocks, -1))
        qs1 = qs1.reshape((n_blocks, -1, 1, 16)) * np.array([1, 3, 9, 27, 81], dtype=np.uint8).reshape((1, 1, 5, 1))
        qs1 = qs1.reshape((n_blocks, -1))
        qh = qh.reshape((n_blocks, -1, 1, 4)) * np.array([1, 3, 9, 27], dtype=np.uint8).reshape((1, 1, 4, 1))
        qh = qh.reshape((n_blocks, -1))
        qs = np.concatenate([qs0, qs1, qh], axis=-1)
        qs = ((qs.astype(np.uint16) * 3) >> 8).astype(np.int8) - np.int8(1)

        return (d * qs.astype(np.float32))


class TQ2_0(__Quant, qtype=GGMLQuantizationType.TQ2_0):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d = abs(blocks).max(axis=-1, keepdims=True)
        with np.errstate(divide="ignore"):
            id = np.where(d == 0, 0, 1 / d)
        qs = np_roundf(blocks * id)
        qs = (qs.astype(np.int8) + np.int8(1)).astype(np.uint8)

        qs = qs.reshape((n_blocks, -1, 4, 32)) << np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
        qs = qs[..., 0, :] | qs[..., 1, :] | qs[..., 2, :] | qs[..., 3, :]
        qs = qs.reshape((n_blocks, -1))

        d = d.astype(np.float16).view(np.uint8)

        return np.concatenate([qs, d], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        qs, d = np.hsplit(blocks, [QK_K // 4])

        d = d.view(np.float16).astype(np.float32)

        qs = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
        qs = (qs & 0x03).reshape((n_blocks, -1)).astype(np.int8) - np.int8(1)

        return (d * qs.astype(np.float32))


class MXFP4(__Quant, qtype=GGMLQuantizationType.MXFP4):
    # e2m1 values (doubled)
    # ref: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    kvalues = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)

    @staticmethod
    # see ggml_e8m0_to_fp32_half in ggml-impl.h
    def e8m0_to_fp32_half(x: np.ndarray) -> np.ndarray:
        bits = np.where(x < 2, np.uint32(0x00200000) << np.uint32(x), np.uint32(x - 1) << np.uint32(23))
        return bits.view(np.float32)

    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d = abs(blocks).max(axis=-1, keepdims=True)

        with np.errstate(divide="ignore"):
            e = np.where(d > 0, np.floor(np.log2(d)) - 2 + 127, 0).astype(np.uint8)

        d = cls.e8m0_to_fp32_half(e)

        kvalues = np.array(cls.kvalues, dtype=np.int8).reshape((1, 1, 16))

        errs = np.abs(d.reshape((n_blocks, 1, 1)) * kvalues.astype(np.float32) - blocks.reshape((n_blocks, cls.block_size, 1)))
        best = np.argmin(errs, axis=-1, keepdims=True)

        qs = best.reshape(n_blocks, 2, cls.block_size // 2).astype(np.uint8)
        qs = qs[:, 0] | (qs[:, 1] << np.uint8(4))

        qs = qs.reshape((n_blocks, cls.block_size // 2))

        return np.concatenate([e, qs], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        e, qs = np.hsplit(blocks, [1])

        d = cls.e8m0_to_fp32_half(e)

        qs = qs.reshape((n_blocks, 1, cls.block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 2, 1))
        qs = (qs & np.uint8(0x0F)).view(np.int8)

        kvalues = np.array(cls.kvalues, dtype=np.int8).reshape(1, 1, 16)
        qs = np.take_along_axis(kvalues, qs, axis=-1).reshape((n_blocks, cls.block_size))

        return (d * qs.astype(np.float32))


class NVFP4(__Quant, qtype=GGMLQuantizationType.NVFP4):
    # E2M1 values doubled, matching the GGML reference kvalue convention.
    kvalues = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)

    @staticmethod
    def ue4m3_to_fp32(x: np.ndarray) -> np.ndarray:
        exp = (x >> np.uint8(3)).astype(np.int32) & 0xF
        man = (x & np.uint8(0x07)).astype(np.float32)
        raw = np.where(
            exp == 0,
            man * np.float32(2**-9),
            (1.0 + man / 8.0) * np.float32(2.0) ** (exp.astype(np.float32) - 7.0),
        )
        return np.where((x == 0) | (x == 0x7F), 0.0, raw * 0.5)

    @staticmethod
    def fp32_to_ue4m3(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, 448.0).astype(np.float32)
        bits = x.view(np.uint32)
        fp32_exp = ((bits >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int32) - 127
        fp32_man = ((bits >> np.uint32(20)) & np.uint32(0x07)).astype(np.int32)
        ue4m3_exp = fp32_exp + 7

        sub_man = np.clip((x * 512.0 + 0.5).astype(np.int32), 0, 7)
        sub_result = np.where(sub_man >= 1, sub_man, 0).astype(np.uint8)

        round_bit = ((bits >> np.uint32(19)) & np.uint32(1)).astype(np.int32)
        man = fp32_man + round_bit
        exp = ue4m3_exp.copy()
        overflow = man > 7
        man = np.where(overflow, 0, man)
        exp = np.where(overflow, exp + 1, exp)
        normal_result = np.where(
            exp >= 15,
            np.uint8(0x7E),
            ((exp << 3) | man).astype(np.uint8),
        )

        return np.where(
            x <= 0.0,
            np.uint8(0),
            np.where(
                ue4m3_exp <= 0,
                sub_result,
                np.where(ue4m3_exp >= 15, np.uint8(0x7E), normal_result),
            ),
        )

    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_super = blocks.shape[0]

        blocks = blocks.reshape((n_super, 4, 16))
        d = abs(blocks).max(axis=-1) / np.float32(6.0)
        d_bytes = cls.fp32_to_ue4m3(d)
        d_fp = cls.ue4m3_to_fp32(d_bytes).reshape((n_super, 4, 1))

        kvalues = np.array(cls.kvalues, dtype=np.int8).reshape((1, 1, 16))
        errs = np.abs(
            d_fp.reshape((n_super, 4, 1, 1))
            * kvalues.astype(np.float32).reshape((1, 1, 1, 16))
            - blocks.reshape((n_super, 4, 16, 1))
        )
        best = np.argmin(errs, axis=-1).astype(np.uint8)

        lo = best[..., :8]
        hi = best[..., 8:] << np.uint8(4)
        qs = (lo | hi).reshape((n_super, 32))

        return np.concatenate([d_bytes, qs], axis=-1)

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_super = blocks.shape[0]

        d_bytes, qs = np.hsplit(blocks, [4])
        d = cls.ue4m3_to_fp32(d_bytes).reshape((n_super, 4, 1))

        qs = qs.reshape((n_super, 4, 8))
        lo = (qs & np.uint8(0x0F)).view(np.int8)
        hi = (qs >> np.uint8(4)).view(np.int8)
        vals = np.concatenate([lo, hi], axis=-1)

        kvalues = np.array(cls.kvalues, dtype=np.int8).reshape((1, 1, 16))
        vals = np.take_along_axis(kvalues, vals, axis=-1)

        return (d * vals.astype(np.float32)).reshape((n_super, cls.block_size))


class IQ2_XXS(__Quant, qtype=GGMLQuantizationType.IQ2_XXS):
    ksigns: bytes = (
        b"\x00\x81\x82\x03\x84\x05\x06\x87\x88\x09\x0a\x8b\x0c\x8d\x8e\x0f"
        b"\x90\x11\x12\x93\x14\x95\x96\x17\x18\x99\x9a\x1b\x9c\x1d\x1e\x9f"
        b"\xa0\x21\x22\xa3\x24\xa5\xa6\x27\x28\xa9\xaa\x2b\xac\x2d\x2e\xaf"
        b"\x30\xb1\xb2\x33\xb4\x35\x36\xb7\xb8\x39\x3a\xbb\x3c\xbd\xbe\x3f"
        b"\xc0\x41\x42\xc3\x44\xc5\xc6\x47\x48\xc9\xca\x4b\xcc\x4d\x4e\xcf"
        b"\x50\xd1\xd2\x53\xd4\x55\x56\xd7\xd8\x59\x5a\xdb\x5c\xdd\xde\x5f"
        b"\x60\xe1\xe2\x63\xe4\x65\x66\xe7\xe8\x69\x6a\xeb\x6c\xed\xee\x6f"
        b"\xf0\x71\x72\xf3\x74\xf5\xf6\x77\x78\xf9\xfa\x7b\xfc\x7d\x7e\xff"
    )

    # iq2xxs_grid, but with each byte of the original packed in 2 bits,
    # by mapping 0x08 to 0, 0x19 to 1, and 0x2b to 2.
    grid_shape = (256, 8)
    grid_map = (0x08, 0x19, 0x2b)
    grid_hex = (
        b"00000200050008000a00110014002000220028002a0041004400500058006100"
        b"6400800082008a00a20001010401100115014001840198010002020222028202"
        b"010404041004210424044004420448046004810484049004a404000502050805"
        b"200546056905800591050906100640068406a406000805080808140828084108"
        b"440850085208880804094009020a140a01100410101021104010601084109010"
        b"951000110811201150115a118011241245120014081420142514491480141815"
        b"6215001616160118041810184018811800190519a019511a002002200a204420"
        b"6120802082202921482100220222012404241024402456240025412564259026"
        b"082820289428442a014004401040184021402440404048405640604081408440"
        b"9040004120416141804185410142104248425642684200440844204480449944"
        b"124524450046014804481048404845480049584961498249454a904a00500850"
        b"1150195020508050885004514251a4519152905492540a550156545600581158"
        b"195864584059085a046010604060686000615561186260620064056410651265"
        b"84654268008002800a8041808280048118814081118201840484108415844084"
        b"608400854685948509864086608602880489118a0490109024904090a1901691"
        b"8091459200942294449451958198209902a050a085a009a100a218a450a804a9"
    )

    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        blocks = blocks.astype(np.float32, copy=False)
        out = np.zeros((n_blocks, cls.type_size), dtype=np.uint8)
        kmap, neighbours, grid_l = _get_iq2_xxs_lookup()
        ksigns = np.frombuffer(cls.ksigns, dtype=np.uint8)
        quant_weights = np.sum(blocks * blocks, axis=0, dtype=np.float32)

        for i, x in enumerate(blocks):
            q2 = np.zeros(QK_K // 16, dtype=np.uint32)
            scales = np.zeros(QK_K // 32, dtype=np.float32)
            max_scale = np.float32(0.0)
            sigma2 = np.float32(np.sum(x * x, dtype=np.float32) / np.float32(QK_K))

            for ib in range(QK_K // 32):
                xb = x[32 * ib : 32 * ib + 32]
                qw = quant_weights[32 * ib : 32 * ib + 32]
                weight = np.empty(32, dtype=np.float32)
                for j in range(32):
                    weight[j] = np.float32(qw[j] * np.float32(np.sqrt(np.float32(sigma2 + np.float32(xb[j] * xb[j])))))
                waux = np.sqrt(weight, dtype=np.float32)
                xval = np.empty(32, dtype=np.float32)
                block_signs = np.zeros(4, dtype=np.uint8)

                for k in range(4):
                    nflip = 0
                    sign = 0
                    for j in range(8):
                        index = 8 * k + j
                        if xb[index] >= 0:
                            xval[index] = xb[index]
                        else:
                            xval[index] = -xb[index]
                            nflip += 1
                            sign |= 1 << j
                    if nflip % 2:
                        imin = 8 * k
                        min_v = np.float32(weight[imin] * xb[imin] * xb[imin])
                        for j in range(1, 8):
                            index = 8 * k + j
                            ax = np.float32(weight[index] * xb[index] * xb[index])
                            if ax < min_v:
                                min_v = ax
                                imin = index
                        xval[imin] = -xval[imin]
                        sign ^= 1 << (imin - 8 * k)
                    block_signs[k] = sign & 0x7F

                if np.max(xval) < GROUP_MAX_EPS:
                    continue

                scale, L = _make_qp_quants(xval, 4, weight)
                eff_max = np.float32(scale * np.float32(3.0))
                if eff_max <= 0:
                    continue

                best = np.float32(0.0)
                for is_ in range(-6, 7):
                    id_ = (np.float32(5.0) + np.float32(is_) * np.float32(0.1)) / eff_max
                    this_scale = np.float32(1.0) / id_
                    Laux = np.zeros(32, dtype=np.int8)
                    for k in range(4):
                        u = 0
                        for j in range(8):
                            index = 8 * k + j
                            l = int(_nearest_int(np.float32(0.5) * np.float32(id_ * xval[index] - np.float32(1.0))).item())
                            l = max(0, min(2, l))
                            Laux[index] = l
                            u |= l << (2 * j)
                        if kmap[u] < 0:
                            _, Laux[8 * k : 8 * k + 8] = _best_lattice_neighbour(
                                neighbours[u],
                                grid_l,
                                xval[8 * k : 8 * k + 8],
                                waux[8 * k : 8 * k + 8],
                                this_scale,
                            )

                    sumqx = np.float32(0.0)
                    sumq2 = np.float32(0.0)
                    for j in range(32):
                        q = np.float32(2 * int(Laux[j]) + 1)
                        w = np.float32(weight[j])
                        sumqx = np.float32(sumqx + np.float32(w * xval[j] * q))
                        sumq2 = np.float32(sumq2 + np.float32(w * q * q))
                    if sumq2 > 0 and sumqx * sumqx > best * sumq2:
                        scale = np.float32(sumqx / sumq2)
                        best = np.float32(scale * sumqx)
                        L[:] = Laux

                if scale > 0:
                    id_ = np.float32(1.0) / scale
                    for k in range(4):
                        u = 0
                        for j in range(8):
                            index = 8 * k + j
                            l = int(_nearest_int(np.float32(0.5) * np.float32(id_ * xval[index] - np.float32(1.0))).item())
                            l = max(0, min(2, l))
                            L[index] = l
                            u |= l << (2 * j)
                        if kmap[u] < 0:
                            _, L[8 * k : 8 * k + 8] = _best_lattice_neighbour(
                                neighbours[u],
                                grid_l,
                                xval[8 * k : 8 * k + 8],
                                waux[8 * k : 8 * k + 8],
                                scale,
                            )

                    sumqx = np.float32(0.0)
                    sumq2 = np.float32(0.0)
                    for j in range(32):
                        q = np.float32(2 * int(L[j]) + 1)
                        w = np.float32(weight[j])
                        sumqx = np.float32(sumqx + np.float32(w * xval[j] * q))
                        sumq2 = np.float32(sumq2 + np.float32(w * q * q))
                    if sumq2 > 0:
                        scale = np.float32(sumqx / sumq2)

                if scale < 0:
                    scale = np.float32(-scale)
                    for k in range(4):
                        block_signs[k] = (~block_signs[k]) & 0x7F

                for k in range(4):
                    u = 0
                    for j in range(8):
                        u |= int(L[8 * k + j]) << (2 * j)
                    q2[2 * ib] |= np.uint32(int(kmap[u]) << (8 * k))
                    q2[2 * ib + 1] |= np.uint32(int(block_signs[k]) << (7 * k))

                scales[ib] = scale
                if scale > max_scale:
                    max_scale = scale

            if max_scale == 0:
                continue

            d = np.float32(max_scale / np.float32(31.0))
            out[i, :2] = np.array([d], dtype=np.float16).view(np.uint8)
            id_ = np.float32(1.0) / d
            for ib in range(QK_K // 32):
                l = int(_nearest_int(np.float32(0.5) * np.float32(id_ * scales[ib] - np.float32(1.0))).item())
                l = max(0, min(15, l))
                q2[2 * ib + 1] |= np.uint32(l << 28)

            out[i, 2:] = q2.view(np.uint8)

        return out

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, qs = np.hsplit(blocks, [2])

        d = d.view(np.float16).astype(np.float32)

        qs = qs.view(np.uint32).reshape(n_blocks, -1, 2)

        db = d * (np.float32(0.5) + (qs[..., 1] >> 28).astype(np.float32)) * np.float32(0.25)
        db = db.reshape((n_blocks, -1, 1, 1))

        # get the sign indices and unpack the bits
        signs = qs[..., 1].reshape((n_blocks, -1, 1)) >> np.array([0, 7, 14, 21], dtype=np.uint32).reshape((1, 1, 4))
        ksigns = np.frombuffer(cls.ksigns, dtype=np.uint8).reshape((1, 1, 1, 128))
        signs = (signs & np.uint32(0x7F)).reshape((n_blocks, -1, 4, 1))
        signs = np.take_along_axis(ksigns, signs, axis=-1)
        signs = signs.reshape((n_blocks, -1, 4, 1)) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 1, 8))
        signs = signs & np.uint8(0x01)
        signs = np.where(signs == 0, np.float32(1), np.float32(-1))
        signs = signs.reshape((n_blocks, -1, 4, 8))

        assert cls.grid is not None
        grid = np.take_along_axis(cls.grid, qs[..., 0].copy().view(np.uint8).reshape((n_blocks, -1, 1, 1)), axis=-2)
        grid = grid.reshape((n_blocks, -1, 4, 8))

        return (db * grid * signs).reshape((n_blocks, -1))


class IQ2_XS(__Quant, qtype=GGMLQuantizationType.IQ2_XS):
    # iq2xs_grid, but with each byte of the original packed in 2 bits,
    # by mapping 0x08 to 0, 0x19 to 1, and 0x2b to 2.
    grid_shape = (512, 8)
    grid_map = (0x08, 0x19, 0x2b)
    grid_hex = (
        b"00000200050008000a0011001400160019002000220025002800410044004600"
        b"49005000520055005800610064008000820085008800910094009900a0000101"
        b"04010601090110011201150118011a0121012401400142014501480151015401"
        b"6001680181018401900100020202050208021102140220024102440250025502"
        b"80028a0201040404060409041004120415041804210424044004420445044804"
        b"5104540456046004810484049004000502050505080511051405200541054405"
        b"500561058005010604061006260640064206840600080208050808080a081108"
        b"14082008250841084408500858088008a008aa08010904091009400981098909"
        b"000a200a280a960aa00a01100410061009101010121015101810211024104010"
        b"4210451048105110541060106a10811084109010001102110511081111111411"
        b"2011411144115011801194119611011204120612101240126012001402140514"
        b"0814111414142014411444144914501464148014011504151015401500161416"
        b"49160118041810181218401854188618001905196619511aa91a002002200520"
        b"08200a201120142020204120442050208020a020012104211021402148216521"
        b"002222228022a82201240424102429244024002541255225992501261a26a626"
        b"002808280a28202855288828a22868299029082a202a822a882a8a2a01400440"
        b"0640094010401240154018402140244040404240454048404a40514054406040"
        b"6540814084409040004102410541084111411441204141414441504180418541"
        b"a241014204421042124229424042004402440544084411441444194420444144"
        b"4444504480449444014504451045244540459a4500460a464446504601480448"
        b"1048404845485448624800491149444950496949044a00500250055008501150"
        b"145020502850415044505050805001510451105115514051425100524452aa52"
        b"0154045410542154405460548154a154005508558055885521566856a1560058"
        b"14584158505899581a5940594259855a0160046010604060546062608660a960"
        b"006124624a62926200641664106540654565a46501686a682569066a546a626a"
        b"00800280058008801180148020802a8041804480508080808280a880aa800181"
        b"0481068110814081518159810082208280828282a082a8820184048410841284"
        b"158440846084898400854485a58518866a860088088825885a8880888288a888"
        b"0689228a808a888a968aa88a0190049010904090569084900091229164915692"
        b"89920094059444945094589429959095929541965198a6984999159a609a00a0"
        b"02a008a00aa020a02aa0a0a051a159a1a6a100a202a208a22aa280a2a0a240a4"
        b"95a465a698a60aa820a822a828a8a0a8a8a804a984a986a928aa2aaa91aaaaaa"
    )

    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        blocks = blocks.astype(np.float32, copy=False)
        out = np.zeros((n_blocks, cls.type_size), dtype=np.uint8)
        kmap, neighbours, grid_l = _get_iq2_xs_lookup()
        quant_weights = np.sum(blocks * blocks, axis=0, dtype=np.float32)

        for i, x in enumerate(blocks):
            q2 = np.zeros(2 * (QK_K // 16), dtype=np.uint16)
            scales = np.zeros(QK_K // 16, dtype=np.float32)
            max_scale = np.float32(0.0)
            sigma2 = np.float32(np.sum(x * x, dtype=np.float32) / np.float32(QK_K))

            for ib in range(QK_K // 16):
                xb = x[16 * ib : 16 * ib + 16]
                qw = quant_weights[16 * ib : 16 * ib + 16]
                weight = np.empty(16, dtype=np.float32)
                for j in range(16):
                    weight[j] = np.float32(qw[j] * np.float32(np.sqrt(np.float32(sigma2 + np.float32(xb[j] * xb[j])))))
                waux = np.sqrt(weight, dtype=np.float32)
                xval = np.empty(16, dtype=np.float32)
                block_signs = np.zeros(2, dtype=np.uint8)

                for k in range(2):
                    nflip = 0
                    sign = 0
                    for j in range(8):
                        index = 8 * k + j
                        if xb[index] >= 0:
                            xval[index] = xb[index]
                        else:
                            xval[index] = -xb[index]
                            nflip += 1
                            sign |= 1 << j
                    if nflip % 2:
                        imin = 8 * k
                        min_v = np.float32(weight[imin] * xb[imin] * xb[imin])
                        for j in range(1, 8):
                            index = 8 * k + j
                            ax = np.float32(weight[index] * xb[index] * xb[index])
                            if ax < min_v:
                                min_v = ax
                                imin = index
                        xval[imin] = -xval[imin]
                        sign ^= 1 << (imin - 8 * k)
                    block_signs[k] = sign & 0x7F

                L = np.zeros(16, dtype=np.int8)
                max_v = np.float32(np.max(xval))
                if max_v < GROUP_MAX_EPS:
                    continue

                best = np.float32(0.0)
                scale = np.float32(max_v / np.float32(5.0))
                is_on_grid = np.ones(2, dtype=bool)
                for is_ in range(-9, 10):
                    id_ = (np.float32(5.0) + np.float32(is_) * np.float32(0.1)) / max_v
                    this_scale = np.float32(1.0) / id_
                    Laux = np.zeros(16, dtype=np.int8)
                    is_on_grid_aux = np.ones(2, dtype=bool)
                    for k in range(2):
                        u = 0
                        for j in range(8):
                            index = 8 * k + j
                            l = int(_nearest_int(np.float32(0.5) * np.float32(id_ * xval[index] - np.float32(1.0))).item())
                            l = max(0, min(2, l))
                            Laux[index] = l
                            u |= l << (2 * j)
                        if kmap[u] < 0:
                            is_on_grid_aux[k] = False
                            _, Laux[8 * k : 8 * k + 8] = _best_lattice_neighbour(
                                neighbours[u],
                                grid_l,
                                xval[8 * k : 8 * k + 8],
                                waux[8 * k : 8 * k + 8],
                                this_scale,
                            )

                    sumqx = np.float32(0.0)
                    sumq2 = np.float32(0.0)
                    for j in range(16):
                        q = np.float32(2 * int(Laux[j]) + 1)
                        w = np.float32(weight[j])
                        sumqx = np.float32(sumqx + np.float32(w * xval[j] * q))
                        sumq2 = np.float32(sumq2 + np.float32(w * q * q))
                    if sumq2 > 0 and sumqx * sumqx > best * sumq2:
                        scale = np.float32(sumqx / sumq2)
                        best = np.float32(scale * sumqx)
                        L[:] = Laux
                        is_on_grid[:] = is_on_grid_aux

                if np.any(~is_on_grid) and scale > 0:
                    id_ = np.float32(1.0) / scale
                    for k in range(2):
                        if is_on_grid[k]:
                            continue
                        u = 0
                        for j in range(8):
                            index = 8 * k + j
                            l = int(_nearest_int(np.float32(0.5) * np.float32(id_ * xval[index] - np.float32(1.0))).item())
                            l = max(0, min(2, l))
                            L[index] = l
                            u |= l << (2 * j)
                        if kmap[u] < 0:
                            _, L[8 * k : 8 * k + 8] = _best_lattice_neighbour(
                                neighbours[u],
                                grid_l,
                                xval[8 * k : 8 * k + 8],
                                waux[8 * k : 8 * k + 8],
                                scale,
                            )

                    sumqx = np.float32(0.0)
                    sumq2 = np.float32(0.0)
                    for j in range(16):
                        q = np.float32(2 * int(L[j]) + 1)
                        w = np.float32(weight[j])
                        sumqx = np.float32(sumqx + np.float32(w * xval[j] * q))
                        sumq2 = np.float32(sumq2 + np.float32(w * q * q))
                    if sumq2 > 0:
                        scale = np.float32(sumqx / sumq2)

                if scale < 0:
                    scale = np.float32(-scale)
                    for k in range(2):
                        block_signs[k] = (~block_signs[k]) & 0x7F

                for k in range(2):
                    u = 0
                    for j in range(8):
                        u |= int(L[8 * k + j]) << (2 * j)
                    q2[2 * ib + k] = np.uint16(int(kmap[u]) | (int(block_signs[k]) << 9))

                scales[ib] = scale
                if scale > max_scale:
                    max_scale = scale

            if max_scale == 0:
                continue

            d = np.float32(max_scale / np.float32(31.0))
            out[i, :2] = np.array([d], dtype=np.float16).view(np.uint8)
            id_ = np.float32(1.0) / d
            packed_scales = out[i, 2 + QK_K // 4 :]
            for ib in range(QK_K // 16):
                l = int(_nearest_int(np.float32(0.5) * np.float32(id_ * scales[ib] - np.float32(1.0))).item())
                l = max(0, min(15, l))
                if ib % 2 == 0:
                    packed_scales[ib // 2] = l
                else:
                    packed_scales[ib // 2] |= l << 4

            out[i, 2 : 2 + QK_K // 4] = q2.view(np.uint8)

        return out

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        qs, scales = np.hsplit(rest, [2 * QK_K // 8])

        d = d.view(np.float16).astype(np.float32)
        qs = qs.view(np.uint16)

        scales = scales.reshape((n_blocks, -1, 1)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2))
        scales = (scales & 0x0F).reshape((n_blocks, -1))
        db = d * (np.float32(0.5) + scales) * np.float32(0.25)
        db = db.reshape((n_blocks, -1, 1, 1))

        # get the sign indices and unpack the bits
        signs = np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8).reshape(1, 1, 128)
        signs = np.take_along_axis(signs, (qs >> 9).reshape((n_blocks, -1, 1)), axis=-1)
        signs = signs.reshape((n_blocks, -1, 1)) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 8))
        signs = signs & np.uint8(0x01)
        signs = np.where(signs == 0, np.float32(1), np.float32(-1))
        signs = signs.reshape((n_blocks, -1, 2, 8))

        assert cls.grid is not None
        grid = np.take_along_axis(cls.grid, (qs & np.uint16(511)).reshape((n_blocks, -1, 1, 1)), axis=-2)
        grid = grid.reshape((n_blocks, -1, 2, 8))

        return (db * grid * signs).reshape((n_blocks, -1))


class IQ2_S(__Quant, qtype=GGMLQuantizationType.IQ2_S):
    # iq2s_grid, but with each byte of the original packed in 2 bits,
    # by mapping 0x08 to 0, 0x19 to 1, and 0x2b to 2.
    grid_shape = (1024, 8)
    grid_map = (0x08, 0x19, 0x2b)
    grid_hex = (
        b"00000200050008000a0011001400160019002000220025002800410044004600"
        b"490050005200550058006100640066006900800082008500880091009400a000"
        b"a500aa0001010401060109011001120115011801210124014001420145014801"
        b"510154015601590160016501680181018401900192019501a101a40100020202"
        b"050208021102140220022a02410244024602490250025502800285028a029402"
        b"a202010404040604090410041204150418042104240426042904400442044504"
        b"48044a0451045404560459046004620465048104840486048904900495049804"
        b"a104a40400050205050508050a05110514051605190520052505280541054405"
        b"46054905500552055505580561056405800582058505880591059405a0050106"
        b"0406060609061006150640064506480651065406600681068406900600080208"
        b"050808081108140816081908200825082a084108440846084908500852085508"
        b"580861086408800885089408aa08010904091009120915091809210940094509"
        b"480951095409600981099009000a110a140a220a280a2a0a500a990a01100410"
        b"0610091010101210151018102110241026104010421045104810511054105610"
        b"59106010621065106810811084108610901095109810a110a410001102110511"
        b"08110a1111111411161119112011221125112811411144114611491150115211"
        b"5511581161116411801182118511881191119411011204120912101215122112"
        b"2412401245125112541281128412901200140214051408141114141416141914"
        b"2014251428144114441446144914501452145514581461146414801482148514"
        b"881491149414a014011504150615091510151215151518152115241540154215"
        b"4515481551155415601581158415901500160516081611161416201641164416"
        b"50168016aa160118041806180918101815181818211840184218451848185118"
        b"541860188118841800190219051908191119141920194119441950196919a219"
        b"041a101a401a561a00200220052008201120142016201920202025202a204120"
        b"4420502052205520642080208a209420aa200121042110211221152121214021"
        b"4221452151215421602181218421902100220a22222228222a22442250228822"
        b"8a22a82201240424062409241024152418242124242440244224452448245124"
        b"5424602481248424902400250525082511251425202541254425502566258025"
        b"0126042610264026592600280528112814284128442850288a28aa2801290429"
        b"102995290a2a222a642a882a8a2a014004400640094010401240154018401a40"
        b"21402440264040404240454048404a4051405440564059406040624065408140"
        b"8440904095409840a140a4400041024105410841114114411641194120412241"
        b"2541414144414641494150415241554158416141644180418241854188419141"
        b"9441a04101420442104212421542184224424042454248425142544260428142"
        b"844200440244054408440a441144144416441944204422442544284441444444"
        b"46444944504452445544584461446444804482448544884491449444a0440145"
        b"0445064509451045124515451845214524454045424545454845514554456045"
        b"6a4581458445904500460246054608461146144620464146444650468046a546"
        b"0148044809481048124815481848214824484048424845484848514854486048"
        b"84489048004902490549084911491449204941494449504980499649014a044a"
        b"104a404a00500250055008501150145016501950205022502550285041504450"
        b"4650495050505250555058506150645080508250855088509150945001510451"
        b"0651095110511251155118512151245140514251455148515151545160518151"
        b"8451905100520552085211521452205241524452505269528052015404540654"
        b"0954105412541554185421542454405442544554485451545454605481548454"
        b"9054005502550555085511551455205541554455505580550156045610562656"
        b"405600580258055808581158145820584158445850585a588058015904591059"
        b"4059005a195a855aa85a01600460066010601260156018602160246040604560"
        b"4860516054606060846090600061026105610861116114612061416144615061"
        b"806199610462106240625662a162006405640864116414642064416444645064"
        b"806401650465106540654a656865926500669466016804681068656898680069"
        b"2a69426aa16a0080028005800880118014801980208025804180448050805280"
        b"5580588061808080858091809480018104810981108112811581188121812481"
        b"408142814581488151815481818184819081a981008205820a82118214824182"
        b"4482508201840484068409841084128415841884218440844284458448845184"
        b"5484608481848484908400850285058508851185148520854185448550858085"
        b"8a85018604861086298640860088058811881488418844885088a28801890489"
        b"40896589228a588a5a8a828aa28a019004900990109012901590189024904090"
        b"4290459048905190549060908190849090900091059111911491419144915091"
        b"5a910192049210924092a6920094029405940894119414942094419444945094"
        b"8094969401950495109540959895a19500964696649601980498109826984098"
        b"a998009949995299909a00a005a00aa014a022a02aa041a044a050a0a2a0aaa0"
        b"40a165a102a20aa222a228a22aa282a288a28aa2a8a201a404a410a440a489a4"
        b"a4a400a519a551a60aa828a8a2a854a986a908aa0aaa20aa22aa28aa88aaaaaa"
    )

    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        blocks = blocks.astype(np.float32, copy=False)
        out = np.zeros((n_blocks, cls.type_size), dtype=np.uint8)
        kmap, neighbours, grid_l = _get_iq2_s_lookup()

        for i, x in enumerate(blocks):
            qs = out[i, 2 : 2 + QK_K // 8]
            signs = out[i, 2 + QK_K // 8 : 2 + QK_K // 4]
            qh = out[i, 2 + QK_K // 4 : 2 + QK_K // 4 + QK_K // 32]
            packed_scales = out[i, 2 + QK_K // 4 + QK_K // 32 :]
            scales = np.zeros(QK_K // 16, dtype=np.float32)
            max_scale = np.float32(0.0)
            sigma2 = np.float32(np.float32(2.0) * np.sum(x * x, dtype=np.float32) / np.float32(QK_K))

            for ib in range(QK_K // 16):
                xb = x[16 * ib : 16 * ib + 16]
                weight = np.empty(16, dtype=np.float32)
                for j in range(16):
                    weight[j] = np.float32(np.float32(0.25) * sigma2 + np.float32(xb[j] * xb[j]))
                waux = np.sqrt(weight, dtype=np.float32)
                xval = np.empty(16, dtype=np.float32)
                block_signs = np.zeros(2, dtype=np.uint8)

                for k in range(2):
                    sign = 0
                    for j in range(8):
                        index = 8 * k + j
                        if xb[index] >= 0:
                            xval[index] = xb[index]
                        else:
                            xval[index] = -xb[index]
                            sign |= 1 << j
                    block_signs[k] = sign

                L = np.zeros(16, dtype=np.int8)
                max_v = np.float32(np.max(xval))
                if max_v < GROUP_MAX_EPS_IQ2_S:
                    continue

                best = np.float32(0.0)
                scale = np.float32(max_v / np.float32(5.0))
                is_on_grid = np.ones(2, dtype=bool)
                for is_ in range(-9, 10):
                    id_ = (np.float32(5.0) + np.float32(is_) * np.float32(0.1)) / max_v
                    this_scale = np.float32(1.0) / id_
                    Laux = np.zeros(16, dtype=np.int8)
                    is_on_grid_aux = np.ones(2, dtype=bool)
                    for k in range(2):
                        u = 0
                        for j in range(8):
                            index = 8 * k + j
                            l = int(_nearest_int(np.float32(0.5) * np.float32(id_ * xval[index] - np.float32(1.0))).item())
                            l = max(0, min(2, l))
                            Laux[index] = l
                            u |= l << (2 * j)
                        if kmap[u] < 0:
                            is_on_grid_aux[k] = False
                            _, Laux[8 * k : 8 * k + 8] = _best_lattice_neighbour(
                                neighbours[u],
                                grid_l,
                                xval[8 * k : 8 * k + 8],
                                waux[8 * k : 8 * k + 8],
                                this_scale,
                            )

                    sumqx = np.float32(0.0)
                    sumq2 = np.float32(0.0)
                    for j in range(16):
                        q = np.float32(2 * int(Laux[j]) + 1)
                        w = np.float32(weight[j])
                        sumqx = np.float32(sumqx + np.float32(w * xval[j] * q))
                        sumq2 = np.float32(sumq2 + np.float32(w * q * q))
                    if sumq2 > 0 and sumqx * sumqx > best * sumq2:
                        scale = np.float32(sumqx / sumq2)
                        best = np.float32(scale * sumqx)
                        L[:] = Laux
                        is_on_grid[:] = is_on_grid_aux

                if np.any(~is_on_grid) and scale > 0:
                    id_ = np.float32(1.0) / scale
                    for k in range(2):
                        if is_on_grid[k]:
                            continue
                        u = 0
                        for j in range(8):
                            index = 8 * k + j
                            l = int(_nearest_int(np.float32(0.5) * np.float32(id_ * xval[index] - np.float32(1.0))).item())
                            l = max(0, min(2, l))
                            L[index] = l
                            u |= l << (2 * j)
                        if kmap[u] < 0:
                            _, L[8 * k : 8 * k + 8] = _best_lattice_neighbour(
                                neighbours[u],
                                grid_l,
                                xval[8 * k : 8 * k + 8],
                                waux[8 * k : 8 * k + 8],
                                scale,
                                np.float32(1.0e-6),
                            )

                    sumqx = np.float32(0.0)
                    sumq2 = np.float32(0.0)
                    for j in range(16):
                        q = np.float32(2 * int(L[j]) + 1)
                        w = np.float32(weight[j])
                        sumqx = np.float32(sumqx + np.float32(w * xval[j] * q))
                        sumq2 = np.float32(sumq2 + np.float32(w * q * q))
                    if sumq2 > 0:
                        scale = np.float32(sumqx / sumq2)

                if scale < 0:
                    scale = np.float32(-scale)
                    block_signs = (~block_signs).astype(np.uint8)

                for k in range(2):
                    u = 0
                    for j in range(8):
                        u |= int(L[8 * k + j]) << (2 * j)
                    grid_index = int(kmap[u])
                    i8 = 2 * ib + k
                    qs[i8] = grid_index & 0xFF
                    qh[i8 // 4] |= ((grid_index >> 8) & 0x03) << (2 * (i8 % 4))
                    signs[i8] = block_signs[k]

                scales[ib] = scale
                if scale > max_scale:
                    max_scale = scale

            if max_scale == 0:
                continue

            d = np.float32(max_scale / np.float32(31.0))
            out[i, :2] = np.array([np.float32(d * np.float32(0.9875))], dtype=np.float16).view(np.uint8)
            id_ = np.float32(1.0) / d
            for ib in range(QK_K // 16):
                l = int(_nearest_int(np.float32(0.5) * np.float32(id_ * scales[ib] - np.float32(1.0))).item())
                l = max(0, min(15, l))
                if ib % 2 == 0:
                    packed_scales[ib // 2] = l
                else:
                    packed_scales[ib // 2] |= l << 4

        return out

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        qs, rest = np.hsplit(rest, [QK_K // 8])
        signs, rest = np.hsplit(rest, [QK_K // 8])
        qh, scales = np.hsplit(rest, [QK_K // 32])

        d = d.view(np.float16).astype(np.float32)

        scales = scales.reshape((n_blocks, -1, 1)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2))
        scales = (scales & 0x0F).reshape((n_blocks, -1))
        db = d * (np.float32(0.5) + scales) * np.float32(0.25)
        db = db.reshape((n_blocks, -1, 1, 1))

        # unpack the sign bits
        signs = signs.reshape((n_blocks, -1, 1)) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 8))
        signs = signs & np.uint8(0x01)
        signs = np.where(signs == 0, np.float32(1), np.float32(-1))
        signs = signs.reshape((n_blocks, -1, 2, 8))

        qh = qh.reshape((n_blocks, -1, 1)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4))
        qs = qs.astype(np.uint16) | ((qh & 0x03).astype(np.uint16) << 8).reshape((n_blocks, -1))

        assert cls.grid is not None
        grid = np.take_along_axis(cls.grid, qs.reshape((n_blocks, -1, 1, 1)), axis=-2)
        grid = grid.reshape((n_blocks, -1, 2, 8))

        return (db * grid * signs).reshape((n_blocks, -1))


class IQ3_XXS(__Quant, qtype=GGMLQuantizationType.IQ3_XXS):
    grid_shape = (256, 4)
    grid_map = (0x04, 0x0c, 0x14, 0x1c, 0x24, 0x2c, 0x34, 0x3e)
    grid_hex = (
        b"0000020004001100130017002000220031004200730075000101030110011201"
        b"2101250130013201410154017001000202020402110220022202310233023702"
        b"5102570275020103070310031203250370031304370444045704730475040105"
        b"0705320552053506640610071407160743076107011003101010121021102310"
        b"3010321034104710501000110211111120112211011203121012121221123012"
        b"7212001302132013311346136613011405145014201524154615711505162217"
        b"4017002002201120132020202220262031204220012103210521102112212121"
        b"3021632167217021002202221122172220222222372240225522012310231423"
        b"7023742335245324032527254125742501270327162745270130103012302130"
        b"2330503065307230003102312031313144314631013203321032253252327232"
        b"1133333330344734723400350635223555351436363663363337603704401740"
        b"3540374053405740744120423742404260426642074345430444514464442545"
        b"4345704505471047124730471250415070500051065126515551145232527252"
        b"0253535310542354275472540255315550562457425724604460466064602161"
        b"6161176264623063366344640565526533660367216703700570077010703270"
        b"5270267140711272457252720073157333736073217441740075027524753076"
    )

    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        blocks = blocks.astype(np.float32, copy=False)
        out = np.zeros((n_blocks, cls.type_size), dtype=np.uint8)
        kmap, neighbours, grid_l = _get_iq3_xxs_lookup()

        for i, x in enumerate(blocks):
            q3 = out[i, 2 : 2 + QK_K // 4]
            scales_and_signs = out[i, 2 + QK_K // 4 :].view(np.uint32)

            scales = np.zeros(QK_K // 32, dtype=np.float32)
            max_scale = np.float32(0.0)
            for ib in range(QK_K // 32):
                xb = x[32 * ib : 32 * ib + 32]
                weight = (xb * xb).astype(np.float32)
                waux = np.sqrt(weight, dtype=np.float32)
                xval = np.empty(32, dtype=np.float32)
                block_signs = np.zeros(4, dtype=np.uint8)

                for k in range(4):
                    nflip = 0
                    sign = 0
                    for j in range(8):
                        index = 8 * k + j
                        if xb[index] >= 0:
                            xval[index] = xb[index]
                        else:
                            xval[index] = -xb[index]
                            nflip += 1
                            sign |= 1 << j
                    if nflip % 2:
                        imin = 8 * k
                        min_v = np.float32(weight[imin] * xb[imin] * xb[imin])
                        for j in range(1, 8):
                            index = 8 * k + j
                            ax = np.float32(weight[index] * xb[index] * xb[index])
                            if ax < min_v:
                                min_v = ax
                                imin = index
                        xval[imin] = -xval[imin]
                        sign ^= 1 << (imin - 8 * k)
                    block_signs[k] = sign & 0x7F

                max_v = np.float32(np.max(xval))
                L = np.zeros(32, dtype=np.int8)
                if max_v < GROUP_MAX_EPS_IQ3_XXS:
                    scales[ib] = np.float32(0.0)
                    continue

                best = np.float32(0.0)
                scale = np.float32(max_v / np.float32(15.0))
                is_on_grid = np.ones(8, dtype=bool)
                for is_ in range(-15, 16):
                    id_ = np.float32(np.float32(15.0) + np.float32(is_) * np.float32(0.2)) / max_v
                    this_scale = np.float32(1.0) / id_
                    Laux = np.zeros(32, dtype=np.int8)
                    is_on_grid_aux = np.ones(8, dtype=bool)
                    for k in range(8):
                        u = 0
                        for j in range(4):
                            index = 4 * k + j
                            l = int(_nearest_int(np.float32(0.5) * np.float32(id_ * xval[index] - np.float32(1.0))).item())
                            l = max(0, min(7, l))
                            Laux[index] = l
                            u |= l << (3 * j)
                        grid_index = int(kmap[u])
                        if grid_index < 0:
                            is_on_grid_aux[k] = False
                            grid_index, Laux[4 * k : 4 * k + 4] = _best_lattice_neighbour(
                                neighbours[u],
                                grid_l,
                                xval[4 * k : 4 * k + 4],
                                waux[4 * k : 4 * k + 4],
                                this_scale,
                            )

                    sumqx = np.float32(0.0)
                    sumq2 = np.float32(0.0)
                    for j in range(32):
                        q = np.float32(2 * int(Laux[j]) + 1)
                        w = np.float32(weight[j])
                        sumqx = np.float32(sumqx + np.float32(w * xval[j] * q))
                        sumq2 = np.float32(sumq2 + np.float32(w * q * q))
                    if sumq2 > 0 and sumqx * sumqx > best * sumq2:
                        scale = np.float32(sumqx / sumq2)
                        best = np.float32(scale * sumqx)
                        L[:] = Laux
                        is_on_grid[:] = is_on_grid_aux

                if np.any(~is_on_grid) and scale > 0:
                    id_ = np.float32(1.0) / scale
                    for k in range(8):
                        if is_on_grid[k]:
                            continue
                        u = 0
                        for j in range(4):
                            index = 4 * k + j
                            l = int(_nearest_int(np.float32(0.5) * np.float32(id_ * xval[index] - np.float32(1.0))).item())
                            l = max(0, min(7, l))
                            u |= l << (3 * j)
                        grid_index = int(kmap[u])
                        if grid_index < 0:
                            grid_index, L[4 * k : 4 * k + 4] = _best_lattice_neighbour(
                                neighbours[u],
                                grid_l,
                                xval[4 * k : 4 * k + 4],
                                waux[4 * k : 4 * k + 4],
                                scale,
                            )
                        else:
                            L[4 * k : 4 * k + 4] = grid_l[grid_index]

                    sumqx = np.float32(0.0)
                    sumq2 = np.float32(0.0)
                    for j in range(32):
                        q = np.float32(2 * int(L[j]) + 1)
                        w = np.float32(weight[j])
                        sumqx = np.float32(sumqx + np.float32(w * xval[j] * q))
                        sumq2 = np.float32(sumq2 + np.float32(w * q * q))
                    if sumq2 > 0:
                        scale = np.float32(sumqx / sumq2)

                if scale < 0:
                    scale = -scale
                    for k in range(4):
                        block_signs[k] = (~block_signs[k]) & 0x7F

                for k in range(8):
                    u = 0
                    for j in range(4):
                        u |= int(L[4 * k + j]) << (3 * j)
                    q3[8 * ib + k] = int(kmap[u])

                scales_and_signs[ib] = (
                    int(block_signs[0])
                    | (int(block_signs[1]) << 7)
                    | (int(block_signs[2]) << 14)
                    | (int(block_signs[3]) << 21)
                )
                scales[ib] = scale
                if scale > max_scale:
                    max_scale = scale

            if max_scale == 0:
                continue

            d = np.float32(max_scale / np.float32(31.0))
            out[i, :2] = np.array([np.float32(d * np.float32(1.0125))], dtype=np.float16).view(np.uint8)
            id_ = np.float32(1.0) / d
            for ib in range(QK_K // 32):
                l = int(_nearest_int(np.float32(0.5) * np.float32(id_ * scales[ib] - np.float32(1.0))).item())
                l = max(0, min(15, l))
                scales_and_signs[ib] |= np.uint32(l << 28)

        return out

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        qs, scales = np.hsplit(rest, [QK_K // 4])

        d = d.view(np.float16).astype(np.float32)
        scales = scales.view(np.uint32)

        db = d * (np.float32(0.5) + (scales >> 28).astype(np.float32)) * np.float32(0.5)
        db = db.reshape((n_blocks, -1, 1, 1))

        # get the sign indices and unpack the bits
        signs = scales.reshape((n_blocks, -1, 1)) >> np.array([0, 7, 14, 21], dtype=np.uint32).reshape((1, 1, 4))
        ksigns = np.frombuffer(IQ2_XXS.ksigns, dtype=np.uint8).reshape((1, 1, 1, 128))
        signs = (signs & np.uint32(0x7F)).reshape((n_blocks, -1, 4, 1))
        signs = np.take_along_axis(ksigns, signs, axis=-1)
        signs = signs.reshape((n_blocks, -1, 4, 1)) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 1, 8))
        signs = signs & np.uint8(0x01)
        signs = np.where(signs == 0, np.float32(1), np.float32(-1))
        signs = signs.reshape((n_blocks, -1, 4, 8))

        assert cls.grid is not None
        grid = np.take_along_axis(cls.grid, qs.reshape((n_blocks, -1, 1, 1)), axis=-2)
        grid = grid.reshape((n_blocks, -1, 4, 8))

        return (db * grid * signs).reshape((n_blocks, -1))


class IQ3_S(__Quant, qtype=GGMLQuantizationType.IQ3_S):
    grid_shape = (512, 4)
    grid_map = (0x01, 0x03, 0x05, 0x07, 0x09, 0x0b, 0x0d, 0x0f)
    grid_hex = (
        b"0000010002000500070010001100120014001600200021002500330040004200"
        b"4500470051005300600062007100740077000001010102010401100111011501"
        b"2001230127013101350144016101650172010002010205020702100213021602"
        b"2102250230023402420245024702510253027002730203031103150320032203"
        b"3103330336034403500352036703710375030004130417042104240432044004"
        b"4304510470040205040520052205260533054105450547056605730506061106"
        b"1306310652067106000702070407200722072607330750075407001001100210"
        b"0410101011101310151017102010221031103410361054105610611072100011"
        b"0111031106111011141121113011331141115011521170117611001212121512"
        b"1712201224123212401243125512601272120113041307131013131321132713"
        b"3013341341136213701303140514121414143114331442144614501454140115"
        b"1015131521153015321551152016241627164416461601170317101712172117"
        b"3517411762177017002001200320052007201020122014201620212023202720"
        b"3020322041204320452050205220672070207320752000210221102113211721"
        b"2221252131213421422151210122042207222122232230223722412253225722"
        b"7122742200230223052311232223242331233323422350236623012407242024"
        b"2324322435244124722475240425112522253725402553257025002602260726"
        b"2126552661260527112726273027432750270230113013301530173022303130"
        b"3330353042304430473051306330713001310331053114312131233140316031"
        b"7231763100321232203232323432503201331033143321332333273330334133"
        b"4333473355337333033411341634223431345234603464340135103512352535"
        b"3235443556357335163641360137033720372237353700400440124020402440"
        b"2740324041405040704002410741114113412241304135414341514155410142"
        b"0342104215422142334240425742624270420443114313432043224331433543"
        b"0044024424443744404471440545074521456245134634466046104715473047"
        b"4347514702501050145022504050445047505250665074500151035105511251"
        b"2151325172510052115223523052365253520253075310532753445351536553"
        b"7353015404542054325446541255265551555355425602570457225711601360"
        b"1560316033606060006120612761646112623462426255626262706200631463"
        b"2163406325644364626400650365346560650566406611671367007004700770"
        b"2070227036704070547062700271117124714371457101720472107216722172"
        b"3072517202733273357353730174057413742074507422754275027631760077"
    )

    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        blocks = blocks.astype(np.float32, copy=False)
        out = np.zeros((n_blocks, cls.type_size), dtype=np.uint8)
        kmap, neighbours, grid_l = _get_iq3_s_lookup()

        for i, x in enumerate(blocks):
            qs = out[i, 2 : 2 + QK_K // 4]
            qh = out[i, 2 + QK_K // 4 : 2 + QK_K // 4 + QK_K // 32]
            signs = out[
                i,
                2 + QK_K // 4 + QK_K // 32 : 2 + QK_K // 4 + QK_K // 32 + QK_K // 8,
            ]
            packed_scales = out[i, 2 + QK_K // 4 + QK_K // 32 + QK_K // 8 :]

            scales = np.zeros(QK_K // 32, dtype=np.float32)
            max_scale = np.float32(0.0)
            for ib in range(QK_K // 32):
                xb = x[32 * ib : 32 * ib + 32]
                weight = (xb * xb).astype(np.float32)
                waux = np.sqrt(weight).astype(np.float32)
                xval = np.empty(32, dtype=np.float32)
                block_signs = np.zeros(4, dtype=np.uint8)

                for k in range(4):
                    sign = 0
                    for j in range(8):
                        index = 8 * k + j
                        if xb[index] >= 0:
                            xval[index] = xb[index]
                        else:
                            xval[index] = -xb[index]
                            sign |= 1 << j
                    block_signs[k] = sign

                max_v = np.float32(np.max(xval))
                L = np.zeros(32, dtype=np.int8)
                if max_v == 0:
                    scales[ib] = np.float32(0.0)
                    continue

                best = np.float32(0.0)
                scale = np.float32(max_v / np.float32(15.0))
                is_on_grid = np.zeros(8, dtype=bool)
                for is_ in range(-9, 10):
                    id_ = np.float32(np.float32(15.0) + np.float32(is_) * np.float32(0.2)) / max_v
                    this_scale = np.float32(1.0) / id_
                    Laux = np.zeros(32, dtype=np.int8)
                    is_on_grid_aux = np.ones(8, dtype=bool)
                    for k in range(8):
                        u = 0
                        for j in range(4):
                            index = 4 * k + j
                            l = int(_nearest_int(np.float32(0.5) * np.float32(id_ * xval[index] - np.float32(1.0))).item())
                            l = max(0, min(7, l))
                            Laux[index] = l
                            u |= l << (3 * j)
                        grid_index = int(kmap[u])
                        if grid_index < 0:
                            is_on_grid_aux[k] = False
                            grid_index, Laux[4 * k : 4 * k + 4] = _best_lattice_neighbour(
                                neighbours[u],
                                grid_l,
                                xval[4 * k : 4 * k + 4],
                                waux[4 * k : 4 * k + 4],
                                this_scale,
                            )

                    sumqx = np.float32(0.0)
                    sumq2 = np.float32(0.0)
                    for j in range(32):
                        q = np.float32(2 * int(Laux[j]) + 1)
                        w = np.float32(weight[j])
                        sumqx = np.float32(sumqx + np.float32(w * xval[j] * q))
                        sumq2 = np.float32(sumq2 + np.float32(w * q * q))
                    if sumq2 > 0 and sumqx * sumqx > best * sumq2:
                        scale = np.float32(sumqx / sumq2)
                        best = np.float32(scale * sumqx)
                        L[:] = Laux
                        is_on_grid[:] = is_on_grid_aux

                if np.any(~is_on_grid) and scale > 0:
                    id_ = np.float32(1.0) / scale
                    for k in range(8):
                        u = 0
                        for j in range(4):
                            index = 4 * k + j
                            l = int(_nearest_int(np.float32(0.5) * np.float32(id_ * xval[index] - np.float32(1.0))).item())
                            l = max(0, min(7, l))
                            u |= l << (3 * j)
                        grid_index = int(kmap[u])
                        if grid_index < 0:
                            grid_index, L[4 * k : 4 * k + 4] = _best_lattice_neighbour(
                                neighbours[u],
                                grid_l,
                                xval[4 * k : 4 * k + 4],
                                waux[4 * k : 4 * k + 4],
                                scale,
                            )
                        else:
                            L[4 * k : 4 * k + 4] = grid_l[grid_index]

                    sumqx = np.float32(0.0)
                    sumq2 = np.float32(0.0)
                    for j in range(32):
                        q = np.float32(2 * int(L[j]) + 1)
                        w = np.float32(weight[j])
                        sumqx = np.float32(sumqx + np.float32(w * xval[j] * q))
                        sumq2 = np.float32(sumq2 + np.float32(w * q * q))
                    if sumq2 > 0:
                        scale = np.float32(sumqx / sumq2)

                if scale < 0:
                    scale = -scale
                    block_signs = (~block_signs).astype(np.uint8)

                for k in range(8):
                    u = 0
                    for j in range(4):
                        u |= int(L[4 * k + j]) << (3 * j)
                    grid_index = int(kmap[u])
                    qs[8 * ib + k] = grid_index & 0xFF
                    qh[(8 * ib + k) // 8] |= ((grid_index >> 8) & 1) << ((8 * ib + k) % 8)

                signs[4 * ib : 4 * ib + 4] = block_signs
                scales[ib] = scale
                if scale > max_scale:
                    max_scale = scale

            if max_scale == 0:
                continue

            d = np.float32(max_scale / np.float32(31.0))
            out[i, :2] = np.array([np.float32(d * np.float32(1.033))], dtype=np.float16).view(np.uint8)
            id_ = np.float32(1.0) / d
            for ib in range(0, QK_K // 32, 2):
                l1 = int(_nearest_int(np.float32(0.5) * np.float32(id_ * scales[ib] - np.float32(1.0))).item())
                l1 = max(0, min(15, l1))
                l2 = int(_nearest_int(np.float32(0.5) * np.float32(id_ * scales[ib + 1] - np.float32(1.0))).item())
                l2 = max(0, min(15, l2))
                packed_scales[ib // 2] = l1 | (l2 << 4)

        return out

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        qs, rest = np.hsplit(rest, [QK_K // 4])
        qh, rest = np.hsplit(rest, [QK_K // 32])
        signs, scales = np.hsplit(rest, [QK_K // 8])

        d = d.view(np.float16).astype(np.float32)

        scales = scales.reshape((n_blocks, -1, 1)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2))
        scales = (scales & 0x0F).reshape((n_blocks, -1))
        db = d * (1 + 2 * scales)
        db = db.reshape((n_blocks, -1, 1, 1))

        # unpack the sign bits
        signs = signs.reshape((n_blocks, -1, 1)) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 8))
        signs = signs & np.uint8(0x01)
        signs = np.where(signs == 0, np.float32(1), np.float32(-1))
        signs = signs.reshape((n_blocks, -1, 4, 8))

        qh = qh.reshape((n_blocks, -1, 1)) >> np.array([i for i in range(8)], dtype=np.uint8)
        qh = (qh & 0x01).astype(np.uint16).reshape((n_blocks, -1))
        qs = qs.astype(np.uint16) | (qh << 8)

        assert cls.grid is not None
        grid = np.take_along_axis(cls.grid, qs.reshape((n_blocks, -1, 1, 1)), axis=-2)
        grid = grid.reshape((n_blocks, -1, 4, 8))

        return (db * grid * signs).reshape((n_blocks, -1))


class IQ1_S(__Quant, qtype=GGMLQuantizationType.IQ1_S):
    # iq1s_grid, with each byte packed into 2 bits
    # -1, 0, 1 <=> 0, 1, 2
    grid_shape = (2048, 8)
    grid_map = (-1, 0, 1)
    grid_hex = (
        b"00000200050008000a00110015002000220028002a0045005100540056006500"
        b"8000820088008a009500a000a200a800aa000401050111011401160119011a01"
        b"2501410146014901520155015a0161016401660168018501910194019601a501"
        b"0002020208020a0215022002220228022a024502510259026402690280028202"
        b"88028a02910295029902a002a202a802aa021104140416042504410449045504"
        b"5a046404650491049904a5040105040505050605150518051a05290540054505"
        b"4a0550055105540555055605590560056205650568056a058105910595059805"
        b"9a05a105a405a505a605a9051406190641064406500652065506580660066106"
        b"6606690685069106940699060008020808080a0815082008220828082a084508"
        b"5108560865088008820888088a089508a008a208a808aa080509110914091909"
        b"2409250941095009510955096109640969099109940996099909a509000a020a"
        b"080a0a0a150a200a220a280a2a0a450a510a590a610a650a800a820a850a880a"
        b"8a0a950aa00aa20aa80aaa0a1010111014101910241025104110441050105510"
        b"58106110641065106910911094109610a110a510011104110611091110111211"
        b"1511181121112411291145114a11501151115211541155115611591160116511"
        b"841192119511a111a41111121412161225124012461249125212551258125a12"
        b"641266128512911294129612a512011406140914141415141814191421142614"
        b"41144514461448144a1451145414551456145914621465146814841489149014"
        b"94149514981499149a14a114a414a514a914021505150a151115141515151615"
        b"191520152215251528152a154115441545154615511552155415551556155915"
        b"5a1561156415651566156915801582158415851588158a159015911594159515"
        b"961599159a15a015a215a51501160416051606161516161618161a1621162616"
        b"401642164416451648164a165116551656165816591661166416651668166916"
        b"6a1686168a1692169516a416a916111816182518411844184618491850185518"
        b"58185a1860186118641866186918851891189418a5181019121915191a192119"
        b"25194219441945194819511954195519561959195a19601965196a1989199119"
        b"921995199819a119a619a919091a161a241a261a441a461a491a501a521a551a"
        b"581a611a661a691a851a911a961a9a1a0020022008200a201520202022202520"
        b"28202a20452051205920612065208020822088208a209520a020a220a520a820"
        b"aa2005211121142119212521422144214921552158215a216121642165216621"
        b"8521902196219921a521012208220a22112215222022222228222a2245225122"
        b"562259226522812288228a2291229522a022a222a822aa220524142416241924"
        b"252444244524462449245224552458245a2466248524912494249924a124a524"
        b"0925152521252925402545254825512554255525592562256525682589259025"
        b"9425952598259a25a125a425a625a92505261026122619262526412649265526"
        b"6026612669268426862690269a260028022808280a2815282028222828282a28"
        b"45285128542865288028822888288a28a028a228a828aa280929112914291929"
        b"2529462949295229552961296429662969298529902996299929a429a529002a"
        b"022a082a0a2a202a222a282a2a2a452a512a562a592a652a802a822a882a8a2a"
        b"952aa02aa22aa82aaa2a054011401640254049405240554058405a4061406440"
        b"664094409940a140a6400041014104410641094112411541164118411a412141"
        b"26412941454148414a41514154415541564159415a41654168416a4181418441"
        b"8641904192419541a041a141a241054211421442164225424142524255425a42"
        b"6442694289429442a5420144154419442944454448444a445144544455445644"
        b"61446244654468446a44814486448944904492449544a044a144a94401450245"
        b"05450a4511451445154516451945204525452a45414544454545464549455045"
        b"5145544555455645584559456145644565456645694582458445854588459145"
        b"94459545964599459a45a545a845aa450146054609461446154618461a462146"
        b"2446294640464246454648465046514652465546564659466246654668468146"
        b"85468a4694469546a146a446a6460548114815481a4825484248494850485548"
        b"5848614864486648694885489148944896489948a5480149054906490a491049"
        b"144915491849214924492649404945494a495149524954495549564959496049"
        b"6249654966496a49864989499249954996499849a149a449a649a949164a444a"
        b"464a494a554a584a5a4a644a694a944aa54a0150045005500650095012501550"
        b"1a50215024502950405045504850515054505550565059506550685086508950"
        b"95509850a050a150a650a9500551085109510a51115114511551165118511951"
        b"20512551265128512a5141514451455146514951505151515251545155515651"
        b"585159515a51615164516551665169518251855191519451955196519951a051"
        b"a551aa5101520652125215521a5221522452425245524a525152545255525652"
        b"595262526552855290529252955299529a52a452045405541154145415541654"
        b"185419542154255428542a54415444544554465449544a545054515454545554"
        b"5654585459545a54615462546454655466546954805488548a54915494549554"
        b"96549954a154a454a554aa540155025504550555065509551055115512551455"
        b"1555165519551a55215524552555265529554055415542554455455546554855"
        b"4955505551555255545555555655585559555a55605561556455655566556855"
        b"69556a5581558455855589558a559055915594559555965598559955a155a455"
        b"a555a655a9550056015602560456065608560956115614561556185619562056"
        b"2156225624562556265628562956415645564656485649564a56505651565256"
        b"545655565656585659565a566156645665566956825685568656885689568a56"
        b"915695569a56a256a556a656a856a95604580558065809581058155818582158"
        b"2a58455848584a58515854585558565858585958605862586458655882588958"
        b"9058925895589858a158a9580159025905590a59115914591559165919592559"
        b"41594459455946594959505951595259545955595659585959595a5961596459"
        b"655966596959815985598959915994599559965998599959a559045a085a155a"
        b"1a5a205a255a265a295a455a485a495a515a555a565a585a595a625a655a685a"
        b"6a5a815a8a5a925a955a965a985a9a5aa15a0560146016601960256044605060"
        b"5560566058605a60616064606660696081609660a56001610461066109611261"
        b"15612161226126612961456149615161556156615961656166616a6184618a61"
        b"92619561a161a661a96111621662196240624162466255625662586260628562"
        b"91629662a56211641264156416641a6421642664296440644264456448644a64"
        b"516454645564566459645a646064626465648464856489649064926494649564"
        b"966498649a64a164a464a964056508650a651165156516651965446545654665"
        b"496550655165546555655665596561656465656566656965866589658a659165"
        b"9565966599659a65a265a565a665a86502660966156620662666286629664066"
        b"456648664a66516654665566566658665a666066656668668066826685668a66"
        b"9466966698669966a066a466a666aa661668196825684168526855685a686168"
        b"6968856891689868a66801690469106915692169246926692969406941694569"
        b"4669486951695469556956695969606965696a69826984698a699569a169a469"
        b"a569a969116a166a186a416a446a496a506a556a586a5a6a646a656a696a866a"
        b"946a986a9a6aa66a0080028008800a802080228028802a804580508051805480"
        b"5680598065808080828088808a809580a080a280a880aa800581118114811681"
        b"1981258141814481498150815281558156815881598164816681698185818981"
        b"948196819981a5810082028208820a8215822082228228822a82518254825982"
        b"65828082828288828a829582a082a282a882aa82148419844184448451845584"
        b"5a846184648469849484998401850985128515851a8526852985408541854585"
        b"4885518554855585568559855a856585668568856a8581858485868589859085"
        b"928595859885a68511861686198625864186448649864a865086558659865a86"
        b"618666866a86858691869a86a4860088028808880a8815882088228828882a88"
        b"41884588518854885988658869888088828888888a889588a088a288a888aa88"
        b"05890689118914891689258941894489468949895089528955895a8961896489"
        b"858996899989a589008a028a088a0a8a158a208a228a288a2a8a458a518a548a"
        b"568a808a828a888a8a8a958aa08aa28aa88aaa8a059011901690189019902590"
        b"419046904990559058905a9069906a9085909190949096909990a59001910491"
        b"069109911091159118911a912191249126912991409145915091519154915591"
        b"569159916291659184918691929195919891a191a491a691a991059211921492"
        b"19922592449246924992509252925592589266926992859294929692a9920194"
        b"04940694109415941894269440944a9451945494559456945894599460946194"
        b"62946594849486949294949495949894a194a9940095059508950a9510951195"
        b"14951595169519952195259529952a9541954495459546954995509551955295"
        b"549555955695589559955a956195649565956695699581958595889591959295"
        b"94959595969599959a95a095a295a595a895aa95019604961096159619962096"
        b"2696299645964896499651965296559656965996659668968296849689968a96"
        b"929694969596a496a696a9960598169819982598419846985098529855985698"
        b"5a98649865988598919896989998a59804990699099910991299159918991a99"
        b"209921992499269940994299459948994a995199549955995699599962996599"
        b"66996a99819984999099929995999a99a199a699059a159a259a449a469a499a"
        b"509a559a589a619a859a919a949a959a969a00a002a008a00aa015a020a022a0"
        b"28a02aa045a051a054a056a059a080a082a088a08aa095a0a0a0a2a0a8a0aaa0"
        b"05a109a111a114a116a119a11aa146a149a151a155a158a15aa161a164a185a1"
        b"90a192a196a199a102a208a20aa210a219a222a228a22aa245a251a256a259a2"
        b"65a280a282a288a28aa295a2a0a2a2a2a8a2aaa219a425a441a444a450a454a4"
        b"55a458a45aa461a465a466a468a469a485a406a509a510a512a515a518a526a5"
        b"29a542a545a551a554a555a556a559a565a56aa581a584a585a586a589a592a5"
        b"95a598a505a611a616a61aa621a625a644a646a64aa652a655a656a658a660a6"
        b"62a686a690a695a696a699a6a1a6a4a6a6a600a802a808a80aa820a822a828a8"
        b"2aa851a854a856a859a880a882a888a88aa895a8a0a8a2a8a8a8aaa805a914a9"
        b"19a921a925a941a950a955a95aa961a966a969a990a996a900aa02aa08aa0aaa"
        b"20aa22aa28aa2aaa51aa54aa56aa80aa82aa88aa8aaa95aaa0aaa2aaa8aaaaaa"
    )

    delta = np.float32(0.125)

    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        blocks = blocks.astype(np.float32, copy=False)
        out = np.zeros((n_blocks, cls.type_size), dtype=np.uint8)
        kmap, neighbours, grid_l = _get_iq1_s_lookup()
        quant_weights = np.sum(blocks * blocks, axis=0, dtype=np.float32)
        x_p = np.array([-np.float32(1.0) + cls.delta, cls.delta, np.float32(1.0) + cls.delta], dtype=np.float32)
        x_m = np.array([-np.float32(1.0) - cls.delta, -cls.delta, np.float32(1.0) - cls.delta], dtype=np.float32)

        for i, x in enumerate(blocks):
            qs = out[i, 2 : 2 + QK_K // 8]
            qh = out[i, 2 + QK_K // 8 :].view(np.uint16)
            scales = np.zeros(QK_K // 32, dtype=np.float32)
            shifts = np.ones(QK_K // 32, dtype=np.int8)
            max_scale = np.float32(0.0)
            sumx2 = np.float32(0.0)
            for value in x:
                sumx2 = np.float32(sumx2 + np.float32(value * value))
            sigma2 = np.float32(np.float32(2.0) * sumx2 / np.float32(QK_K))

            for ib in range(QK_K // 32):
                xb = x[32 * ib : 32 * ib + 32]
                qw = quant_weights[32 * ib : 32 * ib + 32]
                weight = np.empty(32, dtype=np.float32)
                for j in range(32):
                    weight[j] = np.float32(qw[j] * np.float32(np.sqrt(np.float32(sigma2 + np.float32(xb[j] * xb[j])))))

                max_v = np.float32(np.max(np.abs(xb)))
                L = np.ones(32, dtype=np.int8)
                if max_v < GROUP_MAX_EPS_IQ1_S:
                    scales[ib] = np.float32(0.0)
                    shifts[ib] = 1
                    continue

                order = np.argsort(xb, kind="quicksort")
                sumx = np.zeros(33, dtype=np.float32)
                sumw = np.zeros(33, dtype=np.float32)
                for j, index in enumerate(order):
                    sumx[j + 1] = np.float32(sumx[j] + np.float32(weight[index] * xb[index]))
                    sumw[j + 1] = np.float32(sumw[j] + weight[index])

                best_score = np.float32(-np.finfo(np.float32).max)
                scale = max_v
                besti1 = -1
                besti2 = -1
                best_shift = 0
                for i1 in range(33):
                    for i2 in range(i1, 33):
                        for shift, values in ((1, x_p), (-1, x_m)):
                            sumqx = np.float32(
                                np.float32(np.float32(sumx[i1] - sumx[0]) * values[0])
                                + np.float32(np.float32(sumx[i2] - sumx[i1]) * values[1])
                            )
                            sumqx = np.float32(
                                sumqx + np.float32(np.float32(sumx[32] - sumx[i2]) * values[2])
                            )
                            w0 = np.float32(sumw[i1] - sumw[0])
                            w1 = np.float32(sumw[i2] - sumw[i1])
                            w2 = np.float32(sumw[32] - sumw[i2])
                            q20 = np.float32(np.float32(w0 * values[0]) * values[0])
                            q21 = np.float32(np.float32(w1 * values[1]) * values[1])
                            q22 = np.float32(np.float32(w2 * values[2]) * values[2])
                            sumq2 = np.float32(
                                np.float32(q20 + q21) + q22
                            )
                            lhs = np.float32(sumqx * sumqx)
                            rhs = np.float32(-np.inf) if best_score < 0 else np.float32(best_score * sumq2)
                            if sumq2 > 0 and lhs > rhs:
                                scale = np.float32(sumqx / sumq2)
                                best_score = np.float32(scale * sumqx)
                                besti1 = i1
                                besti2 = i2
                                best_shift = shift

                if besti1 < 0 or besti2 < 0 or best_shift == 0:
                    scales[ib] = np.float32(0.0)
                    shifts[ib] = 1
                    continue

                for j in range(besti1):
                    L[order[j]] = 0
                for j in range(besti1, besti2):
                    L[order[j]] = 1
                for j in range(besti2, 32):
                    L[order[j]] = 2

                if scale < 0:
                    for j in range(32):
                        L[j] = 2 - L[j]
                    scale = np.float32(-scale)
                    best_shift = -best_shift

                all_on_grid = True
                values = x_p if best_shift == 1 else x_m
                grid_indices = np.zeros(4, dtype=np.uint16)
                for k in range(4):
                    u = 0
                    for j in range(8):
                        u |= int(L[8 * k + j]) << (2 * j)
                    grid_index = int(kmap[u])
                    if grid_index < 0:
                        all_on_grid = False
                        grid_index, L[8 * k : 8 * k + 8] = _best_iq1_neighbour(
                            neighbours[u],
                            grid_l,
                            xb[8 * k : 8 * k + 8],
                            weight[8 * k : 8 * k + 8],
                            scale,
                            values,
                        )
                    grid_indices[k] = grid_index

                if not all_on_grid:
                    sumqx = np.float32(0.0)
                    sumq2 = np.float32(0.0)
                    for k in range(4):
                        levels = grid_l[int(grid_indices[k])]
                        for j in range(8):
                            index = 8 * k + j
                            w = np.float32(weight[index])
                            q = np.float32(values[int(levels[j])])
                            sumqx = np.float32(sumqx + np.float32(w * q * xb[index]))
                            sumq2 = np.float32(sumq2 + np.float32(w * q * q))
                    if sumqx > 0 and sumq2 > 0:
                        scale = np.float32(sumqx / sumq2)

                h = 0
                for k in range(4):
                    grid_index = int(grid_indices[k])
                    qs[4 * ib + k] = grid_index & 0xFF
                    h |= (grid_index >> 8) << (3 * k)
                qh[ib] = h
                scales[ib] = scale
                shifts[ib] = best_shift
                if scale > max_scale:
                    max_scale = scale

            if max_scale == 0:
                continue

            d = np.float32(max_scale / np.float32(15.0))
            out[i, :2] = np.array([np.float32(d * np.float32(1.125))], dtype=np.float16).view(np.uint8)
            id_ = np.float32(1.0) / d
            for ib in range(QK_K // 32):
                l = int(_nearest_int(np.float32(0.5) * np.float32(id_ * scales[ib] - np.float32(1.0))).item())
                l = max(0, min(7, l))
                if shifts[ib] == -1:
                    l |= 8
                qh[ib] |= np.uint16(l << 12)

        return out

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        qs, qh = np.hsplit(rest, [QK_K // 8])

        d = d.view(np.float16).astype(np.float32)
        qh = qh.view(np.uint16)

        dl = d * (2 * ((qh >> 12) & 7) + 1)
        dl = dl.reshape((n_blocks, -1, 1, 1))
        delta = np.where((qh & np.uint16(0x8000)) == 0, cls.delta, -cls.delta)
        delta = delta.reshape((n_blocks, -1, 1, 1))

        qh = qh.reshape((n_blocks, -1, 1)) >> np.array([0, 3, 6, 9], dtype=np.uint16).reshape((1, 1, 4))
        qs = qs.astype(np.uint16) | ((qh & 7) << 8).reshape((n_blocks, -1))

        assert cls.grid is not None
        grid = np.take_along_axis(cls.grid, qs.reshape((n_blocks, -1, 1, 1)), axis=-2)
        grid = grid.reshape((n_blocks, -1, 4, 8))

        return (dl * (grid + delta)).reshape((n_blocks, -1))


class IQ1_M(__Quant, qtype=GGMLQuantizationType.IQ1_M):
    grid_shape = IQ1_S.grid_shape
    grid_map = IQ1_S.grid_map
    grid_hex = IQ1_S.grid_hex

    delta = IQ1_S.delta

    # Okay *this* type is weird. It's the only one which stores the f16 scales in multiple parts.
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        blocks = blocks.astype(np.float32, copy=False)
        out = np.zeros((n_blocks, cls.type_size), dtype=np.uint8)
        kmap, neighbours, grid_l = _get_iq1_s_lookup()
        x_p = np.array([-np.float32(1.0) + cls.delta, cls.delta, np.float32(1.0) + cls.delta], dtype=np.float32)
        x_m = np.array([-np.float32(1.0) - cls.delta, -cls.delta, np.float32(1.0) - cls.delta], dtype=np.float32)
        masks = (0x00, 0x80, 0x08, 0x88)

        for i, x in enumerate(blocks):
            qs = out[i, : QK_K // 8]
            qh = out[i, QK_K // 8 : QK_K // 8 + QK_K // 16]
            packed_scales = out[i, QK_K // 8 + QK_K // 16 :].view(np.uint16)
            scales = np.zeros(QK_K // 16, dtype=np.float32)
            shifts = np.zeros(QK_K // 16, dtype=np.int8)
            max_scale = np.float32(0.0)
            sumx2 = np.float32(0.0)
            for value in x:
                sumx2 = np.float32(sumx2 + np.float32(value * value))
            sigma2 = np.float32(np.float32(2.0) * sumx2 / np.float32(QK_K))

            for ib in range(QK_K // 16):
                xb = x[16 * ib : 16 * ib + 16]
                weight = (xb * xb).astype(np.float32)
                max_v = np.float32(np.max(np.abs(xb)))
                L = np.ones(16, dtype=np.int8)
                if max_v < GROUP_MAX_EPS_IQ1_M:
                    scales[ib] = np.float32(0.0)
                    shifts[ib] = 0
                    continue

                order = np.argsort(xb, kind="quicksort")
                best_score = np.float32(-np.finfo(np.float32).max)
                scale = max_v
                besti1 = -1
                besti2 = -1
                best_k = -1
                for i1 in range(17):
                    for i2 in range(i1, 17):
                        sumqx = np.zeros(4, dtype=np.float32)
                        sumq2 = np.zeros(4, dtype=np.float32)
                        for j in range(16):
                            index = int(order[j])
                            if j < i1:
                                level = 0
                            elif j < i2:
                                level = 1
                            else:
                                level = 2

                            values0 = x_p if index < 8 else x_p
                            values1 = x_p if index < 8 else x_m
                            values2 = x_m if index < 8 else x_p
                            values3 = x_m if index < 8 else x_m
                            for k, values in enumerate((values0, values1, values2, values3)):
                                q = np.float32(values[level])
                                w = np.float32(weight[index])
                                sumqx[k] = np.float32(sumqx[k] + np.float32(np.float32(w * q) * xb[index]))
                                sumq2[k] = np.float32(sumq2[k] + np.float32(np.float32(w * q) * q))

                        for k in range(4):
                            lhs = np.float32(sumqx[k] * sumqx[k])
                            rhs = np.float32(-np.inf) if best_score < 0 else np.float32(best_score * sumq2[k])
                            if sumq2[k] > 0 and lhs > rhs:
                                scale = np.float32(sumqx[k] / sumq2[k])
                                best_score = np.float32(scale * sumqx[k])
                                besti1 = i1
                                besti2 = i2
                                best_k = k

                if besti1 < 0 or besti2 < 0 or best_k < 0:
                    scales[ib] = np.float32(0.0)
                    shifts[ib] = 0
                    continue

                for j in range(besti1):
                    L[order[j]] = 0
                for j in range(besti1, besti2):
                    L[order[j]] = 1
                for j in range(besti2, 16):
                    L[order[j]] = 2

                if scale < 0:
                    for j in range(16):
                        L[j] = 2 - L[j]
                    scale = np.float32(-scale)
                    best_k = (3, 2, 1, 0)[best_k]

                all_on_grid = True
                grid_indices = np.zeros(2, dtype=np.uint16)
                for k in range(2):
                    values = x_p if (best_k < 2 if k == 0 else best_k % 2 == 0) else x_m
                    u = 0
                    for j in range(8):
                        u |= int(L[8 * k + j]) << (2 * j)
                    grid_index = int(kmap[u])
                    if grid_index < 0:
                        all_on_grid = False
                        grid_index, L[8 * k : 8 * k + 8] = _best_iq1_neighbour(
                            neighbours[u],
                            grid_l,
                            xb[8 * k : 8 * k + 8],
                            weight[8 * k : 8 * k + 8],
                            scale,
                            values,
                        )
                    grid_indices[k] = grid_index

                if not all_on_grid:
                    sumqx_f = np.float32(0.0)
                    sumq2_f = np.float32(0.0)
                    for k in range(2):
                        values = x_p if (best_k < 2 if k == 0 else best_k % 2 == 0) else x_m
                        levels = grid_l[int(grid_indices[k])]
                        for j in range(8):
                            index = 8 * k + j
                            w = np.float32(weight[index])
                            q = np.float32(values[int(levels[j])])
                            sumqx_f = np.float32(sumqx_f + np.float32(np.float32(w * q) * xb[index]))
                            sumq2_f = np.float32(sumq2_f + np.float32(np.float32(w * q) * q))
                    if sumqx_f > 0 and sumq2_f > 0:
                        scale = np.float32(sumqx_f / sumq2_f)

                qs[2 * ib] = int(grid_indices[0]) & 0xFF
                qs[2 * ib + 1] = int(grid_indices[1]) & 0xFF
                qh[ib] = (int(grid_indices[0]) >> 8) | ((int(grid_indices[1]) >> 8) << 4)
                scales[ib] = scale
                shifts[ib] = best_k
                if scale > max_scale:
                    max_scale = scale

            if max_scale == 0:
                continue

            d = np.float32(max_scale / np.float32(15.0))
            id_ = np.float32(1.0) / d
            sumqx_f = np.float32(0.0)
            sumq2_f = np.float32(0.0)
            for ib in range(QK_K // 16):
                l = int(_nearest_int(np.float32(0.5) * np.float32(id_ * scales[ib] - np.float32(1.0))).item())
                l = max(0, min(7, l))
                packed_scales[ib // 4] |= np.uint16(l << (3 * (ib % 4)))
                qh[ib] |= masks[int(shifts[ib])]
                xb = x[16 * ib : 16 * ib + 16]
                weight = (xb * xb).astype(np.float32)
                for k in range(2):
                    values = x_p if (shifts[ib] < 2 if k == 0 else shifts[ib] % 2 == 0) else x_m
                    grid_index = int(qs[2 * ib + k]) | (((int(qh[ib]) >> (4 * k)) & 0x07) << 8)
                    levels = grid_l[grid_index]
                    for j in range(8):
                        index = 8 * k + j
                        w = np.float32(weight[index])
                        q = np.float32(values[int(levels[j])] * np.float32(2 * l + 1))
                        sumqx_f = np.float32(sumqx_f + np.float32(np.float32(w * q) * xb[index]))
                        sumq2_f = np.float32(sumq2_f + np.float32(np.float32(w * q) * q))

            if sumq2_f > 0:
                d = np.float32(sumqx_f / sumq2_f)
            scale_bits = np.array([np.float32(d * np.float32(1.1125))], dtype=np.float16).view(np.uint16)[0]
            packed_scales[0] |= np.uint16((scale_bits & 0x000F) << 12)
            packed_scales[1] |= np.uint16((scale_bits & 0x00F0) << 8)
            packed_scales[2] |= np.uint16((scale_bits & 0x0F00) << 4)
            packed_scales[3] |= np.uint16(scale_bits & 0xF000)

        return out

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        qs, rest = np.hsplit(blocks, [QK_K // 8])
        qh, scales = np.hsplit(rest, [QK_K // 16])

        # The f16 scale is packed across multiple bytes
        scales = scales.view(np.uint16)
        d = (scales.reshape((n_blocks, 4)) & np.uint16(0xF000)) >> np.array([12, 8, 4, 0], dtype=np.uint16).reshape((1, 4))
        d = d[..., 0] | d[..., 1] | d[..., 2] | d[..., 3]
        d = d.view(np.float16).astype(np.float32).reshape((n_blocks, 1))

        scales = scales.reshape(n_blocks, -1, 1) >> np.array([0, 3, 6, 9], dtype=np.uint16).reshape((1, 1, 4))
        scales = (scales & 0x07).reshape((n_blocks, -1))
        dl = d * (2 * scales + 1)
        dl = dl.reshape((n_blocks, -1, 2, 1, 1))

        qh = qh.reshape((n_blocks, -1, 1)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2))
        qs = qs.astype(np.uint16) | ((qh & 0x07).astype(np.uint16) << 8).reshape((n_blocks, -1))

        delta = np.where(qh & 0x08 == 0, cls.delta, -cls.delta)
        delta = delta.reshape((n_blocks, -1, 2, 2, 1))

        assert cls.grid is not None
        grid = np.take_along_axis(cls.grid, qs.reshape((n_blocks, -1, 1, 1)), axis=-2)
        grid = grid.reshape((n_blocks, -1, 2, 2, 8))

        return (dl * (grid + delta)).reshape((n_blocks, -1))


class IQ4_NL(__Quant, qtype=GGMLQuantizationType.IQ4_NL):
    kvalues = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)

    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        blocks = blocks.astype(np.float32, copy=False)
        out = np.zeros((n_blocks, cls.type_size), dtype=np.uint8)
        for i, x in enumerate(blocks):
            d, qs, _, _ = _quantize_iq4_nl_impl(x, 32, 32, cls.kvalues, 7)
            out[i, :2] = np.array([d], dtype=np.float16).view(np.uint8)
            out[i, 2:] = qs
        return out

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, qs = np.hsplit(blocks, [2])

        d = d.view(np.float16).astype(np.float32)

        qs = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))

        qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1, 1))

        kvalues = np.array(cls.kvalues, dtype=np.int8).reshape(1, 1, 16)
        qs = np.take_along_axis(kvalues, qs, axis=-1).astype(np.float32).reshape((n_blocks, -1))

        return (d * qs)


class IQ4_XS(__Quant, qtype=GGMLQuantizationType.IQ4_XS):
    @classmethod
    def quantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        blocks = blocks.astype(np.float32, copy=False)
        out = np.zeros((n_blocks, cls.type_size), dtype=np.uint8)
        for i, x in enumerate(blocks):
            d, qs, scales_h, scales_l = _quantize_iq4_nl_impl(
                x, QK_K, 32, IQ4_NL.kvalues, 7
            )
            assert scales_l is not None
            out[i, :2] = np.array([d], dtype=np.float16).view(np.uint8)
            out[i, 2:4] = np.array([scales_h], dtype=np.uint16).view(np.uint8)
            out[i, 4 : 4 + QK_K // 64] = scales_l
            out[i, 4 + QK_K // 64 :] = qs
        return out

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]

        d, rest = np.hsplit(blocks, [2])
        scales_h, rest = np.hsplit(rest, [2])
        scales_l, qs = np.hsplit(rest, [QK_K // 64])

        d = d.view(np.float16).astype(np.float32)
        scales_h = scales_h.view(np.uint16)

        scales_l = scales_l.reshape((n_blocks, -1, 1)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2))
        scales_h = scales_h.reshape((n_blocks, 1, -1)) >> np.array([2 * i for i in range(QK_K // 32)], dtype=np.uint16).reshape((1, -1, 1))
        scales_l = scales_l.reshape((n_blocks, -1)) & np.uint8(0x0F)
        scales_h = scales_h.reshape((n_blocks, -1)).astype(np.uint8) & np.uint8(0x03)

        scales = (scales_l | (scales_h << np.uint8(4))).astype(np.int8) - np.int8(32)
        dl = (d * scales.astype(np.float32)).reshape((n_blocks, -1, 1))

        qs = qs.reshape((n_blocks, -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = qs.reshape((n_blocks, -1, 32, 1)) & np.uint8(0x0F)

        kvalues = np.array(IQ4_NL.kvalues, dtype=np.int8).reshape((1, 1, 1, -1))
        qs = np.take_along_axis(kvalues, qs, axis=-1).astype(np.float32).reshape((n_blocks, -1, 32))

        return (dl * qs).reshape((n_blocks, -1))
