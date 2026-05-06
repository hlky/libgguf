#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "libgguf_cuda_common.h"
#ifdef GGUF_CUDA_USE_IQ1_GRID_LOOKUP
#include "libgguf_cuda_iq1_lookup.cuh"
#endif

static __device__ __forceinline__ uint8_t gguf_cuda_min_u8(uint8_t lhs, uint8_t rhs) {
    return lhs < rhs ? lhs : rhs;
}

static __device__ __forceinline__ int gguf_cuda_clamp_int(int value, int low, int high) {
    return value < low ? low : value > high ? high : value;
}

static __device__ __forceinline__ float gguf_cuda_warp_reduce_max(float value) {
    constexpr unsigned mask = 0xffffffffu;
    value = fmaxf(value, __shfl_down_sync(mask, value, 16));
    value = fmaxf(value, __shfl_down_sync(mask, value, 8));
    value = fmaxf(value, __shfl_down_sync(mask, value, 4));
    value = fmaxf(value, __shfl_down_sync(mask, value, 2));
    value = fmaxf(value, __shfl_down_sync(mask, value, 1));
    return value;
}

static __device__ __forceinline__ float gguf_cuda_warp_reduce_min(float value) {
    constexpr unsigned mask = 0xffffffffu;
    value = fminf(value, __shfl_down_sync(mask, value, 16));
    value = fminf(value, __shfl_down_sync(mask, value, 8));
    value = fminf(value, __shfl_down_sync(mask, value, 4));
    value = fminf(value, __shfl_down_sync(mask, value, 2));
    value = fminf(value, __shfl_down_sync(mask, value, 1));
    return value;
}

static __device__ __forceinline__ float gguf_cuda_warp_reduce_absmax_first(float value, int lane) {
    constexpr unsigned mask = 0xffffffffu;
    float best_abs = fabsf(value);
    int best_idx = lane;
    float best_value = value;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_abs = __shfl_down_sync(mask, best_abs, offset);
        const int other_idx = __shfl_down_sync(mask, best_idx, offset);
        const float other_value = __shfl_down_sync(mask, best_value, offset);
        if (other_abs > best_abs || (other_abs == best_abs && other_idx < best_idx)) {
            best_abs = other_abs;
            best_idx = other_idx;
            best_value = other_value;
        }
    }
    return __shfl_sync(mask, best_value, 0);
}


static __device__ __forceinline__ float gguf_cuda_e8m0_to_fp32_half(uint8_t value) {
    const uint32_t bits = value < 2 ? (0x00200000u << value) : ((uint32_t)(value - 1) << 23);
    return gguf_cuda_bits_to_fp32(bits);
}

static __device__ __forceinline__ float gguf_cuda_ue4m3_to_fp32(uint8_t value) {
    if (value == 0 || value == 0x7f) {
        return 0.0f;
    }
    const int exp = (value >> 3) & 0x0f;
    const int man = value & 0x07;
    const float raw = exp == 0 ? ldexpf((float)man, -9) : ldexpf(1.0f + (float)man / 8.0f, exp - 7);
    return raw * 0.5f;
}

static __device__ __forceinline__ uint8_t gguf_cuda_fp32_to_ue4m3(float x) {
    if (!(x > 0.0f)) {
        return 0;
    }
    if (x > 448.0f) {
        x = 448.0f;
    }
    const uint32_t bits = gguf_cuda_fp32_to_bits(x);
    const int fp32_exp = ((bits >> 23) & 0xff) - 127;
    const int fp32_man = (bits >> 20) & 0x07;
    int ue4m3_exp = fp32_exp + 7;
    if (ue4m3_exp <= 0) {
        int man = (int)(x * 512.0f + 0.5f);
        if (man > 7) {
            man = 7;
        }
        if (man < 1) {
            return 0;
        }
        return (uint8_t)man;
    }
    if (ue4m3_exp >= 15) {
        return 0x7e;
    }
    const int round_bit = (bits >> 19) & 1;
    int ue4m3_man = fp32_man + round_bit;
    if (ue4m3_man > 7) {
        ue4m3_man = 0;
        ++ue4m3_exp;
        if (ue4m3_exp >= 15) {
            return 0x7e;
        }
    }
    return (uint8_t)((ue4m3_exp << 3) | ue4m3_man);
}

static __device__ __forceinline__ int gguf_cuda_best_index_mxfp4(float x, float e) {
    int best_index = 0;
    float best_err = fabsf(kvalues_e2m1[0] * e - x);
    for (int i = 1; i < 16; ++i) {
        const float err = fabsf(kvalues_e2m1[i] * e - x);
        if (err < best_err) {
            best_index = i;
            best_err = err;
        }
    }
    return best_index;
}


static __device__ __forceinline__ void gguf_cuda_store_u32_le(uint8_t * dst, uint32_t value) {
    dst[0] = value & 0xffu;
    dst[1] = (value >> 8) & 0xffu;
    dst[2] = (value >> 16) & 0xffu;
    dst[3] = (value >> 24) & 0xffu;
}


static __device__ __forceinline__ uint8_t gguf_cuda_pack_trits_5(uint8_t q) {
    return ((uint16_t)q * 256 + (243 - 1)) / 243;
}

static __device__ __forceinline__ float gguf_cuda_make_qx_quants_rmse1(
    int n, int nmax, const float * x, int8_t * l
) {
    float max = 0.0f;
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float ax = fabsf(x[i]);
        if (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }

    if (amax < GROUP_MAX_EPS) {
        for (int i = 0; i < n; ++i) {
            l[i] = 0;
        }
        return 0.0f;
    }

    float iscale = __fdiv_rn(-float(nmax), max);
    float sumlx = 0.0f;
    float suml2 = 0.0f;
    for (int i = 0; i < n; ++i) {
        int q = gguf_cuda_nearest_int(__fmul_rn(iscale, x[i]));
        q = gguf_cuda_clamp_int(q, -nmax, nmax - 1);
        l[i] = q + nmax;
        const float w = __fmul_rn(x[i], x[i]);
        sumlx = __fadd_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), float(q)));
        suml2 = __fadd_rn(suml2, __fmul_rn(__fmul_rn(w, float(q)), float(q)));
    }

    float scale = suml2 != 0.0f ? __fdiv_rn(sumlx, suml2) : 0.0f;
    float best = __fmul_rn(scale, sumlx);
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) {
            continue;
        }
        iscale = __fdiv_rn(-(float(nmax) + __fmul_rn(0.1f, float(is))), max);
        sumlx = 0.0f;
        suml2 = 0.0f;
        for (int i = 0; i < n; ++i) {
            int q = gguf_cuda_nearest_int(__fmul_rn(iscale, x[i]));
            q = gguf_cuda_clamp_int(q, -nmax, nmax - 1);
            const float w = __fmul_rn(x[i], x[i]);
            sumlx = __fadd_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), float(q)));
            suml2 = __fadd_rn(suml2, __fmul_rn(__fmul_rn(w, float(q)), float(q)));
        }
        if (suml2 > 0.0f && __fmul_rn(sumlx, sumlx) > __fmul_rn(best, suml2)) {
            for (int i = 0; i < n; ++i) {
                int q = gguf_cuda_nearest_int(__fmul_rn(iscale, x[i]));
                l[i] = nmax + gguf_cuda_clamp_int(q, -nmax, nmax - 1);
            }
            scale = __fdiv_rn(sumlx, suml2);
            best = __fmul_rn(scale, sumlx);
        }
    }
    return scale;
}

static __device__ __forceinline__ float gguf_cuda_make_qkx2_quants(
    int n,
    int nmax,
    const float * x,
    const float * weights,
    uint8_t * l,
    float * the_min,
    uint8_t * laux,
    float rmin,
    float rdelta,
    int nstep,
    bool use_mad
) {
    float minv = x[0];
    float maxv = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < minv) {
            minv = x[i];
        }
        if (x[i] > maxv) {
            maxv = x[i];
        }
        const float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }

    if (minv > 0.0f) {
        minv = 0.0f;
    }
    if (maxv == minv) {
        for (int i = 0; i < n; ++i) {
            l[i] = 0;
        }
        *the_min = -minv;
        return 0.0f;
    }

    float iscale = nmax / (maxv - minv);
    float scale = 1.0f / iscale;
    float best_error = 0.0f;
    for (int i = 0; i < n; ++i) {
        int q = gguf_cuda_nearest_int(iscale * (x[i] - minv));
        q = gguf_cuda_clamp_int(q, 0, nmax);
        l[i] = q;
        float diff = scale * l[i] + minv - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        best_error += weights[i] * diff;
    }

    if (nstep < 1) {
        *the_min = -minv;
        return scale;
    }

    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta * is + nmax) / (maxv - minv);
        float sum_l = 0.0f;
        float sum_l2 = 0.0f;
        float sum_xl = 0.0f;
        for (int i = 0; i < n; ++i) {
            int q = gguf_cuda_nearest_int(iscale * (x[i] - minv));
            q = gguf_cuda_clamp_int(q, 0, nmax);
            laux[i] = q;
            const float w = weights[i];
            sum_l += w * q;
            sum_l2 += w * q * q;
            sum_xl += w * q * x[i];
        }
        const float d_det = sum_w * sum_l2 - sum_l * sum_l;
        if (d_det > 0.0f) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / d_det;
            float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d_det;
            if (this_min > 0.0f) {
                this_min = 0.0f;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0.0f;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                cur_error += weights[i] * diff;
            }
            if (cur_error < best_error) {
                for (int i = 0; i < n; ++i) {
                    l[i] = laux[i];
                }
                best_error = cur_error;
                scale = this_scale;
                minv = this_min;
            }
        }
    }

    *the_min = -minv;
    return scale;
}


static __device__ __forceinline__ float gguf_cuda_make_q3_quants(
    int n, int nmax, const float * x, int8_t * l
) {
    float max = 0.0f;
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float ax = fabsf(x[i]);
        if (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }

    if (amax < GROUP_MAX_EPS) {
        for (int i = 0; i < n; ++i) {
            l[i] = 0;
        }
        return 0.0f;
    }

    const float iscale = -float(nmax) / max;
    float sumlx = 0.0f;
    float suml2 = 0.0f;
    for (int i = 0; i < n; ++i) {
        int q = gguf_cuda_nearest_int(iscale * x[i]);
        q = gguf_cuda_clamp_int(q, -nmax, nmax - 1);
        l[i] = q;
        const float w = x[i] * x[i];
        sumlx += w * x[i] * q;
        suml2 += w * q * q;
    }

    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            const float w = x[i] * x[i];
            float slx = sumlx - w * x[i] * l[i];
            if (slx > 0.0f) {
                float sl2 = suml2 - w * l[i] * l[i];
                int new_l = gguf_cuda_nearest_int(x[i] * sl2 / slx);
                new_l = gguf_cuda_clamp_int(new_l, -nmax, nmax - 1);
                if (new_l != l[i]) {
                    slx += w * x[i] * new_l;
                    sl2 += w * new_l * new_l;
                    if (sl2 > 0.0f && slx * slx * suml2 > sumlx * sumlx * sl2) {
                        l[i] = new_l;
                        sumlx = slx;
                        suml2 = sl2;
                        ++n_changed;
                    }
                }
            }
        }
        if (!n_changed) {
            break;
        }
    }

    for (int i = 0; i < n; ++i) {
        l[i] += nmax;
    }
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}


static __device__ __forceinline__ void gguf_cuda_get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}


static __device__ __forceinline__ int gguf_cuda_best_index_int8(int n, const int8_t * values, float x) {
    if (x <= values[0]) {
        return 0;
    }
    if (x >= values[n - 1]) {
        return n - 1;
    }
    int ml = 0;
    int mu = n - 1;
    while (mu - ml > 1) {
        const int mav = (ml + mu) / 2;
        if (x < values[mav]) {
            mu = mav;
        } else {
            ml = mav;
        }
    }
    return x - values[mu - 1] < values[mu] - x ? mu - 1 : mu;
}

static __device__ __forceinline__ void gguf_cuda_quantize_iq4_nl_impl(
    int super_block_size,
    int block_size,
    const float * x,
    ggml_half * dh,
    uint8_t * q4,
    uint16_t * scales_h,
    uint8_t * scales_l,
    float * scales,
    float * weight,
    uint8_t * l,
    int ntry
) {
    float sigma2 = 0.0f;
    for (int j = 0; j < super_block_size; ++j) {
        sigma2 += x[j] * x[j];
    }
    sigma2 *= 2.0f / super_block_size;

    for (int j = 0; j < super_block_size / 2; ++j) {
        q4[j] = 0;
    }
    dh[0] = gguf_cuda_compute_fp32_to_fp16(0.0f);

    float max_scale = 0.0f;
    float amax_scale = 0.0f;
    for (int ib = 0; ib < super_block_size / block_size; ++ib) {
        const float * xb = x + ib * block_size;
        uint8_t * lb = l + ib * block_size;
        for (int j = 0; j < block_size; ++j) {
            weight[j] = xb[j] * xb[j];
        }

        float amax = 0.0f;
        float max = 0.0f;
        for (int j = 0; j < block_size; ++j) {
            const float ax = fabsf(xb[j]);
            if (ax > amax) {
                amax = ax;
                max = xb[j];
            }
        }
        if (amax < GROUP_MAX_EPS) {
            scales[ib] = 0.0f;
            continue;
        }

        float d = ntry > 0 ? -max / kvalues_iq4nl[0] : max / kvalues_iq4nl[0];
        float id = 1.0f / d;
        float sumqx = 0.0f;
        float sumq2 = 0.0f;
        for (int j = 0; j < block_size; ++j) {
            const float al = id * xb[j];
            const int qindex = gguf_cuda_best_index_int8(16, kvalues_iq4nl, al);
            lb[j] = qindex;
            const float q = kvalues_iq4nl[qindex];
            const float w = weight[j];
            sumqx += w * q * xb[j];
            sumq2 += w * q * q;
        }
        d = sumq2 > 0.0f ? sumqx / sumq2 : 0.0f;
        float best = d * sumqx;
        for (int itry = -ntry; itry <= ntry; ++itry) {
            id = (itry + kvalues_iq4nl[0]) / max;
            sumqx = 0.0f;
            sumq2 = 0.0f;
            for (int j = 0; j < block_size; ++j) {
                const float al = id * xb[j];
                const int qindex = gguf_cuda_best_index_int8(16, kvalues_iq4nl, al);
                const float q = kvalues_iq4nl[qindex];
                const float w = weight[j];
                sumqx += w * q * xb[j];
                sumq2 += w * q * q;
            }
            if (sumq2 > 0.0f && sumqx * sumqx > best * sumq2) {
                d = sumqx / sumq2;
                best = d * sumqx;
            }
        }
        scales[ib] = d;
        const float abs_d = fabsf(d);
        if (abs_d > amax_scale) {
            amax_scale = abs_d;
            max_scale = d;
        }
    }

    if (super_block_size / block_size > 1) {
        const int nb = super_block_size / block_size;
        for (int j = 0; j < (nb + 7) / 8; ++j) {
            scales_h[j] = 0;
        }
        const float d = -max_scale / 32.0f;
        dh[0] = gguf_cuda_compute_fp32_to_fp16(d);
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        for (int ib = 0; ib < nb; ++ib) {
            int qscale = gguf_cuda_nearest_int(id * scales[ib]);
            qscale = gguf_cuda_clamp_int(qscale, -32, 31);
            const float dl = d * qscale;
            const float idl = dl != 0.0f ? 1.0f / dl : 0.0f;
            uint8_t * lb = l + ib * block_size;
            const float * xb = x + ib * block_size;
            for (int j = 0; j < block_size; ++j) {
                lb[j] = gguf_cuda_best_index_int8(16, kvalues_iq4nl, idl * xb[j]);
            }
            qscale += 32;
            const uint8_t l_l = qscale & 0x0f;
            const uint8_t l_h = qscale >> 4;
            if (ib % 2 == 0) {
                scales_l[ib / 2] = l_l;
            } else {
                scales_l[ib / 2] |= l_l << 4;
            }
            scales_h[ib / 8] |= l_h << (2 * (ib % 8));
        }
    } else {
        dh[0] = gguf_cuda_compute_fp32_to_fp16(scales[0]);
        if (ntry > 0) {
            const float id = scales[0] != 0.0f ? 1.0f / scales[0] : 0.0f;
            for (int j = 0; j < super_block_size; ++j) {
                l[j] = gguf_cuda_best_index_int8(16, kvalues_iq4nl, id * x[j]);
            }
        }
    }

    for (int i = 0; i < super_block_size / 32; ++i) {
        for (int j = 0; j < 16; ++j) {
            q4[16 * i + j] = l[32 * i + j] | (l[32 * i + 16 + j] << 4);
        }
    }
}


static __device__ __forceinline__ float gguf_cuda_make_qp_quants(
    int n, int nmax, const float * x, uint8_t * l, const float * quant_weights
) {
    float max = 0.0f;
    for (int i = 0; i < n; ++i) {
        max = fmaxf(max, x[i]);
    }
    if (max < GROUP_MAX_EPS) {
        for (int i = 0; i < n; ++i) {
            l[i] = 0;
        }
        return 0.0f;
    }

    float iscale = nmax / max;
    for (int i = 0; i < n; ++i) {
        l[i] = gguf_cuda_nearest_int(iscale * x[i]);
    }
    float scale = 1.0f / iscale;
    float best_mse = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float diff = x[i] - scale * l[i];
        best_mse += quant_weights[i] * diff * diff;
    }
    for (int is = -4; is <= 4; ++is) {
        if (is == 0) {
            continue;
        }
        const float iscale_is = (0.1f * is + nmax) / max;
        const float scale_is = 1.0f / iscale_is;
        float mse = 0.0f;
        for (int i = 0; i < n; ++i) {
            int q = gguf_cuda_nearest_int(iscale_is * x[i]);
            q = min(nmax, q);
            const float diff = x[i] - scale_is * q;
            mse += quant_weights[i] * diff * diff;
        }
        if (mse < best_mse) {
            best_mse = mse;
            iscale = iscale_is;
        }
    }

    float sumlx = 0.0f;
    float suml2 = 0.0f;
    for (int i = 0; i < n; ++i) {
        int q = gguf_cuda_nearest_int(iscale * x[i]);
        q = min(nmax, q);
        l[i] = q;
        const float w = quant_weights[i];
        sumlx += w * x[i] * q;
        suml2 += w * q * q;
    }
    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            const float w = quant_weights[i];
            float slx = sumlx - w * x[i] * l[i];
            float sl2 = suml2 - w * l[i] * l[i];
            if (slx > 0.0f && sl2 > 0.0f) {
                int new_l = gguf_cuda_nearest_int(x[i] * sl2 / slx);
                new_l = min(nmax, new_l);
                if (new_l != l[i]) {
                    slx += w * x[i] * new_l;
                    sl2 += w * new_l * new_l;
                    if (slx * slx * suml2 > sumlx * sumlx * sl2) {
                        l[i] = new_l;
                        sumlx = slx;
                        suml2 = sl2;
                        ++n_changed;
                    }
                }
            }
        }
        if (!n_changed) {
            break;
        }
    }
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

static __device__ __forceinline__ int8_t gguf_cuda_iq2_grid_l(const uint64_t * grid, int grid_index, int i) {
    const uint8_t v = ((const uint8_t *)(grid + grid_index))[i];
    return v == 0x08 ? 0 : v == 0x19 ? 1 : 2;
}

static __device__ __forceinline__ int8_t gguf_cuda_iq2_grid_q(const uint64_t * grid, int grid_index, int i) {
    return 2 * gguf_cuda_iq2_grid_l(grid, grid_index, i) + 1;
}

static __device__ __forceinline__ uint64_t gguf_cuda_iq2_grid_key(const int8_t * l) {
    uint64_t key = 0;
    for (int i = 0; i < 8; ++i) {
        if (l[i] < 0 || l[i] > 2) {
            return UINT64_MAX;
        }
        const uint64_t v = l[i] == 0 ? 0x08u : l[i] == 1 ? 0x19u : 0x2bu;
        key |= v << (8 * i);
    }
    return key;
}

static __device__ __forceinline__ int gguf_cuda_iq2_find_grid_index(const uint64_t * grid, int grid_size, const int8_t * l) {
    const uint64_t key = gguf_cuda_iq2_grid_key(l);
    int lo = 0;
    int hi = grid_size - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        const uint64_t value = grid[mid];
        if (value < key) {
            lo = mid + 1;
        } else if (value > key) {
            hi = mid - 1;
        } else {
            return mid;
        }
    }
    return -1;
}

static __device__ __forceinline__ int gguf_cuda_iq2_grid_dist(const uint64_t * grid, int grid_index, const int8_t * l) {
    int dist = 0;
    for (int i = 0; i < 8; ++i) {
        const int diff = (int)gguf_cuda_iq2_grid_l(grid, grid_index, i) - (int)l[i];
        dist += diff * diff;
    }
    return dist;
}

static __device__ __forceinline__ int gguf_cuda_iq2_find_best_neighbour(
    const uint64_t * grid, int grid_size, int nwant, const float * xval, const float * weight, float scale, int8_t * l
) {
    int cutoff_dist = INT_MAX;
    if (nwant <= 1) {
        for (int grid_index = 0; grid_index < grid_size; ++grid_index) {
            cutoff_dist = min(cutoff_dist, gguf_cuda_iq2_grid_dist(grid, grid_index, l));
        }
    } else {
        int shell0 = INT_MAX;
        int shell1 = INT_MAX;
        int shell2 = INT_MAX;
        for (int grid_index = 0; grid_index < grid_size; ++grid_index) {
            const int dist = gguf_cuda_iq2_grid_dist(grid, grid_index, l);
            if (dist == shell0 || dist == shell1 || dist == shell2) {
                continue;
            }
            if (dist < shell0) {
                shell2 = shell1;
                shell1 = shell0;
                shell0 = dist;
            } else if (dist < shell1) {
                shell2 = shell1;
                shell1 = dist;
            } else if (nwant > 2 && dist < shell2) {
                shell2 = dist;
            }
        }
        cutoff_dist = shell0;
        if (nwant > 1 && shell1 != INT_MAX) {
            cutoff_dist = shell1;
        }
        if (nwant > 2 && shell2 != INT_MAX) {
            cutoff_dist = shell2;
        }
    }

    float best_d2 = FLT_MAX;
    int best_index = 0;
    for (int grid_index = 0; grid_index < grid_size; ++grid_index) {
        const int dist = gguf_cuda_iq2_grid_dist(grid, grid_index, l);
        if (dist > cutoff_dist) {
            continue;
        }
        float d2 = 0.0f;
        for (int i = 0; i < 8; ++i) {
            const float q = gguf_cuda_iq2_grid_q(grid, grid_index, i);
            const float diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2;
            best_index = grid_index;
        }
    }
    for (int i = 0; i < 8; ++i) {
        l[i] = gguf_cuda_iq2_grid_l(grid, best_index, i);
    }
    return best_index;
}

static __device__ __forceinline__ int8_t gguf_cuda_iq1_grid_l(int grid_index, int i) {
    const int8_t v = ((const int8_t *)(iq1s_grid + grid_index))[i];
    return v + 1;
}

static __device__ __forceinline__ int gguf_cuda_iq1_find_grid_index(const int8_t * l) {
#ifdef GGUF_CUDA_USE_IQ1_GRID_LOOKUP
    int key = 0;
    int mul = 1;
    for (int i = 0; i < 8; ++i) {
        if (l[i] < 0 || l[i] > 2) {
            return -1;
        }
        key += l[i] * mul;
        mul *= 3;
    }
    int lo = 0;
    int hi = IQ1S_GRID_LOOKUP_SIZE - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        const uint32_t entry = iq1s_grid_lookup[mid];
        const int entry_key = entry >> 12;
        if (entry_key < key) {
            lo = mid + 1;
        } else if (entry_key > key) {
            hi = mid - 1;
        } else {
            return entry & 0x0fffu;
        }
    }
    return -1;
#else
    for (int grid_index = 0; grid_index < NGRID_IQ1S; ++grid_index) {
        bool match = true;
        for (int i = 0; i < 8; ++i) {
            if (gguf_cuda_iq1_grid_l(grid_index, i) != l[i]) {
                match = false;
                break;
            }
        }
        if (match) {
            return grid_index;
        }
    }
    return -1;
#endif
}

static __device__ __forceinline__ int gguf_cuda_iq1_grid_dist(int grid_index, const int8_t * l) {
    int dist = 0;
    for (int i = 0; i < 8; ++i) {
        const int diff = (int)gguf_cuda_iq1_grid_l(grid_index, i) - (int)l[i];
        dist += diff * diff;
    }
    return dist;
}

static __device__ __forceinline__ int gguf_cuda_iq1_find_best_neighbour(
    const float * xval, const float * weight, float scale, const float * values, int8_t * l
) {
    int shell0 = INT_MAX;
    int shell1 = INT_MAX;
    int shell2 = INT_MAX;
    for (int grid_index = 0; grid_index < NGRID_IQ1S; ++grid_index) {
        const int dist = gguf_cuda_iq1_grid_dist(grid_index, l);
        if (dist == shell0 || dist == shell1 || dist == shell2) {
            continue;
        }
        if (dist < shell0) {
            shell2 = shell1;
            shell1 = shell0;
            shell0 = dist;
        } else if (dist < shell1) {
            shell2 = shell1;
            shell1 = dist;
        } else if (dist < shell2) {
            shell2 = dist;
        }
    }
    int cutoff_dist = shell0;
    if (shell1 != INT_MAX) {
        cutoff_dist = shell1;
    }
    if (shell2 != INT_MAX) {
        cutoff_dist = shell2;
    }

    float best_d2 = FLT_MAX;
    int best_index = 0;
    for (int grid_index = 0; grid_index < NGRID_IQ1S; ++grid_index) {
        const int dist = gguf_cuda_iq1_grid_dist(grid_index, l);
        if (dist > cutoff_dist) {
            continue;
        }
        float d2 = 0.0f;
        for (int i = 0; i < 8; ++i) {
            const float q = values[gguf_cuda_iq1_grid_l(grid_index, i)];
            const float diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2;
            best_index = grid_index;
        }
    }
    for (int i = 0; i < 8; ++i) {
        l[i] = gguf_cuda_iq1_grid_l(best_index, i);
    }
    return best_index;
}

static __device__ __forceinline__ int8_t gguf_cuda_iq3_grid_l(const uint32_t * grid, int grid_index, int i) {
    const uint8_t v = ((const uint8_t *)(grid + grid_index))[i];
    if (v & 1) {
        return (v - 1) / 2;
    }
    return v == 0x3e ? 7 : (v - 0x04) / 8;
}

static __device__ __forceinline__ int8_t gguf_cuda_iq3_grid_q(const uint32_t * grid, int grid_index, int i) {
    return 2 * gguf_cuda_iq3_grid_l(grid, grid_index, i) + 1;
}

static __device__ __forceinline__ uint32_t gguf_cuda_iq3_grid_key(const uint32_t * grid, const int8_t * l) {
    const bool odd_encoding = (grid[0] & 1u) != 0;
    uint32_t key = 0;
    for (int i = 0; i < 4; ++i) {
        if (l[i] < 0 || l[i] > 7) {
            return UINT32_MAX;
        }
        const uint32_t v = odd_encoding ? (uint32_t)(2 * l[i] + 1) : l[i] == 7 ? 0x3eu : (uint32_t)(0x04u + 8u * l[i]);
        key |= v << (8 * i);
    }
    return key;
}

static __device__ __forceinline__ int gguf_cuda_iq3_find_grid_index(const uint32_t * grid, int grid_size, const int8_t * l) {
    const uint32_t key = gguf_cuda_iq3_grid_key(grid, l);
    int lo = 0;
    int hi = grid_size - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        const uint32_t value = grid[mid];
        if (value < key) {
            lo = mid + 1;
        } else if (value > key) {
            hi = mid - 1;
        } else {
            return mid;
        }
    }
    return -1;
}

static __device__ __forceinline__ int gguf_cuda_iq3_grid_dist(const uint32_t * grid, int grid_index, const int8_t * l) {
    int dist = 0;
    for (int i = 0; i < 4; ++i) {
        const int diff = (int)gguf_cuda_iq3_grid_l(grid, grid_index, i) - (int)l[i];
        dist += diff * diff;
    }
    return dist;
}

static __device__ __forceinline__ int gguf_cuda_iq3_find_best_neighbour(
    const uint32_t * grid, int grid_size, int nwant, const float * xval, const float * weight, float scale, int8_t * l
) {
    int cutoff_dist = INT_MAX;
    if (nwant <= 1) {
        for (int grid_index = 0; grid_index < grid_size; ++grid_index) {
            cutoff_dist = min(cutoff_dist, gguf_cuda_iq3_grid_dist(grid, grid_index, l));
        }
    } else {
        int shell0 = INT_MAX;
        int shell1 = INT_MAX;
        int shell2 = INT_MAX;
        for (int grid_index = 0; grid_index < grid_size; ++grid_index) {
            const int dist = gguf_cuda_iq3_grid_dist(grid, grid_index, l);
            if (dist == shell0 || dist == shell1 || dist == shell2) {
                continue;
            }
            if (dist < shell0) {
                shell2 = shell1;
                shell1 = shell0;
                shell0 = dist;
            } else if (dist < shell1) {
                shell2 = shell1;
                shell1 = dist;
            } else if (nwant > 2 && dist < shell2) {
                shell2 = dist;
            }
        }
        cutoff_dist = shell0;
        if (nwant > 1 && shell1 != INT_MAX) {
            cutoff_dist = shell1;
        }
        if (nwant > 2 && shell2 != INT_MAX) {
            cutoff_dist = shell2;
        }
    }

    float best_d2 = FLT_MAX;
    int best_index = 0;
    for (int grid_index = 0; grid_index < grid_size; ++grid_index) {
        const int dist = gguf_cuda_iq3_grid_dist(grid, grid_index, l);
        if (dist > cutoff_dist) {
            continue;
        }
        float d2 = 0.0f;
        for (int i = 0; i < 4; ++i) {
            const float q = gguf_cuda_iq3_grid_q(grid, grid_index, i);
            const float diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2;
            best_index = grid_index;
        }
    }
    for (int i = 0; i < 4; ++i) {
        l[i] = gguf_cuda_iq3_grid_l(grid, best_index, i);
    }
    return best_index;
}
