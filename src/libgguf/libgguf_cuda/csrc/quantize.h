#pragma once

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

static __global__ void quantize_block_q8_0_warp(const float * __restrict__ x, block_q8_0 * __restrict__ y, int64_t n_blocks) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t ib = (int64_t)blockIdx.x * warps_per_block + warp;
    if (ib >= n_blocks) {
        return;
    }

    constexpr unsigned mask = 0xffffffffu;
    const float v = x[ib * QK8_0 + lane];
    const float amax = __shfl_sync(mask, gguf_cuda_warp_reduce_max(fabsf(v)), 0);
    const float d = __fdiv_rn(amax, 127.0f);
    const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
    if (lane == 0) {
        y[ib].d = gguf_cuda_compute_fp32_to_fp16(d);
    }
    y[ib].qs[lane] = (int8_t)roundf(__fmul_rn(v, id));
}

static __global__ void quantize_block_q1_0(const float * __restrict__ x, block_q1_0 * __restrict__ y, int64_t n_blocks) {
    const int64_t ib = blockDim.x * blockIdx.x + threadIdx.x;
    if (ib >= n_blocks) {
        return;
    }

    const float * xb = x + ib * QK1_0;
    float sum_abs = 0.0f;
    for (int j = 0; j < QK1_0; ++j) {
        sum_abs += fabsf(xb[j]);
    }
    y[ib].d = gguf_cuda_compute_fp32_to_fp16(sum_abs / QK1_0);

    for (int j = 0; j < QK1_0 / 8; ++j) {
        uint8_t q = 0;
        for (int bit = 0; bit < 8; ++bit) {
            if (xb[8 * j + bit] >= 0.0f) {
                q |= 1 << bit;
            }
        }
        y[ib].qs[j] = q;
    }
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

static __global__ void quantize_block_mxfp4(const float * __restrict__ x, block_mxfp4 * __restrict__ y, int64_t n_blocks) {
    const int64_t ib = blockDim.x * blockIdx.x + threadIdx.x;
    if (ib >= n_blocks) {
        return;
    }

    const float * xb = x + ib * QK_MXFP4;
    float amax = 0.0f;
    for (int j = 0; j < QK_MXFP4; ++j) {
        amax = fmaxf(amax, fabsf(xb[j]));
    }

    const uint8_t e = amax > 0.0f ? (uint8_t)(floorf(log2f(amax)) - 2.0f + 127.0f) : 0;
    const float d = gguf_cuda_e8m0_to_fp32_half(e);
    y[ib].e = e;

    for (int j = 0; j < QK_MXFP4 / 2; ++j) {
        const uint8_t x0 = gguf_cuda_best_index_mxfp4(xb[j], d);
        const uint8_t x1 = gguf_cuda_best_index_mxfp4(xb[QK_MXFP4 / 2 + j], d);
        y[ib].qs[j] = x0 | (x1 << 4);
    }
}

static __global__ void quantize_block_nvfp4(const float * __restrict__ x, block_nvfp4 * __restrict__ y, int64_t n_blocks) {
    const int64_t ib = blockDim.x * blockIdx.x + threadIdx.x;
    if (ib >= n_blocks) {
        return;
    }

    const float * xb0 = x + ib * QK_NVFP4;
    for (int s = 0; s < QK_NVFP4 / QK_NVFP4_SUB; ++s) {
        const float * xb = xb0 + s * QK_NVFP4_SUB;
        float amax = 0.0f;
        for (int j = 0; j < QK_NVFP4_SUB; ++j) {
            amax = fmaxf(amax, fabsf(xb[j]));
        }
        const uint8_t ue = gguf_cuda_fp32_to_ue4m3(amax / 6.0f);
        y[ib].d[s] = ue;
        const float d = gguf_cuda_ue4m3_to_fp32(ue);
        for (int j = 0; j < QK_NVFP4_SUB / 2; ++j) {
            const uint8_t x0 = gguf_cuda_best_index_mxfp4(xb[j], d);
            const uint8_t x1 = gguf_cuda_best_index_mxfp4(xb[QK_NVFP4_SUB / 2 + j], d);
            y[ib].qs[s * (QK_NVFP4_SUB / 2) + j] = x0 | (x1 << 4);
        }
    }
}

static __global__ void quantize_block_q4_0(const float * __restrict__ x, block_q4_0 * __restrict__ y, int64_t n_blocks) {
    const int64_t ib = blockDim.x * blockIdx.x + threadIdx.x;
    if (ib >= n_blocks) {
        return;
    }

    const float * xb = x + ib * QK4_0;
    float amax = 0.0f;
    float max = 0.0f;
    for (int j = 0; j < QK4_0; ++j) {
        const float v = xb[j];
        const float av = fabsf(v);
        if (amax < av) {
            amax = av;
            max = v;
        }
    }

    const float d = __fdiv_rn(max, -8.0f);
    const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
    y[ib].d = gguf_cuda_compute_fp32_to_fp16(d);

    for (int j = 0; j < QK4_0 / 2; ++j) {
        const float x0 = __fmul_rn(xb[j], id);
        const float x1 = __fmul_rn(xb[QK4_0 / 2 + j], id);

        const uint8_t xi0 = gguf_cuda_min_u8(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = gguf_cuda_min_u8(15, (int8_t)(x1 + 8.5f));

        y[ib].qs[j] = xi0 | (xi1 << 4);
    }
}

static __global__ void quantize_block_q4_1(const float * __restrict__ x, block_q4_1 * __restrict__ y, int64_t n_blocks) {
    const int64_t ib = blockDim.x * blockIdx.x + threadIdx.x;
    if (ib >= n_blocks) {
        return;
    }

    const float * xb = x + ib * QK4_1;
    float min = 3.4028234663852886e38f;
    float max = -3.4028234663852886e38f;
    for (int j = 0; j < QK4_1; ++j) {
        const float v = xb[j];
        if (v < min) {
            min = v;
        }
        if (v > max) {
            max = v;
        }
    }

    const float d = __fdiv_rn(max - min, 15.0f);
    const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
    y[ib].d = gguf_cuda_compute_fp32_to_fp16(d);
    y[ib].m = gguf_cuda_compute_fp32_to_fp16(min);

    for (int j = 0; j < QK4_1 / 2; ++j) {
        const float x0 = __fmul_rn(xb[j] - min, id);
        const float x1 = __fmul_rn(xb[QK4_1 / 2 + j] - min, id);

        const uint8_t xi0 = gguf_cuda_min_u8(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = gguf_cuda_min_u8(15, (int8_t)(x1 + 0.5f));

        y[ib].qs[j] = xi0 | (xi1 << 4);
    }
}

static __device__ __forceinline__ void gguf_cuda_store_u32_le(uint8_t * dst, uint32_t value) {
    dst[0] = value & 0xffu;
    dst[1] = (value >> 8) & 0xffu;
    dst[2] = (value >> 16) & 0xffu;
    dst[3] = (value >> 24) & 0xffu;
}

static __global__ void quantize_block_q5_0_warp(const float * __restrict__ x, block_q5_0 * __restrict__ y, int64_t n_blocks) {
    constexpr unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t ib = (int64_t)blockIdx.x * warps_per_block + warp;
    if (ib >= n_blocks) {
        return;
    }

    const float * xb = x + ib * QK5_0;
    const float max = gguf_cuda_warp_reduce_absmax_first(xb[lane], lane);
    const float d = __fdiv_rn(max, -16.0f);
    const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
    uint8_t xi0 = 0;
    uint8_t xi1 = 0;
    if (lane < QK5_0 / 2) {
        const float x0 = __fmul_rn(xb[lane], id);
        const float x1 = __fmul_rn(xb[QK5_0 / 2 + lane], id);
        xi0 = gguf_cuda_min_u8(31, (int8_t)(x0 + 16.5f));
        xi1 = gguf_cuda_min_u8(31, (int8_t)(x1 + 16.5f));
        y[ib].qs[lane] = (xi0 & 0x0f) | ((xi1 & 0x0f) << 4);
    }
    const uint32_t qh0 = __ballot_sync(mask, lane < QK5_0 / 2 && (xi0 & 0x10u)) & 0xffffu;
    const uint32_t qh1 = (__ballot_sync(mask, lane < QK5_0 / 2 && (xi1 & 0x10u)) & 0xffffu) << 16;
    if (lane == 0) {
        y[ib].d = gguf_cuda_compute_fp32_to_fp16(d);
        gguf_cuda_store_u32_le(y[ib].qh, qh0 | qh1);
    }
}

static __global__ void quantize_block_q5_1_warp(const float * __restrict__ x, block_q5_1 * __restrict__ y, int64_t n_blocks) {
    constexpr unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t ib = (int64_t)blockIdx.x * warps_per_block + warp;
    if (ib >= n_blocks) {
        return;
    }

    const float * xb = x + ib * QK5_1;
    const float min = __shfl_sync(mask, gguf_cuda_warp_reduce_min(xb[lane]), 0);
    const float max = __shfl_sync(mask, gguf_cuda_warp_reduce_max(xb[lane]), 0);
    const float d = __fdiv_rn(max - min, 31.0f);
    const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
    uint8_t xi0 = 0;
    uint8_t xi1 = 0;
    if (lane < QK5_1 / 2) {
        const float x0 = __fmul_rn(xb[lane] - min, id);
        const float x1 = __fmul_rn(xb[QK5_1 / 2 + lane] - min, id);
        xi0 = (uint8_t)(x0 + 0.5f);
        xi1 = (uint8_t)(x1 + 0.5f);
        y[ib].qs[lane] = (xi0 & 0x0f) | ((xi1 & 0x0f) << 4);
    }
    const uint32_t qh0 = __ballot_sync(mask, lane < QK5_1 / 2 && (xi0 & 0x10u)) & 0xffffu;
    const uint32_t qh1 = (__ballot_sync(mask, lane < QK5_1 / 2 && (xi1 & 0x10u)) & 0xffffu) << 16;
    if (lane == 0) {
        y[ib].d = gguf_cuda_compute_fp32_to_fp16(d);
        y[ib].m = gguf_cuda_compute_fp32_to_fp16(min);
        gguf_cuda_store_u32_le(y[ib].qh, qh0 | qh1);
    }
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

static __global__ void quantize_block_q2_K(const float * __restrict__ x, block_q2_K * __restrict__ y, int64_t n_blocks) {
    const int64_t iblock = blockDim.x * blockIdx.x + threadIdx.x;
    if (iblock >= n_blocks) {
        return;
    }

    const float * xb = x + iblock * QK_K;
    block_q2_K * yb = y + iblock;
    uint8_t l[QK_K];
    uint8_t laux[16];
    float weights[16];
    float mins[QK_K / 16];
    float scales[QK_K / 16];
    uint8_t quant_scales[QK_K / 16];
    uint8_t quant_mins[QK_K / 16];

    float max_scale = 0.0f;
    float max_min = 0.0f;
    for (int j = 0; j < QK_K / 16; ++j) {
        for (int ii = 0; ii < 16; ++ii) {
            weights[ii] = fabsf(xb[16 * j + ii]);
        }
        scales[j] = gguf_cuda_make_qkx2_quants(16, 3, xb + 16 * j, weights, l + 16 * j, &mins[j], laux, -0.5f, 0.1f, 15, true);
        if (scales[j] > max_scale) {
            max_scale = scales[j];
        }
        if (mins[j] > max_min) {
            max_min = mins[j];
        }
    }

    if (max_scale > 0.0f) {
        const float iscale = __fdiv_rn(15.0f, max_scale);
        for (int j = 0; j < QK_K / 16; ++j) {
            const uint8_t qscale = gguf_cuda_nearest_int(__fmul_rn(iscale, scales[j]));
            quant_scales[j] = qscale;
            yb->scales[j] = qscale;
        }
        yb->d = gguf_cuda_compute_fp32_to_fp16(__fdiv_rn(max_scale, 15.0f));
    } else {
        for (int j = 0; j < QK_K / 16; ++j) {
            quant_scales[j] = 0;
            yb->scales[j] = 0;
        }
        yb->d = gguf_cuda_compute_fp32_to_fp16(0.0f);
    }

    if (max_min > 0.0f) {
        const float iscale = __fdiv_rn(15.0f, max_min);
        for (int j = 0; j < QK_K / 16; ++j) {
            const uint8_t qmin = gguf_cuda_nearest_int(__fmul_rn(iscale, mins[j]));
            quant_mins[j] = qmin;
            yb->scales[j] |= qmin << 4;
        }
        yb->dmin = gguf_cuda_compute_fp32_to_fp16(__fdiv_rn(max_min, 15.0f));
    } else {
        for (int j = 0; j < QK_K / 16; ++j) {
            quant_mins[j] = 0;
        }
        yb->dmin = gguf_cuda_compute_fp32_to_fp16(0.0f);
    }

    const float d_base = __half2float(gguf_cuda_load_half(yb->d));
    const float dm_base = __half2float(gguf_cuda_load_half(yb->dmin));
    for (int j = 0; j < QK_K / 16; ++j) {
        const float d = __fmul_rn(d_base, float(quant_scales[j]));
        if (d == 0.0f) {
            continue;
        }
        const float dm = __fmul_rn(dm_base, float(quant_mins[j]));
        for (int ii = 0; ii < 16; ++ii) {
            int q = gguf_cuda_nearest_int(__fdiv_rn(xb[16 * j + ii] + dm, d));
            q = gguf_cuda_clamp_int(q, 0, 3);
            l[16 * j + ii] = q;
        }
    }

    for (int j = 0; j < QK_K; j += 128) {
        for (int ii = 0; ii < 32; ++ii) {
            yb->qs[j / 4 + ii] = l[j + ii] | (l[j + ii + 32] << 2) |
                                 (l[j + ii + 64] << 4) | (l[j + ii + 96] << 6);
        }
    }
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

static __global__ void quantize_block_q3_K_warp(const float * __restrict__ x, block_q3_K * __restrict__ y, int64_t n_blocks) {
    constexpr unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t iblock = (int64_t)blockIdx.x * warps_per_block + warp;
    if (iblock >= n_blocks) {
        return;
    }

    extern __shared__ int8_t q3_smem[];
    int8_t * l = q3_smem + warp * QK_K;
    float * scales = (float *)(q3_smem + warps_per_block * QK_K) + warp * (QK_K / 16);
    const float * xb = x + iblock * QK_K;
    block_q3_K * yb = y + iblock;

    float scale = 0.0f;
    if (lane < QK_K / 16) {
        scale = gguf_cuda_make_q3_quants(16, 4, xb + 16 * lane, l + 16 * lane);
        scales[lane] = scale;
    }

    const float max_scale = gguf_cuda_warp_reduce_absmax_first(scale, lane);
    __syncwarp(mask);
    ggml_half d_h = gguf_cuda_compute_fp32_to_fp16(0.0f);
    if (max_scale != 0.0f) {
        const float iscale = -32.0f / max_scale;
        d_h = gguf_cuda_compute_fp32_to_fp16(1.0f / iscale);
        if (lane == 0) {
            memset(yb->scales, 0, sizeof(yb->scales));
            for (int j = 0; j < QK_K / 16; ++j) {
                int q = gguf_cuda_nearest_int(iscale * scales[j]);
                q = gguf_cuda_clamp_int(q, -32, 31) + 32;
                if (j < 8) {
                    yb->scales[j] = q & 0x0f;
                } else {
                    yb->scales[j - 8] |= (q & 0x0f) << 4;
                }
                yb->scales[j % 4 + 8] |= (q >> 4) << (2 * (j / 4));
            }
        }
        if (lane < QK_K / 16) {
            int q = gguf_cuda_nearest_int(iscale * scale);
            q = gguf_cuda_clamp_int(q, -32, 31) + 32;
            const int8_t qscale = q - 32;

            const float d = __half2float(gguf_cuda_load_half(d_h)) * qscale;
            if (d != 0.0f) {
                for (int ii = 0; ii < 16; ++ii) {
                    int qv = gguf_cuda_nearest_int(xb[16 * lane + ii] / d);
                    qv = gguf_cuda_clamp_int(qv, -4, 3);
                    l[16 * lane + ii] = qv + 4;
                }
            }
        }
    } else if (lane == 0) {
        memset(yb->scales, 0, sizeof(yb->scales));
    }
    if (lane == 0) {
        yb->d = d_h;
    }

    __syncwarp(mask);

    if (lane < QK_K / 8) {
        uint8_t hm = 0;
        for (int bit = 0; bit < 8; ++bit) {
            hm |= (l[lane + bit * (QK_K / 8)] > 3) << bit;
        }
        yb->hmask[lane] = hm;
    }

    for (int j = 0; j < QK_K; j += 128) {
        yb->qs[j / 4 + lane] = (l[j + lane] & 3) | ((l[j + lane + 32] & 3) << 2) |
                                ((l[j + lane + 64] & 3) << 4) | ((l[j + lane + 96] & 3) << 6);
    }
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

static __global__ void quantize_block_q4_K(const float * __restrict__ x, block_q4_K * __restrict__ y, int64_t n_blocks) {
    constexpr int blocks_per_warp = 4;
    const int warps_per_cta = blockDim.x >> 5;
    const int blocks_per_cta = warps_per_cta * blocks_per_warp;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int block_in_warp = lane >> 3;
    const int subgroup = lane & 7;
    const int block_in_cta = warp * blocks_per_warp + block_in_warp;
    const int64_t iblock = (int64_t)blockIdx.x * blocks_per_cta + block_in_cta;
    if (iblock >= n_blocks) {
        return;
    }

    extern __shared__ uint8_t q4_lane_smem[];
    uint8_t * l_base = q4_lane_smem;
    float * mins_base = (float *)(l_base + blocks_per_cta * QK_K);
    float * scales_base = mins_base + blocks_per_cta * (QK_K / 32);
    uint8_t * l = l_base + block_in_cta * QK_K;
    float * mins = mins_base + block_in_cta * (QK_K / 32);
    float * scales = scales_base + block_in_cta * (QK_K / 32);
    const float * xb = x + iblock * QK_K;
    block_q4_K * yb = y + iblock;

    uint8_t laux[32];
    float weights[32];
    const int base = 32 * subgroup;
    float sum_x2 = 0.0f;
    for (int ii = 0; ii < 32; ++ii) {
        const float v = xb[base + ii];
        sum_x2 += v * v;
    }
    const float av_x = sqrtf(sum_x2 / 32.0f);
    for (int ii = 0; ii < 32; ++ii) {
        weights[ii] = av_x + fabsf(xb[base + ii]);
    }
    scales[subgroup] = gguf_cuda_make_qkx2_quants(
        32, 15, xb + base, weights, l + base, &mins[subgroup], laux, -1.0f, 0.1f, 20, false);

    __syncthreads();

    if (subgroup == 0) {
        float max_scale = 0.0f;
        float max_min = 0.0f;
        for (int j = 0; j < QK_K / 32; ++j) {
            max_scale = fmaxf(max_scale, scales[j]);
            max_min = fmaxf(max_min, mins[j]);
        }
        const float inv_scale = max_scale > 0.0f ? 63.0f / max_scale : 0.0f;
        const float inv_min = max_min > 0.0f ? 63.0f / max_min : 0.0f;
        for (int j = 0; j < QK_K / 32; ++j) {
            uint8_t ls = min(63, gguf_cuda_nearest_int(inv_scale * scales[j]));
            uint8_t lm = min(63, gguf_cuda_nearest_int(inv_min * mins[j]));
            if (j < 4) {
                yb->scales[j] = ls;
                yb->scales[j + 4] = lm;
            } else {
                yb->scales[j + 4] = (ls & 0x0f) | ((lm & 0x0f) << 4);
                yb->scales[j - 4] |= (ls >> 4) << 6;
                yb->scales[j] |= (lm >> 4) << 6;
            }
        }
        yb->d = gguf_cuda_compute_fp32_to_fp16(max_scale / 63.0f);
        yb->dmin = gguf_cuda_compute_fp32_to_fp16(max_min / 63.0f);
    }

    __syncthreads();

    const float d_base = __half2float(gguf_cuda_load_half(yb->d));
    const float dm_base = __half2float(gguf_cuda_load_half(yb->dmin));
    uint8_t sc;
    uint8_t m;
    gguf_cuda_get_scale_min_k4(subgroup, yb->scales, &sc, &m);
    const float d = d_base * sc;
    if (d != 0.0f) {
        const float dm = dm_base * m;
        for (int ii = 0; ii < 32; ++ii) {
            int q = gguf_cuda_nearest_int((xb[base + ii] + dm) / d);
            q = gguf_cuda_clamp_int(q, 0, 15);
            l[base + ii] = q;
        }
    }

    __syncthreads();

    const int pack_pair = subgroup >> 1;
    if ((subgroup & 1) == 0) {
        const int l_pack_base = 64 * pack_pair;
        for (int ii = 0; ii < 32; ++ii) {
            yb->qs[32 * pack_pair + ii] = l[l_pack_base + ii] | (l[l_pack_base + 32 + ii] << 4);
        }
    }
}

static __global__ void quantize_block_q5_K(const float * __restrict__ x, block_q5_K * __restrict__ y, int64_t n_blocks) {
    constexpr int blocks_per_warp = 4;
    const int warps_per_cta = blockDim.x >> 5;
    const int blocks_per_cta = warps_per_cta * blocks_per_warp;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int block_in_warp = lane >> 3;
    const int subgroup = lane & 7;
    const int block_in_cta = warp * blocks_per_warp + block_in_warp;
    const int64_t iblock = (int64_t)blockIdx.x * blocks_per_cta + block_in_cta;
    if (iblock >= n_blocks) {
        return;
    }

    extern __shared__ uint8_t q5_lane_smem[];
    uint8_t * l_base = q5_lane_smem;
    float * mins_base = (float *)(l_base + blocks_per_cta * QK_K);
    float * scales_base = mins_base + blocks_per_cta * (QK_K / 32);
    uint8_t * l = l_base + block_in_cta * QK_K;
    float * mins = mins_base + block_in_cta * (QK_K / 32);
    float * scales = scales_base + block_in_cta * (QK_K / 32);
    const float * xb = x + iblock * QK_K;
    block_q5_K * yb = y + iblock;

    uint8_t laux[32];
    float weights[32];

    const int base = 32 * subgroup;
    float sum_x2 = 0.0f;
    for (int ii = 0; ii < 32; ++ii) {
        const float v = xb[base + ii];
        sum_x2 += v * v;
    }
    const float av_x = sqrtf(sum_x2 / 32.0f);
    for (int ii = 0; ii < 32; ++ii) {
        weights[ii] = av_x + fabsf(xb[base + ii]);
    }
    scales[subgroup] = gguf_cuda_make_qkx2_quants(
        32, 31, xb + base, weights, l + base, &mins[subgroup], laux, -0.5f, 0.1f, 15, false);

    __syncthreads();

    if (subgroup == 0) {
        float max_scale = 0.0f;
        float max_min = 0.0f;
        for (int j = 0; j < QK_K / 32; ++j) {
            max_scale = fmaxf(max_scale, scales[j]);
            max_min = fmaxf(max_min, mins[j]);
        }

        const float inv_scale = max_scale > 0.0f ? 63.0f / max_scale : 0.0f;
        const float inv_min = max_min > 0.0f ? 63.0f / max_min : 0.0f;
        for (int j = 0; j < QK_K / 32; ++j) {
            uint8_t ls = min(63, gguf_cuda_nearest_int(inv_scale * scales[j]));
            uint8_t lm = min(63, gguf_cuda_nearest_int(inv_min * mins[j]));
            if (j < 4) {
                yb->scales[j] = ls;
                yb->scales[j + 4] = lm;
            } else {
                yb->scales[j + 4] = (ls & 0x0f) | ((lm & 0x0f) << 4);
                yb->scales[j - 4] |= (ls >> 4) << 6;
                yb->scales[j] |= (lm >> 4) << 6;
            }
        }
        yb->d = gguf_cuda_compute_fp32_to_fp16(max_scale / 63.0f);
        yb->dmin = gguf_cuda_compute_fp32_to_fp16(max_min / 63.0f);
    }

    __syncthreads();

    const float d_base = __half2float(gguf_cuda_load_half(yb->d));
    const float dm_base = __half2float(gguf_cuda_load_half(yb->dmin));
    uint8_t sc;
    uint8_t m;
    gguf_cuda_get_scale_min_k4(subgroup, yb->scales, &sc, &m);
    const float d = d_base * sc;
    if (d != 0.0f) {
        const float dm = dm_base * m;
        for (int ii = 0; ii < 32; ++ii) {
            int q = gguf_cuda_nearest_int((xb[base + ii] + dm) / d);
            q = gguf_cuda_clamp_int(q, 0, 31);
            l[base + ii] = q;
        }
    }

    __syncthreads();

    for (int j = subgroup; j < 32; j += 8) {
        uint8_t qh = 0;
        for (int pack_pair = 0; pack_pair < 4; ++pack_pair) {
            const int n = 64 * pack_pair;
            int q1 = l[n + j];
            if (q1 > 15) {
                q1 -= 16;
                qh |= 1 << (2 * pack_pair);
            }
            int q2 = l[n + j + 32];
            if (q2 > 15) {
                q2 -= 16;
                qh |= 1 << (2 * pack_pair + 1);
            }
            yb->qs[32 * pack_pair + j] = q1 | (q2 << 4);
        }
        yb->qh[j] = qh;
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

static __global__ void quantize_block_iq4_nl(const float * __restrict__ x, block_iq4_nl * __restrict__ y, int64_t n_blocks) {
    const int64_t iblock = blockDim.x * blockIdx.x + threadIdx.x;
    if (iblock >= n_blocks) {
        return;
    }

    float scale;
    float weight[QK4_NL];
    uint8_t l[QK4_NL];
    uint16_t scales_h_unused;
    gguf_cuda_quantize_iq4_nl_impl(
        QK4_NL, 32, x + iblock * QK4_NL, &y[iblock].d, y[iblock].qs, &scales_h_unused, nullptr,
        &scale, weight, l, 7);
}

static __global__ void quantize_block_iq4_xs(const float * __restrict__ x, block_iq4_xs * __restrict__ y, int64_t n_blocks) {
    const int64_t iblock = blockDim.x * blockIdx.x + threadIdx.x;
    if (iblock >= n_blocks) {
        return;
    }

    float weight[32];
    float scales[QK_K / 32];
    uint8_t l[QK_K];
    gguf_cuda_quantize_iq4_nl_impl(
        QK_K, 32, x + iblock * QK_K, &y[iblock].d, y[iblock].qs, &y[iblock].scales_h, y[iblock].scales_l,
        scales, weight, l, 7);
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

static __global__ void quantize_block_iq2_xxs(
    const float * __restrict__ x,
    const float * __restrict__ quant_weights,
    block_iq2_xxs * __restrict__ y,
    int64_t n_blocks,
    int64_t n_blocks_per_row
) {
    const int64_t iblock = blockDim.x * blockIdx.x + threadIdx.x;
    if (iblock >= n_blocks) {
        return;
    }

    const float * xb = x + iblock * QK_K;
    const int64_t row_block = iblock % n_blocks_per_row;
    const float * qw_block = quant_weights + row_block * QK_K;
    block_iq2_xxs * yb = y + iblock;
    float scales[QK_K / 32];
    float weight[32];
    float xval[32];
    uint8_t l_u8[32];
    int8_t l[32];
    int8_t laux[32];
    float waux[32];
    uint8_t block_signs[4];
    uint32_t q2[2 * (QK_K / 32)];

    yb->d = gguf_cuda_compute_fp32_to_fp16(0.0f);
    for (int i = 0; i < 2 * (QK_K / 32); ++i) {
        q2[i] = 0;
    }

    float max_scale = 0.0f;
    float sumx2 = 0.0f;
    for (int i = 0; i < QK_K; ++i) {
        sumx2 += xb[i] * xb[i];
    }
    const float sigma2 = sumx2 / QK_K;

    for (int ib = 0; ib < QK_K / 32; ++ib) {
        const float * x32 = xb + 32 * ib;
        for (int i = 0; i < 32; ++i) {
            weight[i] = qw_block[32 * ib + i] * sqrtf(sigma2 + x32[i] * x32[i]);
            waux[i] = sqrtf(weight[i]);
        }
        for (int k = 0; k < 4; ++k) {
            int nflip = 0;
            uint8_t s = 0;
            for (int i = 0; i < 8; ++i) {
                const float v = x32[8 * k + i];
                if (v >= 0.0f) {
                    xval[8 * k + i] = v;
                } else {
                    xval[8 * k + i] = -v;
                    ++nflip;
                    s |= 1 << i;
                }
            }
            if (nflip % 2) {
                int imin = 0;
                float minv = weight[8 * k] * x32[8 * k] * x32[8 * k];
                for (int i = 1; i < 8; ++i) {
                    const float ax = weight[8 * k + i] * x32[8 * k + i] * x32[8 * k + i];
                    if (ax < minv) {
                        minv = ax;
                        imin = i;
                    }
                }
                xval[8 * k + imin] = -xval[8 * k + imin];
                s ^= 1 << imin;
            }
            block_signs[k] = s & 127;
        }

        float maxv = xval[0];
        for (int i = 1; i < 32; ++i) {
            maxv = fmaxf(maxv, xval[i]);
        }
        if (maxv < GROUP_MAX_EPS) {
            scales[ib] = 0.0f;
            for (int i = 0; i < 32; ++i) {
                l[i] = 0;
            }
            continue;
        }
        float scale = gguf_cuda_make_qp_quants(32, 4, xval, l_u8, weight);
        for (int i = 0; i < 32; ++i) {
            l[i] = l_u8[i];
        }
        const float eff_max = scale * 3.0f;
        if (eff_max <= 0.0f) {
            scales[ib] = 0.0f;
            for (int i = 0; i < 32; ++i) {
                l[i] = 0;
            }
            continue;
        }

        float best = 0.0f;
        for (int is = -6; is <= 6; ++is) {
            const float id = (5.0f + is * 0.1f) / eff_max;
            const float this_scale = 1.0f / id;
            for (int k = 0; k < 4; ++k) {
                for (int i = 0; i < 8; ++i) {
                    int q = gguf_cuda_nearest_int(0.5f * (id * xval[8 * k + i] - 1.0f));
                    laux[8 * k + i] = gguf_cuda_clamp_int(q, 0, 2);
                }
                int grid_index = gguf_cuda_iq2_find_grid_index(iq2xxs_grid, 256, laux + 8 * k);
                if (grid_index < 0) {
                    grid_index = gguf_cuda_iq2_find_best_neighbour(
                        iq2xxs_grid, 256, 2, xval + 8 * k, waux + 8 * k, this_scale, laux + 8 * k);
                }
            }
            float sumqx = 0.0f;
            float sumq2 = 0.0f;
            for (int i = 0; i < 32; ++i) {
                const float w = weight[i];
                const float q = 2.0f * laux[i] + 1.0f;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0.0f && sumqx * sumqx > best * sumq2) {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                for (int i = 0; i < 32; ++i) {
                    l[i] = laux[i];
                }
            }
        }

        if (scale > 0.0f) {
            const float id = 1.0f / scale;
            for (int k = 0; k < 4; ++k) {
                for (int i = 0; i < 8; ++i) {
                    int q = gguf_cuda_nearest_int(0.5f * (id * xval[8 * k + i] - 1.0f));
                    l[8 * k + i] = gguf_cuda_clamp_int(q, 0, 2);
                }
                int grid_index = gguf_cuda_iq2_find_grid_index(iq2xxs_grid, 256, l + 8 * k);
                if (grid_index < 0) {
                    grid_index = gguf_cuda_iq2_find_best_neighbour(
                        iq2xxs_grid, 256, 2, xval + 8 * k, waux + 8 * k, scale, l + 8 * k);
                } else {
                    for (int i = 0; i < 8; ++i) {
                        l[8 * k + i] = gguf_cuda_iq2_grid_l(iq2xxs_grid, grid_index, i);
                    }
                }
            }
            float sumqx = 0.0f;
            float sumq2 = 0.0f;
            for (int i = 0; i < 32; ++i) {
                const float w = weight[i];
                const float q = 2.0f * l[i] + 1.0f;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0.0f) {
                scale = sumqx / sumq2;
            }
        }
        if (scale < 0.0f) {
            scale = -scale;
            for (int k = 0; k < 4; ++k) {
                block_signs[k] = (~block_signs[k]) & 127;
            }
        }
        for (int k = 0; k < 4; ++k) {
            int grid_index = gguf_cuda_iq2_find_grid_index(iq2xxs_grid, 256, l + 8 * k);
            q2[2 * ib + 0] |= ((uint32_t)grid_index << (8 * k));
            q2[2 * ib + 1] |= ((uint32_t)block_signs[k] << (7 * k));
        }
        scales[ib] = scale;
        max_scale = fmaxf(max_scale, scale);
    }

    if (max_scale == 0.0f) {
        for (int i = 0; i < QK_K / 8; ++i) {
            yb->qs[i] = 0;
        }
        return;
    }

    const float d = max_scale / 31.0f;
    yb->d = gguf_cuda_compute_fp32_to_fp16(d);
    const float id = 1.0f / d;
    for (int ib = 0; ib < QK_K / 32; ++ib) {
        int q = gguf_cuda_nearest_int(0.5f * (id * scales[ib] - 1.0f));
        q = gguf_cuda_clamp_int(q, 0, 15);
        q2[2 * ib + 1] |= (uint32_t)q << 28;
    }
    uint8_t * out = (uint8_t *)yb->qs;
    for (int i = 0; i < 2 * (QK_K / 32); ++i) {
        gguf_cuda_store_u32_le(out + 4 * i, q2[i]);
    }
}

static __global__ void quantize_block_iq2_xs(
    const float * __restrict__ x,
    const float * __restrict__ quant_weights,
    block_iq2_xs * __restrict__ y,
    int64_t n_blocks,
    int64_t n_blocks_per_row
) {
    const int64_t iblock = blockDim.x * blockIdx.x + threadIdx.x;
    if (iblock >= n_blocks) {
        return;
    }

    const float * xb = x + iblock * QK_K;
    const int64_t row_block = iblock % n_blocks_per_row;
    const float * qw_block = quant_weights + row_block * QK_K;
    block_iq2_xs * yb = y + iblock;
    float scales[QK_K / 16];
    float weight[16];
    float xval[16];
    int8_t l[16];
    int8_t laux[16];
    float waux[16];
    bool is_on_grid[2];
    bool is_on_grid_aux[2];
    uint8_t block_signs[2];
    uint16_t q2[2 * (QK_K / 16)];

    yb->d = gguf_cuda_compute_fp32_to_fp16(0.0f);
    for (int i = 0; i < 2 * (QK_K / 16); ++i) {
        q2[i] = 0;
    }
    for (int i = 0; i < QK_K / 32; ++i) {
        yb->scales[i] = 0;
    }

    float max_scale = 0.0f;
    float sumx2 = 0.0f;
    for (int i = 0; i < QK_K; ++i) {
        sumx2 += xb[i] * xb[i];
    }
    const float sigma2 = sumx2 / QK_K;

    for (int ib = 0; ib < QK_K / 16; ++ib) {
        const float * x16 = xb + 16 * ib;
        for (int i = 0; i < 16; ++i) {
            weight[i] = qw_block[16 * ib + i] * sqrtf(sigma2 + x16[i] * x16[i]);
            waux[i] = sqrtf(weight[i]);
            l[i] = 0;
        }

        for (int k = 0; k < 2; ++k) {
            int nflip = 0;
            uint8_t s = 0;
            for (int i = 0; i < 8; ++i) {
                const float v = x16[8 * k + i];
                if (v >= 0.0f) {
                    xval[8 * k + i] = v;
                } else {
                    xval[8 * k + i] = -v;
                    ++nflip;
                    s |= 1 << i;
                }
            }
            if (nflip % 2) {
                int imin = 0;
                float minv = weight[8 * k] * x16[8 * k] * x16[8 * k];
                for (int i = 1; i < 8; ++i) {
                    const float ax = weight[8 * k + i] * x16[8 * k + i] * x16[8 * k + i];
                    if (ax < minv) {
                        minv = ax;
                        imin = i;
                    }
                }
                xval[8 * k + imin] = -xval[8 * k + imin];
                s ^= 1 << imin;
            }
            block_signs[k] = s & 127;
        }

        float maxv = xval[0];
        for (int i = 1; i < 16; ++i) {
            maxv = fmaxf(maxv, xval[i]);
        }
        if (maxv < GROUP_MAX_EPS) {
            scales[ib] = 0.0f;
            continue;
        }

        float best = 0.0f;
        float scale = maxv / 5.0f;
        is_on_grid[0] = true;
        is_on_grid[1] = true;
        for (int is = -9; is <= 9; ++is) {
            const float id = (5.0f + is * 0.1f) / maxv;
            const float this_scale = 1.0f / id;
            is_on_grid_aux[0] = true;
            is_on_grid_aux[1] = true;
            for (int k = 0; k < 2; ++k) {
                for (int i = 0; i < 8; ++i) {
                    int q = gguf_cuda_nearest_int(0.5f * (id * xval[8 * k + i] - 1.0f));
                    laux[8 * k + i] = gguf_cuda_clamp_int(q, 0, 2);
                }
                int grid_index = gguf_cuda_iq2_find_grid_index(iq2xs_grid, 512, laux + 8 * k);
                if (grid_index < 0) {
                    is_on_grid_aux[k] = false;
                    grid_index = gguf_cuda_iq2_find_best_neighbour(
                        iq2xs_grid, 512, 2, xval + 8 * k, waux + 8 * k, this_scale, laux + 8 * k);
                }
            }

            float sumqx = 0.0f;
            float sumq2 = 0.0f;
            for (int i = 0; i < 16; ++i) {
                const float w = weight[i];
                const float q = 2.0f * laux[i] + 1.0f;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0.0f && sumqx * sumqx > best * sumq2) {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                for (int i = 0; i < 16; ++i) {
                    l[i] = laux[i];
                }
                is_on_grid[0] = is_on_grid_aux[0];
                is_on_grid[1] = is_on_grid_aux[1];
            }
        }

        if ((!is_on_grid[0] || !is_on_grid[1]) && scale > 0.0f) {
            const float id = 1.0f / scale;
            for (int k = 0; k < 2; ++k) {
                if (is_on_grid[k]) {
                    continue;
                }
                for (int i = 0; i < 8; ++i) {
                    int q = gguf_cuda_nearest_int(0.5f * (id * xval[8 * k + i] - 1.0f));
                    l[8 * k + i] = gguf_cuda_clamp_int(q, 0, 2);
                }
                int grid_index = gguf_cuda_iq2_find_grid_index(iq2xs_grid, 512, l + 8 * k);
                if (grid_index < 0) {
                    grid_index = gguf_cuda_iq2_find_best_neighbour(
                        iq2xs_grid, 512, 2, xval + 8 * k, waux + 8 * k, scale, l + 8 * k);
                } else {
                    for (int i = 0; i < 8; ++i) {
                        l[8 * k + i] = gguf_cuda_iq2_grid_l(iq2xs_grid, grid_index, i);
                    }
                }
            }
            float sumqx = 0.0f;
            float sumq2 = 0.0f;
            for (int i = 0; i < 16; ++i) {
                const float w = weight[i];
                const float q = 2.0f * l[i] + 1.0f;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0.0f) {
                scale = sumqx / sumq2;
            }
        }

        if (scale < 0.0f) {
            scale = -scale;
            for (int k = 0; k < 2; ++k) {
                block_signs[k] = (~block_signs[k]) & 127;
            }
        }
        for (int k = 0; k < 2; ++k) {
            int grid_index = gguf_cuda_iq2_find_grid_index(iq2xs_grid, 512, l + 8 * k);
            q2[2 * ib + k] = (uint16_t)(grid_index | (block_signs[k] << 9));
        }
        scales[ib] = scale;
        max_scale = fmaxf(max_scale, scale);
    }

    if (max_scale == 0.0f) {
        for (int i = 0; i < QK_K / 8; ++i) {
            yb->qs[i] = 0;
        }
        return;
    }

    const float d = max_scale / 31.0f;
    yb->d = gguf_cuda_compute_fp32_to_fp16(d);
    const float id = 1.0f / d;
    for (int ib = 0; ib < QK_K / 16; ++ib) {
        int q = gguf_cuda_nearest_int(0.5f * (id * scales[ib] - 1.0f));
        q = gguf_cuda_clamp_int(q, 0, 15);
        if (ib % 2 == 0) {
            yb->scales[ib / 2] = q;
        } else {
            yb->scales[ib / 2] |= q << 4;
        }
    }
    for (int i = 0; i < 2 * (QK_K / 16); ++i) {
        yb->qs[i] = q2[i];
    }
}

static __global__ void quantize_block_iq2_s(
    const float * __restrict__ x,
    const float * __restrict__ quant_weights,
    block_iq2_s * __restrict__ y,
    int64_t n_blocks,
    int64_t n_blocks_per_row
) {
    const int64_t iblock = blockDim.x * blockIdx.x + threadIdx.x;
    if (iblock >= n_blocks) {
        return;
    }

    const float * xb = x + iblock * QK_K;
    const int64_t row_block = iblock % n_blocks_per_row;
    const float * qw_block = quant_weights ? quant_weights + row_block * QK_K : nullptr;
    block_iq2_s * yb = y + iblock;
    float scales[QK_K / 16];
    float weight[16];
    float xval[16];
    int8_t l[16];
    int8_t laux[16];
    float waux[16];
    bool is_on_grid[2];
    bool is_on_grid_aux[2];
    uint8_t block_signs[2];

    memset(yb, 0, sizeof(block_iq2_s));
    yb->d = gguf_cuda_compute_fp32_to_fp16(0.0f);

    float max_scale = 0.0f;
    float sumx2 = 0.0f;
    for (int i = 0; i < QK_K; ++i) {
        sumx2 += xb[i] * xb[i];
    }
    const float sigma2 = 2.0f * sumx2 / QK_K;

    for (int ib = 0; ib < QK_K / 16; ++ib) {
        const float * x16 = xb + 16 * ib;
        for (int i = 0; i < 16; ++i) {
            weight[i] = qw_block ? qw_block[16 * ib + i] * sqrtf(sigma2 + x16[i] * x16[i])
                                 : 0.25f * sigma2 + x16[i] * x16[i];
            waux[i] = sqrtf(weight[i]);
            l[i] = 0;
        }

        for (int k = 0; k < 2; ++k) {
            uint8_t s = 0;
            for (int i = 0; i < 8; ++i) {
                const float v = x16[8 * k + i];
                if (v >= 0.0f) {
                    xval[8 * k + i] = v;
                } else {
                    xval[8 * k + i] = -v;
                    s |= 1 << i;
                }
            }
            block_signs[k] = s;
        }

        float maxv = xval[0];
        for (int i = 1; i < 16; ++i) {
            maxv = fmaxf(maxv, xval[i]);
        }
        if (maxv < GROUP_MAX_EPS_IQ2_S) {
            scales[ib] = 0.0f;
            continue;
        }

        float best = 0.0f;
        float scale = maxv / 5.0f;
        is_on_grid[0] = true;
        is_on_grid[1] = true;
        for (int is = -9; is <= 9; ++is) {
            const float id = (5.0f + is * 0.1f) / maxv;
            const float this_scale = 1.0f / id;
            is_on_grid_aux[0] = true;
            is_on_grid_aux[1] = true;
            for (int k = 0; k < 2; ++k) {
                for (int i = 0; i < 8; ++i) {
                    int q = gguf_cuda_nearest_int(0.5f * (id * xval[8 * k + i] - 1.0f));
                    laux[8 * k + i] = gguf_cuda_clamp_int(q, 0, 2);
                }
                int grid_index = gguf_cuda_iq2_find_grid_index(iq2s_grid, 1024, laux + 8 * k);
                if (grid_index < 0) {
                    is_on_grid_aux[k] = false;
                    grid_index = gguf_cuda_iq2_find_best_neighbour(
                        iq2s_grid, 1024, 1, xval + 8 * k, waux + 8 * k, this_scale, laux + 8 * k);
                }
            }

            float sumqx = 0.0f;
            float sumq2 = 0.0f;
            for (int i = 0; i < 16; ++i) {
                const float w = weight[i];
                const float q = 2.0f * laux[i] + 1.0f;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0.0f && sumqx * sumqx > best * sumq2) {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                for (int i = 0; i < 16; ++i) {
                    l[i] = laux[i];
                }
                is_on_grid[0] = is_on_grid_aux[0];
                is_on_grid[1] = is_on_grid_aux[1];
            }
        }

        if ((!is_on_grid[0] || !is_on_grid[1]) && scale > 0.0f) {
            const float id = 1.0f / scale;
            for (int k = 0; k < 2; ++k) {
                if (is_on_grid[k]) {
                    continue;
                }
                for (int i = 0; i < 8; ++i) {
                    int q = gguf_cuda_nearest_int(0.5f * (id * xval[8 * k + i] - 1.0f));
                    l[8 * k + i] = gguf_cuda_clamp_int(q, 0, 2);
                }
                int grid_index = gguf_cuda_iq2_find_grid_index(iq2s_grid, 1024, l + 8 * k);
                if (grid_index < 0) {
                    grid_index = gguf_cuda_iq2_find_best_neighbour(
                        iq2s_grid, 1024, 1, xval + 8 * k, waux + 8 * k, scale, l + 8 * k);
                } else {
                    for (int i = 0; i < 8; ++i) {
                        l[8 * k + i] = gguf_cuda_iq2_grid_l(iq2s_grid, grid_index, i);
                    }
                }
            }
            float sumqx = 0.0f;
            float sumq2 = 0.0f;
            for (int i = 0; i < 16; ++i) {
                const float w = weight[i];
                const float q = 2.0f * l[i] + 1.0f;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0.0f) {
                scale = sumqx / sumq2;
            }
        }

        if (scale < 0.0f) {
            scale = -scale;
            for (int k = 0; k < 2; ++k) {
                block_signs[k] = ~block_signs[k];
            }
        }
        for (int k = 0; k < 2; ++k) {
            int grid_index = gguf_cuda_iq2_find_grid_index(iq2s_grid, 1024, l + 8 * k);
            const int i8 = 2 * ib + k;
            yb->qs[i8] = grid_index & 255;
            yb->qh[i8 / 4] |= ((grid_index >> 8) << (2 * (i8 % 4)));
            yb->qs[QK_K / 8 + i8] = block_signs[k];
        }
        scales[ib] = scale;
        max_scale = fmaxf(max_scale, scale);
    }

    if (max_scale == 0.0f) {
        return;
    }

    const float d = max_scale / 31.0f;
    yb->d = gguf_cuda_compute_fp32_to_fp16(d * 0.9875f);
    const float id = 1.0f / d;
    for (int ib = 0; ib < QK_K / 16; ++ib) {
        int q = gguf_cuda_nearest_int(0.5f * (id * scales[ib] - 1.0f));
        q = gguf_cuda_clamp_int(q, 0, 15);
        if (ib % 2 == 0) {
            yb->scales[ib / 2] = q;
        } else {
            yb->scales[ib / 2] |= q << 4;
        }
    }
}

static __global__ void quantize_block_iq3_xxs(const float * __restrict__ x, block_iq3_xxs * __restrict__ y, int64_t n_blocks) {
    const int64_t iblock = blockDim.x * blockIdx.x + threadIdx.x;
    if (iblock >= n_blocks) {
        return;
    }

    const float * xb = x + iblock * QK_K;
    block_iq3_xxs * yb = y + iblock;
    float scales[QK_K / 32];
    float weight[32];
    float xval[32];
    int8_t l[32];
    int8_t laux[32];
    float waux[32];
    bool is_on_grid[8];
    bool is_on_grid_aux[8];
    uint8_t block_signs[4];
    uint8_t q3[3 * (QK_K / 8)];
    uint32_t * scales_and_signs = (uint32_t *)(q3 + QK_K / 4);

    yb->d = gguf_cuda_compute_fp32_to_fp16(0.0f);
    for (int i = 0; i < 3 * (QK_K / 8); ++i) {
        q3[i] = 0;
    }

    float max_scale = 0.0f;
    for (int ib = 0; ib < QK_K / 32; ++ib) {
        const float * x32 = xb + 32 * ib;
        for (int i = 0; i < 32; ++i) {
            weight[i] = x32[i] * x32[i];
            waux[i] = sqrtf(weight[i]);
            l[i] = 0;
        }

        for (int k = 0; k < 4; ++k) {
            int nflip = 0;
            uint8_t s = 0;
            for (int i = 0; i < 8; ++i) {
                const float v = x32[8 * k + i];
                if (v >= 0.0f) {
                    xval[8 * k + i] = v;
                } else {
                    xval[8 * k + i] = -v;
                    ++nflip;
                    s |= 1 << i;
                }
            }
            if (nflip % 2) {
                int imin = 0;
                float minv = weight[8 * k] * x32[8 * k] * x32[8 * k];
                for (int i = 1; i < 8; ++i) {
                    const float ax = weight[8 * k + i] * x32[8 * k + i] * x32[8 * k + i];
                    if (ax < minv) {
                        minv = ax;
                        imin = i;
                    }
                }
                xval[8 * k + imin] = -xval[8 * k + imin];
                s ^= 1 << imin;
            }
            block_signs[k] = s & 127;
        }

        float maxv = xval[0];
        for (int i = 1; i < 32; ++i) {
            maxv = fmaxf(maxv, xval[i]);
        }
        if (maxv < GROUP_MAX_EPS_IQ3_XXS) {
            scales[ib] = 0.0f;
            continue;
        }

        float best = 0.0f;
        float scale = maxv / 15.0f;
        for (int k = 0; k < 8; ++k) {
            is_on_grid[k] = true;
        }

        for (int is = -15; is <= 15; ++is) {
            const float id = (15.0f + is * 0.2f) / maxv;
            const float this_scale = 1.0f / id;
            for (int k = 0; k < 8; ++k) {
                is_on_grid_aux[k] = true;
                for (int i = 0; i < 4; ++i) {
                    int q = gguf_cuda_nearest_int(0.5f * (id * xval[4 * k + i] - 1.0f));
                    laux[4 * k + i] = gguf_cuda_clamp_int(q, 0, 7);
                }
                int grid_index = gguf_cuda_iq3_find_grid_index(iq3xxs_grid, 256, laux + 4 * k);
                if (grid_index < 0) {
                    is_on_grid_aux[k] = false;
                    grid_index = gguf_cuda_iq3_find_best_neighbour(
                        iq3xxs_grid, 256, 2, xval + 4 * k, waux + 4 * k, this_scale, laux + 4 * k);
                }
            }

            float sumqx = 0.0f;
            float sumq2 = 0.0f;
            for (int i = 0; i < 32; ++i) {
                const float w = weight[i];
                const float q = 2.0f * laux[i] + 1.0f;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0.0f && sumqx * sumqx > best * sumq2) {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                for (int i = 0; i < 32; ++i) {
                    l[i] = laux[i];
                }
                for (int k = 0; k < 8; ++k) {
                    is_on_grid[k] = is_on_grid_aux[k];
                }
            }
        }

        bool any_not_on_grid = false;
        for (int k = 0; k < 8; ++k) {
            any_not_on_grid = any_not_on_grid || !is_on_grid[k];
        }
        if (any_not_on_grid && scale > 0.0f) {
            const float id = 1.0f / scale;
            for (int k = 0; k < 8; ++k) {
                if (is_on_grid[k]) {
                    continue;
                }
                for (int i = 0; i < 4; ++i) {
                    int q = gguf_cuda_nearest_int(0.5f * (id * xval[4 * k + i] - 1.0f));
                    l[4 * k + i] = gguf_cuda_clamp_int(q, 0, 7);
                }
                int grid_index = gguf_cuda_iq3_find_grid_index(iq3xxs_grid, 256, l + 4 * k);
                if (grid_index < 0) {
                    grid_index = gguf_cuda_iq3_find_best_neighbour(
                        iq3xxs_grid, 256, 2, xval + 4 * k, waux + 4 * k, scale, l + 4 * k);
                } else {
                    for (int i = 0; i < 4; ++i) {
                        l[4 * k + i] = gguf_cuda_iq3_grid_l(iq3xxs_grid, grid_index, i);
                    }
                }
            }
            float sumqx = 0.0f;
            float sumq2 = 0.0f;
            for (int i = 0; i < 32; ++i) {
                const float w = weight[i];
                const float q = 2.0f * l[i] + 1.0f;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0.0f) {
                scale = sumqx / sumq2;
            }
        }

        if (scale < 0.0f) {
            scale = -scale;
            for (int k = 0; k < 4; ++k) {
                block_signs[k] = (~block_signs[k]) & 127;
            }
        }
        for (int k = 0; k < 8; ++k) {
            int grid_index = gguf_cuda_iq3_find_grid_index(iq3xxs_grid, 256, l + 4 * k);
            q3[8 * ib + k] = grid_index;
        }
        scales_and_signs[ib] = block_signs[0] | (block_signs[1] << 7) |
                               (block_signs[2] << 14) | (block_signs[3] << 21);
        scales[ib] = scale;
        max_scale = fmaxf(max_scale, scale);
    }

    if (max_scale == 0.0f) {
        memset(yb->qs, 0, sizeof(block_iq3_xxs::qs));
        return;
    }

    const float d = max_scale / 31.0f;
    yb->d = gguf_cuda_compute_fp32_to_fp16(d * 1.0125f);
    const float id = 1.0f / d;
    for (int ib = 0; ib < QK_K / 32; ++ib) {
        int q = gguf_cuda_nearest_int(0.5f * (id * scales[ib] - 1.0f));
        q = gguf_cuda_clamp_int(q, 0, 15);
        scales_and_signs[ib] |= (uint32_t)q << 28;
    }
    for (int i = 0; i < 3 * (QK_K / 8); ++i) {
        yb->qs[i] = q3[i];
    }
}

static __global__ void quantize_block_iq3_s(const float * __restrict__ x, block_iq3_s * __restrict__ y, int64_t n_blocks) {
    const int64_t iblock = blockDim.x * blockIdx.x + threadIdx.x;
    if (iblock >= n_blocks) {
        return;
    }

    const float * xb = x + iblock * QK_K;
    block_iq3_s * yb = y + iblock;
    float scales[QK_K / 32];
    float weight[32];
    float xval[32];
    int8_t l[32];
    int8_t laux[32];
    float waux[32];
    bool is_on_grid[8];
    bool is_on_grid_aux[8];
    uint8_t block_signs[4];

    memset(yb, 0, sizeof(block_iq3_s));
    yb->d = gguf_cuda_compute_fp32_to_fp16(0.0f);

    float max_scale = 0.0f;
    for (int ib = 0; ib < QK_K / 32; ++ib) {
        const float * x32 = xb + 32 * ib;
        for (int i = 0; i < 32; ++i) {
            weight[i] = x32[i] * x32[i];
            waux[i] = sqrtf(weight[i]);
            l[i] = 0;
        }

        for (int k = 0; k < 4; ++k) {
            uint8_t s = 0;
            for (int i = 0; i < 8; ++i) {
                const float v = x32[8 * k + i];
                if (v >= 0.0f) {
                    xval[8 * k + i] = v;
                } else {
                    xval[8 * k + i] = -v;
                    s |= 1 << i;
                }
            }
            block_signs[k] = s;
        }

        float maxv = xval[0];
        for (int i = 1; i < 32; ++i) {
            maxv = fmaxf(maxv, xval[i]);
        }
        if (maxv == 0.0f) {
            scales[ib] = 0.0f;
            continue;
        }

        float best = 0.0f;
        float scale = maxv / 15.0f;
        for (int k = 0; k < 8; ++k) {
            is_on_grid[k] = false;
        }

        for (int is = -9; is <= 9; ++is) {
            const float id = (15.0f + is * 0.2f) / maxv;
            const float this_scale = 1.0f / id;
            for (int k = 0; k < 8; ++k) {
                is_on_grid_aux[k] = true;
                for (int i = 0; i < 4; ++i) {
                    int q = gguf_cuda_nearest_int(0.5f * (id * xval[4 * k + i] - 1.0f));
                    laux[4 * k + i] = gguf_cuda_clamp_int(q, 0, 7);
                }
                int grid_index = gguf_cuda_iq3_find_grid_index(iq3xs_grid, 512, laux + 4 * k);
                if (grid_index < 0) {
                    is_on_grid_aux[k] = false;
                    grid_index = gguf_cuda_iq3_find_best_neighbour(
                        iq3xs_grid, 512, 3, xval + 4 * k, waux + 4 * k, this_scale, laux + 4 * k);
                }
            }

            float sumqx = 0.0f;
            float sumq2 = 0.0f;
            for (int i = 0; i < 32; ++i) {
                const float w = weight[i];
                const float q = 2.0f * laux[i] + 1.0f;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0.0f && sumqx * sumqx > best * sumq2) {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                for (int i = 0; i < 32; ++i) {
                    l[i] = laux[i];
                }
                for (int k = 0; k < 8; ++k) {
                    is_on_grid[k] = is_on_grid_aux[k];
                }
            }
        }

        bool any_not_on_grid = false;
        for (int k = 0; k < 8; ++k) {
            any_not_on_grid = any_not_on_grid || !is_on_grid[k];
        }
        if (any_not_on_grid && scale > 0.0f) {
            const float id = 1.0f / scale;
            for (int k = 0; k < 8; ++k) {
                for (int i = 0; i < 4; ++i) {
                    int q = gguf_cuda_nearest_int(0.5f * (id * xval[4 * k + i] - 1.0f));
                    l[4 * k + i] = gguf_cuda_clamp_int(q, 0, 7);
                }
                int grid_index = gguf_cuda_iq3_find_grid_index(iq3xs_grid, 512, l + 4 * k);
                if (grid_index < 0) {
                    grid_index = gguf_cuda_iq3_find_best_neighbour(
                        iq3xs_grid, 512, 3, xval + 4 * k, waux + 4 * k, scale, l + 4 * k);
                } else {
                    for (int i = 0; i < 4; ++i) {
                        l[4 * k + i] = gguf_cuda_iq3_grid_l(iq3xs_grid, grid_index, i);
                    }
                }
            }
            float sumqx = 0.0f;
            float sumq2 = 0.0f;
            for (int i = 0; i < 32; ++i) {
                const float w = weight[i];
                const float q = 2.0f * l[i] + 1.0f;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0.0f) {
                scale = sumqx / sumq2;
            }
        }

        if (scale < 0.0f) {
            scale = -scale;
            for (int k = 0; k < 4; ++k) {
                block_signs[k] = ~block_signs[k];
            }
        }
        for (int k = 0; k < 8; ++k) {
            int grid_index = gguf_cuda_iq3_find_grid_index(iq3xs_grid, 512, l + 4 * k);
            yb->qs[8 * ib + k] = grid_index & 255;
            yb->qh[(8 * ib + k) / 8] |= ((grid_index >> 8) << ((8 * ib + k) % 8));
        }
        for (int k = 0; k < 4; ++k) {
            yb->signs[4 * ib + k] = block_signs[k];
        }
        scales[ib] = scale;
        max_scale = fmaxf(max_scale, scale);
    }

    if (max_scale == 0.0f) {
        return;
    }

    const float d = max_scale / 31.0f;
    yb->d = gguf_cuda_compute_fp32_to_fp16(d * 1.033f);
    const float id = 1.0f / d;
    for (int ib = 0; ib < QK_K / 32; ib += 2) {
        int q1 = gguf_cuda_nearest_int(0.5f * (id * scales[ib + 0] - 1.0f));
        int q2 = gguf_cuda_nearest_int(0.5f * (id * scales[ib + 1] - 1.0f));
        q1 = gguf_cuda_clamp_int(q1, 0, 15);
        q2 = gguf_cuda_clamp_int(q2, 0, 15);
        yb->scales[ib / 2] = q1 | (q2 << 4);
    }
}

static __global__ void quantize_block_iq1_s(
    const float * __restrict__ x,
    const float * __restrict__ quant_weights,
    block_iq1_s * __restrict__ y,
    int64_t n_blocks,
    int64_t n_blocks_per_row
) {
    const int64_t iblock = blockDim.x * blockIdx.x + threadIdx.x;
    if (iblock >= n_blocks) {
        return;
    }

    const float * xb = x + iblock * QK_K;
    const int64_t row_block = iblock % n_blocks_per_row;
    const float * qw_block = quant_weights + row_block * QK_K;
    block_iq1_s * yb = y + iblock;
    const float x_p[3] = {-1.0f + IQ1S_DELTA, IQ1S_DELTA, 1.0f + IQ1S_DELTA};
    const float x_m[3] = {-1.0f - IQ1S_DELTA, -IQ1S_DELTA, 1.0f - IQ1S_DELTA};
    float scales[QK_K / 32];
    float weight[32];
    float sumx[33];
    float sumw[33];
    int idx[32];
    int8_t l[32];
    uint16_t index[4];
    int8_t shifts[QK_K / 32];

    yb->d = gguf_cuda_compute_fp32_to_fp16(0.0f);
    for (int i = 0; i < QK_K / 8; ++i) {
        yb->qs[i] = 0;
    }
    for (int i = 0; i < QK_K / 32; ++i) {
        yb->qh[i] = 0;
    }

    float max_scale = 0.0f;
    float sumx2 = 0.0f;
    for (int i = 0; i < QK_K; ++i) {
        sumx2 += xb[i] * xb[i];
    }
    const float sigma2 = 2.0f * sumx2 / QK_K;

    for (int ib = 0; ib < QK_K / 32; ++ib) {
        const float * x32 = xb + 32 * ib;
        for (int i = 0; i < 32; ++i) {
            weight[i] = qw_block[32 * ib + i] * sqrtf(sigma2 + x32[i] * x32[i]);
            idx[i] = i;
            l[i] = 1;
        }
        float maxv = fabsf(x32[0]);
        for (int i = 1; i < 32; ++i) {
            maxv = fmaxf(maxv, fabsf(x32[i]));
        }
        if (maxv < GROUP_MAX_EPS_IQ1_S) {
            scales[ib] = 0.0f;
            shifts[ib] = 1;
            continue;
        }

        for (int i = 1; i < 32; ++i) {
            const int value = idx[i];
            int j = i - 1;
            while (j >= 0 && x32[idx[j]] > x32[value]) {
                idx[j + 1] = idx[j];
                --j;
            }
            idx[j + 1] = value;
        }

        sumx[0] = 0.0f;
        sumw[0] = 0.0f;
        for (int j = 0; j < 32; ++j) {
            const int i = idx[j];
            sumx[j + 1] = sumx[j] + weight[i] * x32[i];
            sumw[j + 1] = sumw[j] + weight[i];
        }

        float best_score = -FLT_MAX;
        float scale = maxv;
        int best_i1 = -1;
        int best_i2 = -1;
        int best_shift = 0;
        for (int i1 = 0; i1 <= 32; ++i1) {
            for (int i2 = i1; i2 <= 32; ++i2) {
                float w0 = sumw[i1] - sumw[0];
                float w1 = sumw[i2] - sumw[i1];
                float w2 = sumw[32] - sumw[i2];
                float sx0 = sumx[i1] - sumx[0];
                float sx1 = sumx[i2] - sumx[i1];
                float sx2 = sumx[32] - sumx[i2];
                float sumqx = sx0 * x_p[0] + sx1 * x_p[1] + sx2 * x_p[2];
                float sumq2 = w0 * x_p[0] * x_p[0] + w1 * x_p[1] * x_p[1] + w2 * x_p[2] * x_p[2];
                if (sumq2 > 0.0f && sumqx * sumqx > best_score * sumq2) {
                    scale = sumqx / sumq2;
                    best_score = scale * sumqx;
                    best_i1 = i1;
                    best_i2 = i2;
                    best_shift = 1;
                }
                sumqx = sx0 * x_m[0] + sx1 * x_m[1] + sx2 * x_m[2];
                sumq2 = w0 * x_m[0] * x_m[0] + w1 * x_m[1] * x_m[1] + w2 * x_m[2] * x_m[2];
                if (sumq2 > 0.0f && sumqx * sumqx > best_score * sumq2) {
                    scale = sumqx / sumq2;
                    best_score = scale * sumqx;
                    best_i1 = i1;
                    best_i2 = i2;
                    best_shift = -1;
                }
            }
        }
        if (best_i1 < 0 || best_i2 < 0 || best_shift == 0) {
            scales[ib] = 0.0f;
            shifts[ib] = 1;
            continue;
        }
        for (int j = 0; j < best_i1; ++j) {
            l[idx[j]] = 0;
        }
        for (int j = best_i1; j < best_i2; ++j) {
            l[idx[j]] = 1;
        }
        for (int j = best_i2; j < 32; ++j) {
            l[idx[j]] = 2;
        }
        if (scale < 0.0f) {
            for (int j = 0; j < 32; ++j) {
                l[j] = 2 - l[j];
            }
            scale = -scale;
            best_shift = -best_shift;
        }

        bool all_on_grid = true;
        const float * values = best_shift == 1 ? x_p : x_m;
        for (int k = 0; k < 4; ++k) {
            int grid_index = gguf_cuda_iq1_find_grid_index(l + 8 * k);
            if (grid_index < 0) {
                all_on_grid = false;
                grid_index = gguf_cuda_iq1_find_best_neighbour(x32 + 8 * k, weight + 8 * k, scale, values, l + 8 * k);
            }
            index[k] = grid_index;
        }
        if (!all_on_grid) {
            float sumqx = 0.0f;
            float sumq2 = 0.0f;
            for (int k = 0; k < 4; ++k) {
                for (int j = 0; j < 8; ++j) {
                    const float w = weight[8 * k + j];
                    const float q = values[gguf_cuda_iq1_grid_l(index[k], j)];
                    sumqx += w * q * x32[8 * k + j];
                    sumq2 += w * q * q;
                }
            }
            if (sumqx > 0.0f && sumq2 > 0.0f) {
                scale = sumqx / sumq2;
            }
        }

        uint16_t h = 0;
        for (int k = 0; k < 4; ++k) {
            yb->qs[4 * ib + k] = index[k] & 255;
            h |= (index[k] >> 8) << (3 * k);
        }
        yb->qh[ib] = h;
        scales[ib] = scale;
        shifts[ib] = best_shift;
        max_scale = fmaxf(max_scale, scale);
    }

    if (max_scale == 0.0f) {
        return;
    }

    const float d = max_scale / 15.0f;
    yb->d = gguf_cuda_compute_fp32_to_fp16(d * 1.125f);
    const float id = 1.0f / d;
    for (int ib = 0; ib < QK_K / 32; ++ib) {
        int q = gguf_cuda_nearest_int(0.5f * (id * scales[ib] - 1.0f));
        q = gguf_cuda_clamp_int(q, 0, 7);
        if (shifts[ib] == -1) {
            q |= 8;
        }
        yb->qh[ib] |= q << 12;
    }
}

static __global__ void quantize_block_iq1_m(
    const float * __restrict__ x,
    const float * __restrict__ quant_weights,
    block_iq1_m * __restrict__ y,
    int64_t n_blocks,
    int64_t n_blocks_per_row
) {
    const int64_t iblock = blockDim.x * blockIdx.x + threadIdx.x;
    if (iblock >= n_blocks) {
        return;
    }

    const float * xb = x + iblock * QK_K;
    const int64_t row_block = iblock % n_blocks_per_row;
    const float * qw_block = quant_weights ? quant_weights + row_block * QK_K : nullptr;
    block_iq1_m * yb = y + iblock;
    const float x_p[3] = {-1.0f + IQ1M_DELTA, IQ1M_DELTA, 1.0f + IQ1M_DELTA};
    const float x_m[3] = {-1.0f - IQ1M_DELTA, -IQ1M_DELTA, 1.0f - IQ1M_DELTA};
    const uint8_t masks[4] = {0x00, 0x80, 0x08, 0x88};
    float scales[QK_K / 16];
    float weight[16];
    int idx[16];
    int8_t l[16];
    uint16_t index[2];
    int8_t shifts[QK_K / 16];

    for (int i = 0; i < QK_K / 8; ++i) {
        yb->qs[i] = 0;
    }
    for (int i = 0; i < QK_K / 16; ++i) {
        yb->qh[i] = 0;
    }
    for (int i = 0; i < QK_K / 32; ++i) {
        yb->scales[i] = 0;
    }

    float max_scale = 0.0f;
    float sumx2 = 0.0f;
    for (int i = 0; i < QK_K; ++i) {
        sumx2 += xb[i] * xb[i];
    }
    const float sigma2 = 2.0f * sumx2 / QK_K;

    for (int ib = 0; ib < QK_K / 16; ++ib) {
        const float * x16 = xb + 16 * ib;
        for (int i = 0; i < 16; ++i) {
            weight[i] = qw_block ? qw_block[16 * ib + i] * sqrtf(sigma2 + x16[i] * x16[i])
                                 : x16[i] * x16[i];
            idx[i] = i;
            l[i] = 1;
        }
        float maxv = fabsf(x16[0]);
        for (int i = 1; i < 16; ++i) {
            maxv = fmaxf(maxv, fabsf(x16[i]));
        }
        if (maxv < GROUP_MAX_EPS_IQ1_M) {
            scales[ib] = 0.0f;
            shifts[ib] = 0;
            continue;
        }

        for (int i = 1; i < 16; ++i) {
            const int value = idx[i];
            int j = i - 1;
            while (j >= 0 && x16[idx[j]] > x16[value]) {
                idx[j + 1] = idx[j];
                --j;
            }
            idx[j + 1] = value;
        }

        float best_score = -FLT_MAX;
        float scale = maxv;
        int best_i1 = -1;
        int best_i2 = -1;
        int best_k = -1;
        for (int i1 = 0; i1 <= 16; ++i1) {
            for (int i2 = i1; i2 <= 16; ++i2) {
                float sumqx[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                float sumq2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (int j = 0; j < 16; ++j) {
                    const int i = idx[j];
                    const int level = j < i1 ? 0 : j < i2 ? 1 : 2;
                    for (int k = 0; k < 4; ++k) {
                        const float * values = (i < 8 ? (k < 2 ? x_p : x_m) : (k % 2 == 0 ? x_p : x_m));
                        const float q = values[level];
                        const float w = weight[i];
                        sumqx[k] += w * q * x16[i];
                        sumq2[k] += w * q * q;
                    }
                }
                for (int k = 0; k < 4; ++k) {
                    if (sumq2[k] > 0.0f && sumqx[k] * sumqx[k] > best_score * sumq2[k]) {
                        scale = sumqx[k] / sumq2[k];
                        best_score = scale * sumqx[k];
                        best_i1 = i1;
                        best_i2 = i2;
                        best_k = k;
                    }
                }
            }
        }
        if (best_i1 < 0 || best_i2 < 0 || best_k < 0) {
            scales[ib] = 0.0f;
            shifts[ib] = 0;
            continue;
        }
        for (int j = 0; j < best_i1; ++j) {
            l[idx[j]] = 0;
        }
        for (int j = best_i1; j < best_i2; ++j) {
            l[idx[j]] = 1;
        }
        for (int j = best_i2; j < 16; ++j) {
            l[idx[j]] = 2;
        }
        if (scale < 0.0f) {
            for (int j = 0; j < 16; ++j) {
                l[j] = 2 - l[j];
            }
            scale = -scale;
            best_k = best_k == 0 ? 3 : best_k == 1 ? 2 : best_k == 2 ? 1 : 0;
        }

        bool all_on_grid = true;
        for (int k = 0; k < 2; ++k) {
            const float * values = (k == 0 ? (best_k < 2 ? x_p : x_m) : (best_k % 2 == 0 ? x_p : x_m));
            int grid_index = gguf_cuda_iq1_find_grid_index(l + 8 * k);
            if (grid_index < 0) {
                all_on_grid = false;
                grid_index = gguf_cuda_iq1_find_best_neighbour(x16 + 8 * k, weight + 8 * k, scale, values, l + 8 * k);
            }
            index[k] = grid_index;
        }
        if (!all_on_grid) {
            float sumqx = 0.0f;
            float sumq2 = 0.0f;
            for (int k = 0; k < 2; ++k) {
                const float * values = (k == 0 ? (best_k < 2 ? x_p : x_m) : (best_k % 2 == 0 ? x_p : x_m));
                for (int j = 0; j < 8; ++j) {
                    const float w = weight[8 * k + j];
                    const float q = values[gguf_cuda_iq1_grid_l(index[k], j)];
                    sumqx += w * q * x16[8 * k + j];
                    sumq2 += w * q * q;
                }
            }
            if (sumqx > 0.0f && sumq2 > 0.0f) {
                scale = sumqx / sumq2;
            }
        }

        yb->qs[2 * ib + 0] = index[0] & 255;
        yb->qs[2 * ib + 1] = index[1] & 255;
        yb->qh[ib] = (index[0] >> 8) | ((index[1] >> 8) << 4);
        scales[ib] = scale;
        shifts[ib] = best_k;
        max_scale = fmaxf(max_scale, scale);
    }

    if (max_scale == 0.0f) {
        return;
    }

    uint16_t * sc = (uint16_t *)yb->scales;
    const float d0 = max_scale / 15.0f;
    const float id = 1.0f / d0;
    float sumqx = 0.0f;
    float sumq2 = 0.0f;
    for (int ib = 0; ib < QK_K / 16; ++ib) {
        int qscale = gguf_cuda_nearest_int(0.5f * (id * scales[ib] - 1.0f));
        qscale = gguf_cuda_clamp_int(qscale, 0, 7);
        sc[ib / 4] |= qscale << (3 * (ib % 4));
        yb->qh[ib] |= masks[shifts[ib]];

        const float * x16 = xb + 16 * ib;
        for (int i = 0; i < 16; ++i) {
            weight[i] = qw_block ? qw_block[16 * ib + i] * sqrtf(sigma2 + x16[i] * x16[i])
                                 : x16[i] * x16[i];
        }
        for (int k = 0; k < 2; ++k) {
            const float * values = (k == 0 ? (shifts[ib] < 2 ? x_p : x_m) : (shifts[ib] % 2 == 0 ? x_p : x_m));
            const int grid_index = yb->qs[2 * ib + k] | (((yb->qh[ib] >> (4 * k)) & 0x07) << 8);
            for (int j = 0; j < 8; ++j) {
                const float w = weight[8 * k + j];
                const float q = values[gguf_cuda_iq1_grid_l(grid_index, j)] * (2 * qscale + 1);
                sumqx += w * q * x16[8 * k + j];
                sumq2 += w * q * q;
            }
        }
    }
    float d = d0;
    if (sumq2 > 0.0f) {
        d = sumqx / sumq2;
    }
    const uint16_t scale_bits = gguf_cuda_compute_fp32_to_fp16(d * 1.1125f);
    sc[0] |= (scale_bits & 0x000fu) << 12;
    sc[1] |= (scale_bits & 0x00f0u) << 8;
    sc[2] |= (scale_bits & 0x0f00u) << 4;
    sc[3] |= (scale_bits & 0xf000u);
}

static __global__ void quantize_block_tq1_0(const float * __restrict__ x, block_tq1_0 * __restrict__ y, int64_t n_blocks) {
    const int64_t ib = blockDim.x * blockIdx.x + threadIdx.x;
    if (ib >= n_blocks) {
        return;
    }

    const float * xb = x + ib * QK_K;
    float amax = 0.0f;
    for (int j = 0; j < QK_K; ++j) {
        amax = fmaxf(amax, fabsf(xb[j]));
    }

    const float d = amax;
    const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
    y[ib].d = gguf_cuda_compute_fp32_to_fp16(d);

    int offset = 0;
    for (size_t j = 0; j < sizeof(block_tq1_0::qs) - sizeof(block_tq1_0::qs) % 32; j += 32) {
        for (size_t m = 0; m < 32; ++m) {
            uint8_t q = 0;
            for (size_t n = 0; n < 5; ++n) {
                const int xi = lroundf(__fmul_rn(xb[offset + m + n * 32], id)) + 1;
                q *= 3;
                q += xi;
            }
            y[ib].qs[j + m] = gguf_cuda_pack_trits_5(q);
        }
        offset += 5 * 32;
    }
    for (size_t j = sizeof(block_tq1_0::qs) - sizeof(block_tq1_0::qs) % 32; j < sizeof(block_tq1_0::qs); j += 16) {
        for (size_t m = 0; m < 16; ++m) {
            uint8_t q = 0;
            for (size_t n = 0; n < 5; ++n) {
                const int xi = lroundf(__fmul_rn(xb[offset + m + n * 16], id)) + 1;
                q *= 3;
                q += xi;
            }
            y[ib].qs[j + m] = gguf_cuda_pack_trits_5(q);
        }
        offset += 5 * 16;
    }
    for (size_t j = 0; j < sizeof(block_tq1_0::qh); ++j) {
        uint8_t q = 0;
        for (size_t m = 0; m < 4; ++m) {
            const int xi = lroundf(__fmul_rn(xb[offset + j + m * sizeof(block_tq1_0::qh)], id)) + 1;
            q *= 3;
            q += xi;
        }
        q *= 3;
        y[ib].qh[j] = gguf_cuda_pack_trits_5(q);
    }
}

static __global__ void quantize_block_tq2_0(const float * __restrict__ x, block_tq2_0 * __restrict__ y, int64_t n_blocks) {
    const int64_t ib = blockDim.x * blockIdx.x + threadIdx.x;
    if (ib >= n_blocks) {
        return;
    }

    const float * xb = x + ib * QK_K;
    float amax = 0.0f;
    for (int j = 0; j < QK_K; ++j) {
        amax = fmaxf(amax, fabsf(xb[j]));
    }

    const float d = amax;
    const float id = d != 0.0f ? __fdiv_rn(1.0f, d) : 0.0f;
    y[ib].d = gguf_cuda_compute_fp32_to_fp16(d);

    int offset = 0;
    for (size_t j = 0; j < sizeof(block_tq2_0::qs); j += 32) {
        for (size_t m = 0; m < 32; ++m) {
            uint8_t q = 0;
            for (size_t n = 0; n < 4; ++n) {
                const int xi = lroundf(__fmul_rn(xb[offset + m + n * 32], id)) + 1;
                q += (xi & 3) << (2 * n);
            }
            y[ib].qs[j + m] = q;
        }
        offset += 4 * 32;
    }
}

static __global__ void quantize_block_q6_K(const float * __restrict__ x, block_q6_K * __restrict__ y, int64_t n_blocks) {
    const int64_t iblock = blockDim.x * blockIdx.x + threadIdx.x;
    if (iblock >= n_blocks) {
        return;
    }

    const float * xb = x + iblock * QK_K;
    block_q6_K * yb = y + iblock;
    int8_t l[QK_K];
    float scales[QK_K / 16];
    int8_t quant_scales[QK_K / 16];

    float max_scale = 0.0f;
    float max_abs_scale = 0.0f;
    for (int ib = 0; ib < QK_K / 16; ++ib) {
        const float scale = gguf_cuda_make_qx_quants_rmse1(16, 32, xb + 16 * ib, l + 16 * ib);
        scales[ib] = scale;
        const float abs_scale = fabsf(scale);
        if (abs_scale > max_abs_scale) {
            max_abs_scale = abs_scale;
            max_scale = scale;
        }
    }

    if (max_abs_scale < GROUP_MAX_EPS) {
        memset(yb, 0, sizeof(block_q6_K));
        yb->d = gguf_cuda_compute_fp32_to_fp16(0.0f);
        return;
    }

    const float iscale = __fdiv_rn(-128.0f, max_scale);
    yb->d = gguf_cuda_compute_fp32_to_fp16(__fdiv_rn(1.0f, iscale));
    const float d_super = __half2float(gguf_cuda_load_half(yb->d));
    for (int ib = 0; ib < QK_K / 16; ++ib) {
        const int8_t qscale = min(127, gguf_cuda_nearest_int(__fmul_rn(iscale, scales[ib])));
        quant_scales[ib] = qscale;
        yb->scales[ib] = qscale;
    }

    for (int j = 0; j < QK_K / 16; ++j) {
        const float d = d_super * quant_scales[j];
        if (d == 0.0f) {
            continue;
        }
        for (int ii = 0; ii < 16; ++ii) {
            int q = gguf_cuda_nearest_int(__fdiv_rn(xb[16 * j + ii], d));
            q = max(-32, min(31, q));
            l[16 * j + ii] = q + 32;
        }
    }

    uint8_t * ql = yb->ql;
    uint8_t * qh = yb->qh;
    for (int j = 0; j < QK_K; j += 128) {
        for (int i = 0; i < 32; ++i) {
            const uint8_t q1 = l[j + i + 0] & 0x0f;
            const uint8_t q2 = l[j + i + 32] & 0x0f;
            const uint8_t q3 = l[j + i + 64] & 0x0f;
            const uint8_t q4 = l[j + i + 96] & 0x0f;
            ql[i + 0] = q1 | (q3 << 4);
            ql[i + 32] = q2 | (q4 << 4);
            qh[i] = (l[j + i] >> 4) | ((l[j + i + 32] >> 4) << 2) |
                    ((l[j + i + 64] >> 4) << 4) | ((l[j + i + 96] >> 4) << 6);
        }
        ql += 64;
        qh += 32;
    }
}

static inline int64_t gguf_cuda_quantize_row_size(int64_t type, int64_t n) {
    switch (type) {
        case GGML_TYPE_IQ4_NL:
            return n * (int64_t)sizeof(block_iq4_nl) / QK4_NL;
        case GGML_TYPE_IQ4_XS:
            return n * (int64_t)sizeof(block_iq4_xs) / QK_K;
        case GGML_TYPE_IQ2_XXS:
            return n * (int64_t)sizeof(block_iq2_xxs) / QK_K;
        case GGML_TYPE_IQ2_XS:
            return n * (int64_t)sizeof(block_iq2_xs) / QK_K;
        case GGML_TYPE_IQ2_S:
            return n * (int64_t)sizeof(block_iq2_s) / QK_K;
        case GGML_TYPE_IQ3_XXS:
            return n * (int64_t)sizeof(block_iq3_xxs) / QK_K;
        case GGML_TYPE_IQ3_S:
            return n * (int64_t)sizeof(block_iq3_s) / QK_K;
        case GGML_TYPE_IQ1_S:
            return n * (int64_t)sizeof(block_iq1_s) / QK_K;
        case GGML_TYPE_IQ1_M:
            return n * (int64_t)sizeof(block_iq1_m) / QK_K;
        case GGML_TYPE_MXFP4:
            return n * (int64_t)sizeof(block_mxfp4) / QK_MXFP4;
        case GGML_TYPE_NVFP4:
            return n * (int64_t)sizeof(block_nvfp4) / QK_NVFP4;
        case GGML_TYPE_Q1_0:
            return n * (int64_t)sizeof(block_q1_0) / QK1_0;
        case GGML_TYPE_Q2_K:
            return n * (int64_t)sizeof(block_q2_K) / QK_K;
        case GGML_TYPE_Q3_K:
            return n * (int64_t)sizeof(block_q3_K) / QK_K;
        case GGML_TYPE_Q4_K:
            return n * (int64_t)sizeof(block_q4_K) / QK_K;
        case GGML_TYPE_Q4_0:
            return n * (int64_t)sizeof(block_q4_0) / QK4_0;
        case GGML_TYPE_Q4_1:
            return n * (int64_t)sizeof(block_q4_1) / QK4_1;
        case GGML_TYPE_Q5_0:
            return n * (int64_t)sizeof(block_q5_0) / QK5_0;
        case GGML_TYPE_Q5_1:
            return n * (int64_t)sizeof(block_q5_1) / QK5_1;
        case GGML_TYPE_Q5_K:
            return n * (int64_t)sizeof(block_q5_K) / QK_K;
        case GGML_TYPE_Q6_K:
            return n * (int64_t)sizeof(block_q6_K) / QK_K;
        case GGML_TYPE_Q8_0:
            return n * (int64_t)sizeof(block_q8_0) / QK8_0;
        case GGML_TYPE_TQ1_0:
            return n * (int64_t)sizeof(block_tq1_0) / QK_K;
        case GGML_TYPE_TQ2_0:
            return n * (int64_t)sizeof(block_tq2_0) / QK_K;
        default:
            return 0;
    }
}

static inline int64_t gguf_cuda_quantize_block_size(int64_t type) {
    switch (type) {
        case GGML_TYPE_IQ4_NL:
            return QK4_NL;
        case GGML_TYPE_IQ4_XS:
            return QK_K;
        case GGML_TYPE_IQ2_XXS:
            return QK_K;
        case GGML_TYPE_IQ2_XS:
            return QK_K;
        case GGML_TYPE_IQ2_S:
            return QK_K;
        case GGML_TYPE_IQ3_XXS:
            return QK_K;
        case GGML_TYPE_IQ3_S:
            return QK_K;
        case GGML_TYPE_IQ1_S:
            return QK_K;
        case GGML_TYPE_IQ1_M:
            return QK_K;
        case GGML_TYPE_MXFP4:
            return QK_MXFP4;
        case GGML_TYPE_NVFP4:
            return QK_NVFP4;
        case GGML_TYPE_Q1_0:
            return QK1_0;
        case GGML_TYPE_Q2_K:
            return QK_K;
        case GGML_TYPE_Q3_K:
            return QK_K;
        case GGML_TYPE_Q4_K:
            return QK_K;
        case GGML_TYPE_Q4_0:
            return QK4_0;
        case GGML_TYPE_Q4_1:
            return QK4_1;
        case GGML_TYPE_Q5_0:
            return QK5_0;
        case GGML_TYPE_Q5_1:
            return QK5_1;
        case GGML_TYPE_Q5_K:
            return QK_K;
        case GGML_TYPE_Q6_K:
            return QK_K;
        case GGML_TYPE_Q8_0:
            return QK8_0;
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
            return QK_K;
        default:
            return 0;
    }
}

static inline bool gguf_cuda_quantize_needs_imatrix(int64_t type) {
    switch (type) {
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ1_S:
            return true;
        default:
            return false;
    }
}

static inline void quantize_row_cuda(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t type,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
) {
    const int threads = 256;
    constexpr int q_warp_threads = 128;
    constexpr int q_blocks_per_cta = q_warp_threads / 32;
    // K-quant launch sizes were selected empirically on SM 8.6 while preserving byte-exact math.
    const int q2_k_threads = 128;
    constexpr int q3_k_warp_threads = 128;
    constexpr int q3_k_warps_per_cta = q3_k_warp_threads / 32;
    constexpr int q4_k_threads = 128;
    constexpr int q4_k_blocks_per_cta = (q4_k_threads / 32) * 4;
    constexpr int q5_k_threads = 64;
    constexpr int q5_k_blocks_per_cta = (q5_k_threads / 32) * 4;
    const int q6_k_threads = 96;
    switch (type) {
        case GGML_TYPE_IQ4_NL: {
            const int64_t n_blocks = k / QK4_NL;
            const int blocks = (int)((n_blocks + threads - 1) / threads);
            quantize_block_iq4_nl<<<blocks, threads, 0, stream>>>(x, (block_iq4_nl *)y, n_blocks);
            return;
        }
        case GGML_TYPE_IQ4_XS: {
            const int64_t n_blocks = k / QK_K;
            const int blocks = (int)((n_blocks + threads - 1) / threads);
            quantize_block_iq4_xs<<<blocks, threads, 0, stream>>>(x, (block_iq4_xs *)y, n_blocks);
            return;
        }
        case GGML_TYPE_IQ2_XXS: {
            const int64_t n_blocks = k / QK_K;
            const int blocks = (int)((n_blocks + threads - 1) / threads);
            const int64_t n_blocks_per_row = n_per_row / QK_K;
            quantize_block_iq2_xxs<<<blocks, threads, 0, stream>>>(
                x, quant_weights, (block_iq2_xxs *)y, n_blocks, n_blocks_per_row);
            return;
        }
        case GGML_TYPE_IQ2_XS: {
            const int64_t n_blocks = k / QK_K;
            const int blocks = (int)((n_blocks + threads - 1) / threads);
            const int64_t n_blocks_per_row = n_per_row / QK_K;
            quantize_block_iq2_xs<<<blocks, threads, 0, stream>>>(
                x, quant_weights, (block_iq2_xs *)y, n_blocks, n_blocks_per_row);
            return;
        }
        case GGML_TYPE_IQ2_S: {
            const int64_t n_blocks = k / QK_K;
            const int blocks = (int)((n_blocks + threads - 1) / threads);
            const int64_t n_blocks_per_row = n_per_row / QK_K;
            quantize_block_iq2_s<<<blocks, threads, 0, stream>>>(
                x, quant_weights, (block_iq2_s *)y, n_blocks, n_blocks_per_row);
            return;
        }
        case GGML_TYPE_IQ3_XXS: {
            const int64_t n_blocks = k / QK_K;
            const int blocks = (int)((n_blocks + threads - 1) / threads);
            quantize_block_iq3_xxs<<<blocks, threads, 0, stream>>>(x, (block_iq3_xxs *)y, n_blocks);
            return;
        }
        case GGML_TYPE_IQ3_S: {
            const int64_t n_blocks = k / QK_K;
            const int blocks = (int)((n_blocks + threads - 1) / threads);
            quantize_block_iq3_s<<<blocks, threads, 0, stream>>>(x, (block_iq3_s *)y, n_blocks);
            return;
        }
        case GGML_TYPE_IQ1_S: {
            const int64_t n_blocks = k / QK_K;
            const int blocks = (int)((n_blocks + threads - 1) / threads);
            const int64_t n_blocks_per_row = n_per_row / QK_K;
            quantize_block_iq1_s<<<blocks, threads, 0, stream>>>(
                x, quant_weights, (block_iq1_s *)y, n_blocks, n_blocks_per_row);
            return;
        }
        case GGML_TYPE_IQ1_M: {
            const int64_t n_blocks = k / QK_K;
            const int blocks = (int)((n_blocks + threads - 1) / threads);
            const int64_t n_blocks_per_row = n_per_row / QK_K;
            quantize_block_iq1_m<<<blocks, threads, 0, stream>>>(
                x, quant_weights, (block_iq1_m *)y, n_blocks, n_blocks_per_row);
            return;
        }
        case GGML_TYPE_MXFP4: {
            const int64_t n_blocks = k / QK_MXFP4;
            const int blocks = (int)((n_blocks + threads - 1) / threads);
            quantize_block_mxfp4<<<blocks, threads, 0, stream>>>(x, (block_mxfp4 *)y, n_blocks);
            return;
        }
        case GGML_TYPE_NVFP4: {
            const int64_t n_blocks = k / QK_NVFP4;
            const int blocks = (int)((n_blocks + threads - 1) / threads);
            quantize_block_nvfp4<<<blocks, threads, 0, stream>>>(x, (block_nvfp4 *)y, n_blocks);
            return;
        }
        case GGML_TYPE_Q1_0: {
            const int64_t n_blocks = k / QK1_0;
            const int blocks = (int)((n_blocks + threads - 1) / threads);
            quantize_block_q1_0<<<blocks, threads, 0, stream>>>(x, (block_q1_0 *)y, n_blocks);
            return;
        }
        case GGML_TYPE_Q2_K: {
            const int64_t n_blocks = k / QK_K;
            const int blocks = (int)((n_blocks + q2_k_threads - 1) / q2_k_threads);
            quantize_block_q2_K<<<blocks, q2_k_threads, 0, stream>>>(x, (block_q2_K *)y, n_blocks);
            return;
        }
        case GGML_TYPE_Q3_K: {
            const int64_t n_blocks = k / QK_K;
            const int blocks = (int)((n_blocks + q3_k_warps_per_cta - 1) / q3_k_warps_per_cta);
            const size_t smem = q3_k_warps_per_cta * (QK_K * sizeof(int8_t) + (QK_K / 16) * sizeof(float));
            quantize_block_q3_K_warp<<<blocks, q3_k_warp_threads, smem, stream>>>(x, (block_q3_K *)y, n_blocks);
            return;
        }
        case GGML_TYPE_Q4_K: {
            const int64_t n_blocks = k / QK_K;
            const int blocks = (int)((n_blocks + q4_k_blocks_per_cta - 1) / q4_k_blocks_per_cta);
            const size_t smem = q4_k_blocks_per_cta * (QK_K * sizeof(uint8_t) + 2 * (QK_K / 32) * sizeof(float));
            quantize_block_q4_K<<<blocks, q4_k_threads, smem, stream>>>(x, (block_q4_K *)y, n_blocks);
            return;
        }
        case GGML_TYPE_Q4_0: {
            const int64_t n_blocks = k / QK4_0;
            const int blocks = (int)((n_blocks + threads - 1) / threads);
            quantize_block_q4_0<<<blocks, threads, 0, stream>>>(x, (block_q4_0 *)y, n_blocks);
            return;
        }
        case GGML_TYPE_Q4_1: {
            const int64_t n_blocks = k / QK4_1;
            const int blocks = (int)((n_blocks + threads - 1) / threads);
            quantize_block_q4_1<<<blocks, threads, 0, stream>>>(x, (block_q4_1 *)y, n_blocks);
            return;
        }
        case GGML_TYPE_Q5_0: {
            const int64_t n_blocks = k / QK5_0;
            const int blocks = (int)((n_blocks + q_blocks_per_cta - 1) / q_blocks_per_cta);
            quantize_block_q5_0_warp<<<blocks, q_warp_threads, 0, stream>>>(x, (block_q5_0 *)y, n_blocks);
            return;
        }
        case GGML_TYPE_Q5_1: {
            const int64_t n_blocks = k / QK5_1;
            const int blocks = (int)((n_blocks + q_blocks_per_cta - 1) / q_blocks_per_cta);
            quantize_block_q5_1_warp<<<blocks, q_warp_threads, 0, stream>>>(x, (block_q5_1 *)y, n_blocks);
            return;
        }
        case GGML_TYPE_Q5_K: {
            const int64_t n_blocks = k / QK_K;
            const int blocks = (int)((n_blocks + q5_k_blocks_per_cta - 1) / q5_k_blocks_per_cta);
            const size_t smem = q5_k_blocks_per_cta * (QK_K * sizeof(uint8_t) + 2 * (QK_K / 32) * sizeof(float));
            quantize_block_q5_K<<<blocks, q5_k_threads, smem, stream>>>(x, (block_q5_K *)y, n_blocks);
            return;
        }
        case GGML_TYPE_Q6_K: {
            const int64_t n_blocks = k / QK_K;
            const int blocks = (int)((n_blocks + q6_k_threads - 1) / q6_k_threads);
            quantize_block_q6_K<<<blocks, q6_k_threads, 0, stream>>>(x, (block_q6_K *)y, n_blocks);
            return;
        }
        case GGML_TYPE_Q8_0: {
            const int64_t n_blocks = k / QK8_0;
            const int blocks = (int)((n_blocks + q_blocks_per_cta - 1) / q_blocks_per_cta);
            quantize_block_q8_0_warp<<<blocks, q_warp_threads, 0, stream>>>(x, (block_q8_0 *)y, n_blocks);
            return;
        }
        case GGML_TYPE_TQ1_0: {
            const int64_t n_blocks = k / QK_K;
            const int blocks = (int)((n_blocks + threads - 1) / threads);
            quantize_block_tq1_0<<<blocks, threads, 0, stream>>>(x, (block_tq1_0 *)y, n_blocks);
            return;
        }
        case GGML_TYPE_TQ2_0: {
            const int64_t n_blocks = k / QK_K;
            const int blocks = (int)((n_blocks + threads - 1) / threads);
            quantize_block_tq2_0<<<blocks, threads, 0, stream>>>(x, (block_tq2_0 *)y, n_blocks);
            return;
        }
        default:
            return;
    }
}
