#pragma once

#include <libgguf/gpu/quantize/common.cuh>

namespace libgguf
{
namespace gpu
{

template<class Input>
static __global__ void quantize_weighted_q3_k(const Input* src, const float* importance,
                                             block_q3_K* dst, int64_t blocks_per_row) {
    constexpr int width = 16;
    constexpr int groups = QK_K / width;
    const int lane = threadIdx.x;
    const int64_t block = blockIdx.x;
    const Input* x = src + block * QK_K;
    const float* qw = importance + (block % blocks_per_row) * QK_K;
    auto& y = dst[block];
    __shared__ float sigma2, rounded_scale;
    __shared__ float scales[groups], sums[groups];
    __shared__ int8_t scale_codes[groups], codes[QK_K];

    if (lane == 0) {
        float sum = 0;
        for (int i = 0; i < QK_K; ++i) {
            const float v = static_cast<float>(x[i]);
            sum = __fadd_rn(sum, __fmul_rn(v, v));
        }
        sigma2 = sum * (2.0f / QK_K);
    }
    __syncthreads();
    if (lane < groups) {
        float weights[width], total = 0;
        const int start = lane * width;
        for (int i = 0; i < width; ++i) {
            const float v = static_cast<float>(x[start + i]);
            weights[i] = qw[start + i] * sqrtf(sigma2 + v * v);
            total += weights[i];
        }
        sums[lane] = total;
        scales[lane] = make_qx_quants_rmse1(width, 4, x + start, codes + start, weights);
    }
    __syncthreads();
    if (lane == 0) {
        const auto d = compute_fp32_to_fp16(make_qx_quants_rmse1(groups, 32, scales, scale_codes, sums));
        y.d = d;
        rounded_scale = __half2float(load_half(d));
    }
    __syncthreads();
    if (lane < 8)
        y.scales[lane] = (scale_codes[lane] & 15) | ((scale_codes[lane + 8] & 15) << 4);
    if (lane < 4) {
        unsigned high = 0;
        for (int j = 0; j < 4; ++j)
            high |= (scale_codes[lane + 4 * j] >> 4) << (2 * j);
        y.scales[8 + lane] = high;
    }
    for (int i = lane; i < QK_K; i += blockDim.x) {
        const float d = rounded_scale * (scale_codes[i / width] - 32);
        if (d != 0)
            codes[i] = 4 + clamp_int(nearest_int(__fdiv_rn(static_cast<float>(x[i]), d)), -4, 3);
    }
    __syncthreads();
    if (lane < 32) {
        unsigned high = 0;
        for (int j = 0; j < 8; ++j)
            high |= (codes[lane + 32 * j] >> 2) << j;
        y.hmask[lane] = high;
        for (int i = 0; i < QK_K; i += 128) {
            unsigned packed = 0;
            for (int j = 0; j < 4; ++j)
                packed |= (codes[lane + i + 32 * j] & 3) << (2 * j);
            y.qs[i / 4 + lane] = packed;
        }
    }
}

template<typename src_t>
static __device__ void quantize_q3_k(
    const src_t * x,
    block_q3_K * y,
    int64_t n_blocks,
    int8_t * shared)
{
    constexpr int blocks_per_warp = 2;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_cta = blockDim.x >> 5;
    const int blocks_per_cta = warps_per_cta * blocks_per_warp;
    const int block_in_warp = lane >> 4;
    const int subgroup = lane & 15;
    const int block_in_cta = warp * blocks_per_warp + block_in_warp;
    const int64_t iblock = static_cast<int64_t>(blockIdx.x) * blocks_per_cta + block_in_cta;
    const bool valid = iblock < n_blocks;
    const auto warp_mask = __ballot_sync(__activemask(), valid);
    if (!valid)
    {
        return;
    }

    int8_t * l_base = shared;
    float * scales_base = reinterpret_cast<float *>(l_base + blocks_per_cta * QK_K);
    int8_t * qscales_base = reinterpret_cast<int8_t *>(scales_base + blocks_per_cta * (QK_K / 16));
    int8_t * l = l_base + block_in_cta * QK_K;
    float * scales = scales_base + block_in_cta * (QK_K / 16);
    int8_t * qscales = qscales_base + block_in_cta * (QK_K / 16);
    const src_t * xb = x + iblock * QK_K;
    block_q3_K * yb = y + iblock;

    const float scale = make_q3_quants(16, 4, xb + 16 * subgroup, l + 16 * subgroup);
    scales[subgroup] = scale;

    __syncwarp(warp_mask);

    if (subgroup == 0)
    {
        float max_scale = 0.0f;
        float max_abs_scale = 0.0f;
        for (int j = 0; j < QK_K / 16; ++j)
        {
            const float sj = scales[j];
            const float abs_scale = fabsf(sj);
            if (abs_scale > max_abs_scale)
            {
                max_abs_scale = abs_scale;
                max_scale = sj;
            }
        }

        if (max_abs_scale != 0.0f)
        {
            const float iscale = -32.0f / max_scale;
            yb->d = compute_fp32_to_fp16(1.0f / iscale);
            memset(yb->scales, 0, sizeof(yb->scales));
            for (int j = 0; j < QK_K / 16; ++j)
            {
                int q = nearest_int(iscale * scales[j]);
                q = clamp_int(q, -32, 31) + 32;
                qscales[j] = q - 32;
                if (j < 8)
                {
                    yb->scales[j] = q & 0x0f;
                }
                else
                {
                    yb->scales[j - 8] |= (q & 0x0f) << 4;
                }
                yb->scales[j % 4 + 8] |= (q >> 4) << (2 * (j / 4));
            }
        }
        else
        {
            memset(yb->scales, 0, sizeof(yb->scales));
            yb->d = compute_fp32_to_fp16(0.0f);
            for (int j = 0; j < QK_K / 16; ++j)
            {
                qscales[j] = 0;
            }
        }
    }

    __syncwarp(warp_mask);

    const float d = __half2float(load_half(yb->d)) * qscales[subgroup];
    if (d != 0.0f)
    {
        for (int ii = 0; ii < 16; ++ii)
        {
            int qv = nearest_int(static_cast<float>(xb[16 * subgroup + ii]) / d);
            qv = clamp_int(qv, -4, 3);
            l[16 * subgroup + ii] = qv + 4;
        }
    }

    __syncwarp(warp_mask);

    for (int lane_idx = subgroup; lane_idx < QK_K / 8; lane_idx += 16)
    {
        uint8_t hm = 0;
        for (int bit = 0; bit < 8; ++bit)
        {
            hm |= (l[lane_idx + bit * (QK_K / 8)] > 3) << bit;
        }
        yb->hmask[lane_idx] = hm;
    }

    for (int j = 0; j < QK_K; j += 128)
    {
        for (int lane_idx = subgroup; lane_idx < 32; lane_idx += 16)
        {
            yb->qs[j / 4 + lane_idx] = (l[j + lane_idx] & 3) | ((l[j + lane_idx + 32] & 3) << 2) |
                                       ((l[j + lane_idx + 64] & 3) << 4) | ((l[j + lane_idx + 96] & 3) << 6);
        }
    }
}

template<typename src_t>
static __global__ void quantize_block_q3_k(
    const src_t * __restrict__ x,
    block_q3_K * __restrict__ y,
    int64_t n_blocks)
{
    extern __shared__ int8_t shared[];
    quantize_q3_k(x, y, n_blocks, shared);
}

}
}
