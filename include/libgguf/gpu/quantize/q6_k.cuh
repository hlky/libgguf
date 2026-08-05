#pragma once

#include <libgguf/gpu/quantize/common.cuh>

namespace libgguf
{
namespace gpu
{

template<typename src_t>
static __device__ void quantize_q6_k(
    const src_t * x,
    block_q6_K * y,
    int64_t n_blocks,
    int8_t * shared)
{
    constexpr int blocks_per_warp = 2;
    const int warps_per_cta = blockDim.x >> 5;
    const int blocks_per_cta = warps_per_cta * blocks_per_warp;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int block_in_warp = lane >> 4;
    const int subgroup = lane & 15;
    const int block_in_cta = warp * blocks_per_warp + block_in_warp;
    const int64_t iblock = static_cast<int64_t>(blockIdx.x) * blocks_per_cta + block_in_cta;
    if (iblock >= n_blocks)
    {
        return;
    }

    int8_t * l_base = shared;
    float * scales_base = reinterpret_cast<float *>(l_base + blocks_per_cta * QK_K);
    int8_t * l = l_base + block_in_cta * QK_K;
    float * scales = scales_base + block_in_cta * (QK_K / 16);
    const src_t * xb = x + iblock * QK_K;
    block_q6_K * yb = y + iblock;

    scales[subgroup] = make_qx_quants_rmse1(16, 32, xb + 16 * subgroup, l + 16 * subgroup);

    __syncthreads();

    if (subgroup == 0)
    {
        float max_scale = 0.0f;
        float max_abs_scale = 0.0f;
        for (int ib = 0; ib < QK_K / 16; ++ib)
        {
            const float scale = scales[ib];
            const float abs_scale = fabsf(scale);
            if (abs_scale > max_abs_scale)
            {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }
        }

        if (max_abs_scale < GROUP_MAX_EPS)
        {
            memset(yb, 0, sizeof(block_q6_K));
            yb->d = compute_fp32_to_fp16(0.0f);
        }
        else
        {
            const float iscale = __fdiv_rn(-128.0f, max_scale);
            yb->d = compute_fp32_to_fp16(__fdiv_rn(1.0f, iscale));
            for (int ib = 0; ib < QK_K / 16; ++ib)
            {
                const int8_t qscale = min(127, nearest_int(__fmul_rn(iscale, scales[ib])));
                yb->scales[ib] = qscale;
            }
        }
    }

    __syncthreads();

    const float d_super = __half2float(load_half(yb->d));
    const float d = d_super * yb->scales[subgroup];
    if (d != 0.0f)
    {
        for (int ii = 0; ii < 16; ++ii)
        {
            int q = nearest_int(__fdiv_rn(static_cast<float>(xb[16 * subgroup + ii]), d));
            q = max(-32, min(31, q));
            l[16 * subgroup + ii] = q + 32;
        }
    }

    __syncthreads();

    for (int j = 0; j < QK_K; j += 128)
    {
        const int pack_block = j / 128;
        for (int i = subgroup; i < 32; i += 16)
        {
            const uint8_t q1 = l[j + i] & 0x0f;
            const uint8_t q2 = l[j + i + 32] & 0x0f;
            const uint8_t q3 = l[j + i + 64] & 0x0f;
            const uint8_t q4 = l[j + i + 96] & 0x0f;
            yb->ql[64 * pack_block + i] = q1 | (q3 << 4);
            yb->ql[64 * pack_block + i + 32] = q2 | (q4 << 4);
            yb->qh[32 * pack_block + i] = (l[j + i] >> 4) | ((l[j + i + 32] >> 4) << 2) |
                                          ((l[j + i + 64] >> 4) << 4) | ((l[j + i + 96] >> 4) << 6);
        }
    }
}

template<typename src_t>
static __global__ void quantize_block_q6_k(
    const src_t * __restrict__ x,
    block_q6_K * __restrict__ y,
    int64_t n_blocks)
{
    extern __shared__ int8_t shared[];
    quantize_q6_k(x, y, n_blocks, shared);
}

}
}
