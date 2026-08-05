#pragma once

#define LIBGGUF_GPU_USE_IQ1_GRID_LOOKUP
#define LIBGGUF_GPU_USE_IQ1_NEIGHBOURS
#include <libgguf/gpu/quantize/common.cuh>

namespace libgguf
{
namespace gpu
{

template<typename src_t>
static __device__ void quantize_iq1_s(
    const src_t * x,
    const float * quant_weights,
    block_iq1_s * y,
    int64_t n_blocks,
    int64_t n_blocks_per_row)
{
    const int64_t iblock = blockDim.x * blockIdx.x + threadIdx.x;
    if (iblock >= n_blocks)
    {
        return;
    }

    const src_t * xb = x + iblock * QK_K;
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

    yb->d = compute_fp32_to_fp16(0.0f);
    for (int i = 0; i < QK_K / 8; ++i)
    {
        yb->qs[i] = 0;
    }
    for (int i = 0; i < QK_K / 32; ++i)
    {
        yb->qh[i] = 0;
    }

    float max_scale = 0.0f;
    float sumx2 = 0.0f;
    for (int i = 0; i < QK_K; ++i)
    {
        const float xi = static_cast<float>(xb[i]);
        sumx2 += xi * xi;
    }
    const float sigma2 = 2.0f * sumx2 / QK_K;

    for (int ib = 0; ib < QK_K / 32; ++ib)
    {
        const src_t * x32 = xb + 32 * ib;
        for (int i = 0; i < 32; ++i)
        {
            const float xi = static_cast<float>(x32[i]);
            weight[i] = qw_block[32 * ib + i] * sqrtf(sigma2 + xi * xi);
            idx[i] = i;
            l[i] = 1;
        }
        float maxv = fabsf(static_cast<float>(x32[0]));
        for (int i = 1; i < 32; ++i)
        {
            maxv = fmaxf(maxv, fabsf(static_cast<float>(x32[i])));
        }
        if (maxv < GROUP_MAX_EPS_IQ1_S)
        {
            scales[ib] = 0.0f;
            shifts[ib] = 1;
            continue;
        }

        for (int i = 1; i < 32; ++i)
        {
            const int value = idx[i];
            int j = i - 1;
            while (j >= 0 && static_cast<float>(x32[idx[j]]) > static_cast<float>(x32[value]))
            {
                idx[j + 1] = idx[j];
                --j;
            }
            idx[j + 1] = value;
        }

        sumx[0] = 0.0f;
        sumw[0] = 0.0f;
        for (int j = 0; j < 32; ++j)
        {
            const int i = idx[j];
            sumx[j + 1] = sumx[j] + weight[i] * static_cast<float>(x32[i]);
            sumw[j + 1] = sumw[j] + weight[i];
        }

        float best_score = -FLT_MAX;
        float scale = maxv;
        int best_i1 = -1;
        int best_i2 = -1;
        int best_shift = 0;
        for (int i1 = 0; i1 <= 32; ++i1)
        {
            for (int i2 = i1; i2 <= 32; ++i2)
            {
                const float w0 = sumw[i1] - sumw[0];
                const float w1 = sumw[i2] - sumw[i1];
                const float w2 = sumw[32] - sumw[i2];
                const float sx0 = sumx[i1] - sumx[0];
                const float sx1 = sumx[i2] - sumx[i1];
                const float sx2 = sumx[32] - sumx[i2];
                float sumqx = sx0 * x_p[0] + sx1 * x_p[1] + sx2 * x_p[2];
                float sumq2 = w0 * x_p[0] * x_p[0] + w1 * x_p[1] * x_p[1] + w2 * x_p[2] * x_p[2];
                if (sumq2 > 0.0f && sumqx * sumqx > best_score * sumq2)
                {
                    scale = sumqx / sumq2;
                    best_score = scale * sumqx;
                    best_i1 = i1;
                    best_i2 = i2;
                    best_shift = 1;
                }
                sumqx = sx0 * x_m[0] + sx1 * x_m[1] + sx2 * x_m[2];
                sumq2 = w0 * x_m[0] * x_m[0] + w1 * x_m[1] * x_m[1] + w2 * x_m[2] * x_m[2];
                if (sumq2 > 0.0f && sumqx * sumqx > best_score * sumq2)
                {
                    scale = sumqx / sumq2;
                    best_score = scale * sumqx;
                    best_i1 = i1;
                    best_i2 = i2;
                    best_shift = -1;
                }
            }
        }
        if (best_i1 < 0 || best_i2 < 0 || best_shift == 0)
        {
            scales[ib] = 0.0f;
            shifts[ib] = 1;
            continue;
        }
        for (int j = 0; j < best_i1; ++j)
        {
            l[idx[j]] = 0;
        }
        for (int j = best_i1; j < best_i2; ++j)
        {
            l[idx[j]] = 1;
        }
        for (int j = best_i2; j < 32; ++j)
        {
            l[idx[j]] = 2;
        }
        if (scale < 0.0f)
        {
            for (int j = 0; j < 32; ++j)
            {
                l[j] = 2 - l[j];
            }
            scale = -scale;
            best_shift = -best_shift;
        }

        bool all_on_grid = true;
        const float * values = best_shift == 1 ? x_p : x_m;
        for (int k = 0; k < 4; ++k)
        {
            bool on_grid = false;
            const int grid_index = iq1_find_grid_or_best_neighbour(
                x32 + 8 * k, weight + 8 * k, scale, values, l + 8 * k, &on_grid);
            all_on_grid = all_on_grid && on_grid;
            index[k] = grid_index;
        }
        if (!all_on_grid)
        {
            float sumqx = 0.0f;
            float sumq2 = 0.0f;
            for (int k = 0; k < 4; ++k)
            {
                for (int j = 0; j < 8; ++j)
                {
                    const float w = weight[8 * k + j];
                    const float q = values[iq1_grid_l(index[k], j)];
                    sumqx += w * q * static_cast<float>(x32[8 * k + j]);
                    sumq2 += w * q * q;
                }
            }
            if (sumqx > 0.0f && sumq2 > 0.0f)
            {
                scale = sumqx / sumq2;
            }
        }

        uint16_t h = 0;
        for (int k = 0; k < 4; ++k)
        {
            yb->qs[4 * ib + k] = index[k] & 255;
            h |= (index[k] >> 8) << (3 * k);
        }
        yb->qh[ib] = h;
        scales[ib] = scale;
        shifts[ib] = best_shift;
        max_scale = fmaxf(max_scale, scale);
    }

    if (max_scale == 0.0f)
    {
        return;
    }

    const float d = max_scale / 15.0f;
    yb->d = compute_fp32_to_fp16(d * 1.125f);
    const float id = 1.0f / d;
    for (int ib = 0; ib < QK_K / 32; ++ib)
    {
        int q = nearest_int(0.5f * (id * scales[ib] - 1.0f));
        q = clamp_int(q, 0, 7);
        if (shifts[ib] == -1)
        {
            q |= 8;
        }
        yb->qh[ib] |= q << 12;
    }
}

template<typename src_t>
static __global__ void quantize_block_iq1_s(
    const src_t * __restrict__ x,
    const float * __restrict__ quant_weights,
    block_iq1_s * __restrict__ y,
    int64_t n_blocks,
    int64_t n_blocks_per_row)
{
    quantize_iq1_s(x, quant_weights, y, n_blocks, n_blocks_per_row);
}

}
}
