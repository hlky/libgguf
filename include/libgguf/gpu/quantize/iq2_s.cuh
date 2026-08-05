#pragma once

#define LIBGGUF_GPU_USE_IQ2_GRID_LOOKUP
#define LIBGGUF_GPU_USE_IQ2_NEIGHBOURS
#define LIBGGUF_GPU_USE_IQ2_S_NEIGHBOURS
#define LIBGGUF_GPU_IQ2_HOIST_BEST_NEIGHBOUR
#include <libgguf/gpu/quantize/common.cuh>

namespace libgguf
{
namespace gpu
{

template<typename src_t>
static __device__ void quantize_iq2_s(
    const src_t * x,
    const float * quant_weights,
    block_iq2_s * y,
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
    yb->d = compute_fp32_to_fp16(0.0f);

    float max_scale = 0.0f;
    float sumx2 = 0.0f;
    for (int i = 0; i < QK_K; ++i)
    {
        const float xi = static_cast<float>(xb[i]);
        sumx2 += xi * xi;
    }
    const float sigma2 = 2.0f * sumx2 / QK_K;

    for (int ib = 0; ib < QK_K / 16; ++ib)
    {
        const src_t * x16 = xb + 16 * ib;
        for (int i = 0; i < 16; ++i)
        {
            const float xi = static_cast<float>(x16[i]);
            weight[i] = qw_block ? qw_block[16 * ib + i] * sqrtf(sigma2 + xi * xi)
                                 : 0.25f * sigma2 + xi * xi;
            waux[i] = sqrtf(weight[i]);
            l[i] = 0;
        }

        for (int k = 0; k < 2; ++k)
        {
            uint8_t signs = 0;
            for (int i = 0; i < 8; ++i)
            {
                const float value = static_cast<float>(x16[8 * k + i]);
                if (value >= 0.0f)
                {
                    xval[8 * k + i] = value;
                }
                else
                {
                    xval[8 * k + i] = -value;
                    signs |= 1 << i;
                }
            }
            block_signs[k] = signs;
        }

        float maxv = xval[0];
        for (int i = 1; i < 16; ++i)
        {
            maxv = fmaxf(maxv, xval[i]);
        }
        if (maxv < GROUP_MAX_EPS_IQ2_S)
        {
            scales[ib] = 0.0f;
            continue;
        }

        float best = 0.0f;
        float scale = maxv / 5.0f;
        is_on_grid[0] = true;
        is_on_grid[1] = true;
        for (int is = -9; is <= 9; ++is)
        {
            const float id = (5.0f + is * 0.1f) / maxv;
            const float this_scale = 1.0f / id;
            is_on_grid_aux[0] = true;
            is_on_grid_aux[1] = true;
            for (int k = 0; k < 2; ++k)
            {
                for (int i = 0; i < 8; ++i)
                {
                    const int q = nearest_int(0.5f * (id * xval[8 * k + i] - 1.0f));
                    laux[8 * k + i] = clamp_int(q, 0, 2);
                }
                const int grid_index = iq2_find_grid_or_best_neighbour(
                    iq2s_grid, 1024, 1, xval + 8 * k, waux + 8 * k,
                    this_scale, laux + 8 * k, is_on_grid_aux + k);
                (void)grid_index;
            }

            float sumqx = 0.0f;
            float sumq2 = 0.0f;
            for (int i = 0; i < 16; ++i)
            {
                const float w = weight[i];
                const float q = 2.0f * laux[i] + 1.0f;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0.0f && sumqx * sumqx > best * sumq2)
            {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                for (int i = 0; i < 16; ++i)
                {
                    l[i] = laux[i];
                }
                is_on_grid[0] = is_on_grid_aux[0];
                is_on_grid[1] = is_on_grid_aux[1];
            }
        }

        if ((!is_on_grid[0] || !is_on_grid[1]) && scale > 0.0f)
        {
            const float id = 1.0f / scale;
            for (int k = 0; k < 2; ++k)
            {
                if (is_on_grid[k])
                {
                    continue;
                }
                for (int i = 0; i < 8; ++i)
                {
                    const int q = nearest_int(0.5f * (id * xval[8 * k + i] - 1.0f));
                    l[8 * k + i] = clamp_int(q, 0, 2);
                }
                bool on_grid = false;
                const int grid_index = iq2_find_grid_or_best_neighbour(
                    iq2s_grid, 1024, 1, xval + 8 * k, waux + 8 * k, scale, l + 8 * k, &on_grid);
                (void)grid_index;
            }
            float sumqx = 0.0f;
            float sumq2 = 0.0f;
            for (int i = 0; i < 16; ++i)
            {
                const float w = weight[i];
                const float q = 2.0f * l[i] + 1.0f;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0.0f)
            {
                scale = sumqx / sumq2;
            }
        }

        if (scale < 0.0f)
        {
            scale = -scale;
            for (int k = 0; k < 2; ++k)
            {
                block_signs[k] = ~block_signs[k];
            }
        }
        for (int k = 0; k < 2; ++k)
        {
            const int grid_index = iq2_find_grid_index(iq2s_grid, 1024, l + 8 * k);
            const int i8 = 2 * ib + k;
            yb->qs[i8] = grid_index & 255;
            yb->qh[i8 / 4] |= (grid_index >> 8) << (2 * (i8 % 4));
            yb->qs[QK_K / 8 + i8] = block_signs[k];
        }
        scales[ib] = scale;
        max_scale = fmaxf(max_scale, scale);
    }

    if (max_scale == 0.0f)
    {
        return;
    }

    const float d = max_scale / 31.0f;
    yb->d = compute_fp32_to_fp16(d * 0.9875f);
    const float id = 1.0f / d;
    for (int ib = 0; ib < QK_K / 16; ++ib)
    {
        int q = nearest_int(0.5f * (id * scales[ib] - 1.0f));
        q = clamp_int(q, 0, 15);
        if (ib % 2 == 0)
        {
            yb->scales[ib / 2] = q;
        }
        else
        {
            yb->scales[ib / 2] |= q << 4;
        }
    }
}

template<typename src_t>
static __global__ void quantize_block_iq2_s(
    const src_t * __restrict__ x,
    const float * __restrict__ quant_weights,
    block_iq2_s * __restrict__ y,
    int64_t n_blocks,
    int64_t n_blocks_per_row)
{
    quantize_iq2_s(x, quant_weights, y, n_blocks, n_blocks_per_row);
}

}
}
