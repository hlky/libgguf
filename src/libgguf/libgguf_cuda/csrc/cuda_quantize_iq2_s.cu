#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGUF_CUDA_USE_IQ2_GRID_LOOKUP
#define GGUF_CUDA_USE_IQ2_NEIGHBOURS
#define GGUF_CUDA_USE_IQ2_S_NEIGHBOURS
#define GGUF_CUDA_IQ2_HOIST_BEST_NEIGHBOUR
#include "cuda_quantize_common.cuh"
#include "cuda_quantize_kernels.h"

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
                        l[8 * k + i] = gguf_cuda_iq2_grid_l_fast(iq2s_grid, grid_index, i);
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


void gguf_cuda_quantize_launch_iq2_s(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
) {
    const int threads = 256;
        const int64_t n_blocks = k / QK_K;
        const int blocks = (int)((n_blocks + threads - 1) / threads);
        const int64_t n_blocks_per_row = n_per_row / QK_K;
        quantize_block_iq2_s<<<blocks, threads, 0, stream>>>(x, quant_weights, (block_iq2_s *)y, n_blocks, n_blocks_per_row);
}
