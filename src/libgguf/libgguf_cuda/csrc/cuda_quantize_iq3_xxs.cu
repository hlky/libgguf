#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGUF_CUDA_USE_IQ3_GRID_LOOKUP
#define GGUF_CUDA_USE_IQ3_NEIGHBOURS
#include "cuda_quantize_common.cuh"
#include "cuda_quantize_kernels.h"

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
                int grid_index = gguf_cuda_iq3_find_grid_or_best_neighbour(
                    iq3xxs_grid, 256, 2, xval + 4 * k, waux + 4 * k, this_scale, laux + 4 * k,
                    is_on_grid_aux + k);
                (void)grid_index;
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
                bool on_grid = false;
                int grid_index = gguf_cuda_iq3_find_grid_or_best_neighbour(
                    iq3xxs_grid, 256, 2, xval + 4 * k, waux + 4 * k, scale, l + 4 * k, &on_grid);
                (void)grid_index;
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
        for (int i = 0; i < 3 * (QK_K / 8); ++i) {
            yb->qs[i] = 0;
        }
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


void gguf_cuda_quantize_launch_iq3_xxs(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
) {
    (void)quant_weights;
    (void)n_per_row;
    const int threads = 64;
        const int64_t n_blocks = k / QK_K;
        const int blocks = (int)((n_blocks + threads - 1) / threads);
        quantize_block_iq3_xxs<<<blocks, threads, 0, stream>>>(x, (block_iq3_xxs *)y, n_blocks);
}
