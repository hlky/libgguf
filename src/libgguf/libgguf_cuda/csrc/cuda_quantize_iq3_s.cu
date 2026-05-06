#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGUF_CUDA_USE_IQ3_GRID_LOOKUP
#include "cuda_quantize_common.cuh"
#include "cuda_quantize_kernels.h"

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


void gguf_cuda_quantize_launch_iq3_s(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
) {
    (void)quant_weights;
    (void)n_per_row;
    const int threads = 256;
        const int64_t n_blocks = k / QK_K;
        const int blocks = (int)((n_blocks + threads - 1) / threads);
        quantize_block_iq3_s<<<blocks, threads, 0, stream>>>(x, (block_iq3_s *)y, n_blocks);
}
