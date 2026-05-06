#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_quantize_common.cuh"
#include "cuda_quantize_kernels.h"

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


void gguf_cuda_quantize_launch_iq2_xs(
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
        quantize_block_iq2_xs<<<blocks, threads, 0, stream>>>(x, quant_weights, (block_iq2_xs *)y, n_blocks, n_blocks_per_row);
}
