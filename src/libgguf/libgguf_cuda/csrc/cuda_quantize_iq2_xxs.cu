#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_quantize_common.cuh"
#include "cuda_quantize_kernels.h"

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


void gguf_cuda_quantize_launch_iq2_xxs(
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
        quantize_block_iq2_xxs<<<blocks, threads, 0, stream>>>(x, quant_weights, (block_iq2_xxs *)y, n_blocks, n_blocks_per_row);
}
