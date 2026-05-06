#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define GGUF_CUDA_USE_IQ1_GRID_LOOKUP
#include "cuda_quantize_common.cuh"
#include "cuda_quantize_kernels.h"

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


void gguf_cuda_quantize_launch_iq1_m(
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
        quantize_block_iq1_m<<<blocks, threads, 0, stream>>>(x, quant_weights, (block_iq1_m *)y, n_blocks, n_blocks_per_row);
}
