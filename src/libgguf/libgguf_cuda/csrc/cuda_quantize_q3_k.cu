#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_quantize_common.cuh"
#include "cuda_quantize_kernels.h"

static __global__ void quantize_block_q3_K(const float * __restrict__ x, block_q3_K * __restrict__ y, int64_t n_blocks) {
    constexpr int blocks_per_warp = 2;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_cta = blockDim.x >> 5;
    const int blocks_per_cta = warps_per_cta * blocks_per_warp;
    const int block_in_warp = lane >> 4;
    const int subgroup = lane & 15;
    const int block_in_cta = warp * blocks_per_warp + block_in_warp;
    const int64_t iblock = (int64_t)blockIdx.x * blocks_per_cta + block_in_cta;
    if (iblock >= n_blocks) {
        return;
    }

    extern __shared__ int8_t q3_smem[];
    int8_t * l_base = q3_smem;
    float * scales_base = (float *)(l_base + blocks_per_cta * QK_K);
    int8_t * qscales_base = (int8_t *)(scales_base + blocks_per_cta * (QK_K / 16));
    int8_t * l = l_base + block_in_cta * QK_K;
    float * scales = scales_base + block_in_cta * (QK_K / 16);
    int8_t * qscales = qscales_base + block_in_cta * (QK_K / 16);
    const float * xb = x + iblock * QK_K;
    block_q3_K * yb = y + iblock;

    const float scale = gguf_cuda_make_q3_quants(16, 4, xb + 16 * subgroup, l + 16 * subgroup);
    scales[subgroup] = scale;

    __syncthreads();

    if (subgroup == 0) {
        float max_scale = 0.0f;
        float max_abs_scale = 0.0f;
        for (int j = 0; j < QK_K / 16; ++j) {
            const float sj = scales[j];
            const float abs_scale = fabsf(sj);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = sj;
            }
        }

        if (max_abs_scale != 0.0f) {
            const float iscale = -32.0f / max_scale;
            yb->d = gguf_cuda_compute_fp32_to_fp16(1.0f / iscale);
            memset(yb->scales, 0, sizeof(yb->scales));
            for (int j = 0; j < QK_K / 16; ++j) {
                int q = gguf_cuda_nearest_int(iscale * scales[j]);
                q = gguf_cuda_clamp_int(q, -32, 31) + 32;
                qscales[j] = q - 32;
                if (j < 8) {
                    yb->scales[j] = q & 0x0f;
                } else {
                    yb->scales[j - 8] |= (q & 0x0f) << 4;
                }
                yb->scales[j % 4 + 8] |= (q >> 4) << (2 * (j / 4));
            }
        } else {
            memset(yb->scales, 0, sizeof(yb->scales));
            yb->d = gguf_cuda_compute_fp32_to_fp16(0.0f);
            for (int j = 0; j < QK_K / 16; ++j) {
                qscales[j] = 0;
            }
        }
    }

    __syncthreads();

    const float d = __half2float(gguf_cuda_load_half(yb->d)) * qscales[subgroup];
    if (d != 0.0f) {
        for (int ii = 0; ii < 16; ++ii) {
            int qv = gguf_cuda_nearest_int(xb[16 * subgroup + ii] / d);
            qv = gguf_cuda_clamp_int(qv, -4, 3);
            l[16 * subgroup + ii] = qv + 4;
        }
    }

    __syncthreads();

    for (int lane_idx = subgroup; lane_idx < QK_K / 8; lane_idx += 16) {
        uint8_t hm = 0;
        for (int bit = 0; bit < 8; ++bit) {
            hm |= (l[lane_idx + bit * (QK_K / 8)] > 3) << bit;
        }
        yb->hmask[lane_idx] = hm;
    }

    for (int j = 0; j < QK_K; j += 128) {
        for (int lane_idx = subgroup; lane_idx < 32; lane_idx += 16) {
            yb->qs[j / 4 + lane_idx] = (l[j + lane_idx] & 3) | ((l[j + lane_idx + 32] & 3) << 2) |
                                       ((l[j + lane_idx + 64] & 3) << 4) | ((l[j + lane_idx + 96] & 3) << 6);
        }
    }
}


void gguf_cuda_quantize_launch_q3_k(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
) {
    (void)quant_weights;
    (void)n_per_row;
    constexpr int q3_k_threads = 64;
    constexpr int q3_k_blocks_per_cta = (q3_k_threads / 32) * 2;
    const int64_t n_blocks = k / QK_K;
    const int blocks = (int)((n_blocks + q3_k_blocks_per_cta - 1) / q3_k_blocks_per_cta);
    const size_t smem = q3_k_blocks_per_cta * (QK_K * sizeof(int8_t) + (QK_K / 16) * (sizeof(float) + sizeof(int8_t)));
    quantize_block_q3_K<<<blocks, q3_k_threads, smem, stream>>>(x, (block_q3_K *)y, n_blocks);
}
