#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_quantize_common.cuh"
#include "cuda_quantize_kernels.h"

static __global__ void quantize_block_q2_K(const float * __restrict__ x, block_q2_K * __restrict__ y, int64_t n_blocks) {
    constexpr int blocks_per_warp = 2;
    const int warps_per_cta = blockDim.x >> 5;
    const int blocks_per_cta = warps_per_cta * blocks_per_warp;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int block_in_warp = lane >> 4;
    const int subgroup = lane & 15;
    const int block_in_cta = warp * blocks_per_warp + block_in_warp;
    const int64_t iblock = (int64_t)blockIdx.x * blocks_per_cta + block_in_cta;
    if (iblock >= n_blocks) {
        return;
    }

    extern __shared__ uint8_t q2_lane_smem[];
    uint8_t * l_base = q2_lane_smem;
    float * mins_base = (float *)(l_base + blocks_per_cta * QK_K);
    float * scales_base = mins_base + blocks_per_cta * (QK_K / 16);
    uint8_t * l = l_base + block_in_cta * QK_K;
    float * mins = mins_base + block_in_cta * (QK_K / 16);
    float * scales = scales_base + block_in_cta * (QK_K / 16);
    const float * xb = x + iblock * QK_K;
    block_q2_K * yb = y + iblock;
    uint8_t laux[16];
    float weights[16];

    const int base = 16 * subgroup;
    for (int ii = 0; ii < 16; ++ii) {
        weights[ii] = fabsf(xb[base + ii]);
    }
    scales[subgroup] = gguf_cuda_make_qkx2_quants(
        16, 3, xb + base, weights, l + base, &mins[subgroup], laux, -0.5f, 0.1f, 15, true);

    __syncthreads();

    if (subgroup == 0) {
        float max_scale = 0.0f;
        float max_min = 0.0f;
        for (int j = 0; j < QK_K / 16; ++j) {
            if (scales[j] > max_scale) {
                max_scale = scales[j];
            }
            if (mins[j] > max_min) {
                max_min = mins[j];
            }
        }

        if (max_scale > 0.0f) {
            const float iscale = __fdiv_rn(15.0f, max_scale);
            for (int j = 0; j < QK_K / 16; ++j) {
                const uint8_t qscale = gguf_cuda_nearest_int(__fmul_rn(iscale, scales[j]));
                yb->scales[j] = qscale;
            }
            yb->d = gguf_cuda_compute_fp32_to_fp16(__fdiv_rn(max_scale, 15.0f));
        } else {
            for (int j = 0; j < QK_K / 16; ++j) {
                yb->scales[j] = 0;
            }
            yb->d = gguf_cuda_compute_fp32_to_fp16(0.0f);
        }

        if (max_min > 0.0f) {
            const float iscale = __fdiv_rn(15.0f, max_min);
            for (int j = 0; j < QK_K / 16; ++j) {
                const uint8_t qmin = gguf_cuda_nearest_int(__fmul_rn(iscale, mins[j]));
                yb->scales[j] |= qmin << 4;
            }
            yb->dmin = gguf_cuda_compute_fp32_to_fp16(__fdiv_rn(max_min, 15.0f));
        } else {
            yb->dmin = gguf_cuda_compute_fp32_to_fp16(0.0f);
        }
    }

    __syncthreads();

    const float d_base = __half2float(gguf_cuda_load_half(yb->d));
    const float dm_base = __half2float(gguf_cuda_load_half(yb->dmin));
    const uint8_t qscale_min = yb->scales[subgroup];
    const float d = __fmul_rn(d_base, float(qscale_min & 0x0f));
    if (d != 0.0f) {
        const float dm = __fmul_rn(dm_base, float(qscale_min >> 4));
        for (int ii = 0; ii < 16; ++ii) {
            int q = gguf_cuda_nearest_int(__fdiv_rn(xb[base + ii] + dm, d));
            q = gguf_cuda_clamp_int(q, 0, 3);
            l[base + ii] = q;
        }
    }

    __syncthreads();

    for (int j = 0; j < QK_K; j += 128) {
        for (int ii = subgroup; ii < 32; ii += 16) {
            yb->qs[j / 4 + ii] = l[j + ii] | (l[j + ii + 32] << 2) |
                                 (l[j + ii + 64] << 4) | (l[j + ii + 96] << 6);
        }
    }
}


void gguf_cuda_quantize_launch_q2_k(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
) {
    (void)quant_weights;
    (void)n_per_row;
    constexpr int q2_k_threads = 64;
    constexpr int q2_k_blocks_per_cta = (q2_k_threads / 32) * 2;
    const int64_t n_blocks = k / QK_K;
    const int blocks = (int)((n_blocks + q2_k_blocks_per_cta - 1) / q2_k_blocks_per_cta);
    const size_t smem = q2_k_blocks_per_cta * (QK_K * sizeof(uint8_t) + 2 * (QK_K / 16) * sizeof(float));
    quantize_block_q2_K<<<blocks, q2_k_threads, smem, stream>>>(x, (block_q2_K *)y, n_blocks);
}
