#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_quantize_common.cuh"
#include "cuda_quantize_kernels.h"

static __global__ void quantize_block_q4_K(const float * __restrict__ x, block_q4_K * __restrict__ y, int64_t n_blocks) {
    constexpr int blocks_per_warp = 4;
    const int warps_per_cta = blockDim.x >> 5;
    const int blocks_per_cta = warps_per_cta * blocks_per_warp;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int block_in_warp = lane >> 3;
    const int subgroup = lane & 7;
    const int block_in_cta = warp * blocks_per_warp + block_in_warp;
    const int64_t iblock = (int64_t)blockIdx.x * blocks_per_cta + block_in_cta;
    if (iblock >= n_blocks) {
        return;
    }

    extern __shared__ uint8_t q4_lane_smem[];
    uint8_t * l_base = q4_lane_smem;
    float * mins_base = (float *)(l_base + blocks_per_cta * QK_K);
    float * scales_base = mins_base + blocks_per_cta * (QK_K / 32);
    uint8_t * l = l_base + block_in_cta * QK_K;
    float * mins = mins_base + block_in_cta * (QK_K / 32);
    float * scales = scales_base + block_in_cta * (QK_K / 32);
    const float * xb = x + iblock * QK_K;
    block_q4_K * yb = y + iblock;

    uint8_t laux[32];
    float weights[32];
    const int base = 32 * subgroup;
    float sum_x2 = 0.0f;
    for (int ii = 0; ii < 32; ++ii) {
        const float v = xb[base + ii];
        sum_x2 += v * v;
    }
    const float av_x = sqrtf(sum_x2 / 32.0f);
    for (int ii = 0; ii < 32; ++ii) {
        weights[ii] = av_x + fabsf(xb[base + ii]);
    }
    scales[subgroup] = gguf_cuda_make_qkx2_quants(
        32, 15, xb + base, weights, l + base, &mins[subgroup], laux, -1.0f, 0.1f, 20, false);

    __syncthreads();

    if (subgroup == 0) {
        float max_scale = 0.0f;
        float max_min = 0.0f;
        for (int j = 0; j < QK_K / 32; ++j) {
            max_scale = fmaxf(max_scale, scales[j]);
            max_min = fmaxf(max_min, mins[j]);
        }
        const float inv_scale = max_scale > 0.0f ? 63.0f / max_scale : 0.0f;
        const float inv_min = max_min > 0.0f ? 63.0f / max_min : 0.0f;
        for (int j = 0; j < QK_K / 32; ++j) {
            uint8_t ls = min(63, gguf_cuda_nearest_int(inv_scale * scales[j]));
            uint8_t lm = min(63, gguf_cuda_nearest_int(inv_min * mins[j]));
            if (j < 4) {
                yb->scales[j] = ls;
                yb->scales[j + 4] = lm;
            } else {
                yb->scales[j + 4] = (ls & 0x0f) | ((lm & 0x0f) << 4);
                yb->scales[j - 4] |= (ls >> 4) << 6;
                yb->scales[j] |= (lm >> 4) << 6;
            }
        }
        yb->d = gguf_cuda_compute_fp32_to_fp16(max_scale / 63.0f);
        yb->dmin = gguf_cuda_compute_fp32_to_fp16(max_min / 63.0f);
    }

    __syncthreads();

    const float d_base = __half2float(gguf_cuda_load_half(yb->d));
    const float dm_base = __half2float(gguf_cuda_load_half(yb->dmin));
    uint8_t sc;
    uint8_t m;
    gguf_cuda_get_scale_min_k4(subgroup, yb->scales, &sc, &m);
    const float d = d_base * sc;
    if (d != 0.0f) {
        const float dm = dm_base * m;
        for (int ii = 0; ii < 32; ++ii) {
            int q = gguf_cuda_nearest_int((xb[base + ii] + dm) / d);
            q = gguf_cuda_clamp_int(q, 0, 15);
            l[base + ii] = q;
        }
    }

    __syncthreads();

    const int pack_pair = subgroup >> 1;
    const int pack_offset = (subgroup & 1) * 16;
    const int l_pack_base = 64 * pack_pair + pack_offset;
    for (int ii = 0; ii < 16; ++ii) {
        yb->qs[32 * pack_pair + pack_offset + ii] = l[l_pack_base + ii] | (l[l_pack_base + 32 + ii] << 4);
    }
}


void gguf_cuda_quantize_launch_q4_k(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
) {
    (void)quant_weights;
    (void)n_per_row;
    constexpr int q4_k_threads = 64;
    constexpr int q4_k_blocks_per_cta = (q4_k_threads / 32) * 4;
    const int64_t n_blocks = k / QK_K;
    const int blocks = (int)((n_blocks + q4_k_blocks_per_cta - 1) / q4_k_blocks_per_cta);
    const size_t smem = q4_k_blocks_per_cta * (QK_K * sizeof(uint8_t) + 2 * (QK_K / 32) * sizeof(float));
    quantize_block_q4_K<<<blocks, q4_k_threads, smem, stream>>>(x, (block_q4_K *)y, n_blocks);
}
