#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_quantize_common.cuh"
#include "cuda_quantize_kernels.h"

static __global__ void quantize_block_q2_K(const float * __restrict__ x, block_q2_K * __restrict__ y, int64_t n_blocks) {
    const int64_t iblock = blockDim.x * blockIdx.x + threadIdx.x;
    if (iblock >= n_blocks) {
        return;
    }

    const float * xb = x + iblock * QK_K;
    block_q2_K * yb = y + iblock;
    uint8_t l[QK_K];
    uint8_t laux[16];
    float weights[16];
    float mins[QK_K / 16];
    float scales[QK_K / 16];
    uint8_t quant_scales[QK_K / 16];
    uint8_t quant_mins[QK_K / 16];

    float max_scale = 0.0f;
    float max_min = 0.0f;
    for (int j = 0; j < QK_K / 16; ++j) {
        for (int ii = 0; ii < 16; ++ii) {
            weights[ii] = fabsf(xb[16 * j + ii]);
        }
        scales[j] = gguf_cuda_make_qkx2_quants(16, 3, xb + 16 * j, weights, l + 16 * j, &mins[j], laux, -0.5f, 0.1f, 15, true);
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
            quant_scales[j] = qscale;
            yb->scales[j] = qscale;
        }
        yb->d = gguf_cuda_compute_fp32_to_fp16(__fdiv_rn(max_scale, 15.0f));
    } else {
        for (int j = 0; j < QK_K / 16; ++j) {
            quant_scales[j] = 0;
            yb->scales[j] = 0;
        }
        yb->d = gguf_cuda_compute_fp32_to_fp16(0.0f);
    }

    if (max_min > 0.0f) {
        const float iscale = __fdiv_rn(15.0f, max_min);
        for (int j = 0; j < QK_K / 16; ++j) {
            const uint8_t qmin = gguf_cuda_nearest_int(__fmul_rn(iscale, mins[j]));
            quant_mins[j] = qmin;
            yb->scales[j] |= qmin << 4;
        }
        yb->dmin = gguf_cuda_compute_fp32_to_fp16(__fdiv_rn(max_min, 15.0f));
    } else {
        for (int j = 0; j < QK_K / 16; ++j) {
            quant_mins[j] = 0;
        }
        yb->dmin = gguf_cuda_compute_fp32_to_fp16(0.0f);
    }

    const float d_base = __half2float(gguf_cuda_load_half(yb->d));
    const float dm_base = __half2float(gguf_cuda_load_half(yb->dmin));
    for (int j = 0; j < QK_K / 16; ++j) {
        const float d = __fmul_rn(d_base, float(quant_scales[j]));
        if (d == 0.0f) {
            continue;
        }
        const float dm = __fmul_rn(dm_base, float(quant_mins[j]));
        for (int ii = 0; ii < 16; ++ii) {
            int q = gguf_cuda_nearest_int(__fdiv_rn(xb[16 * j + ii] + dm, d));
            q = gguf_cuda_clamp_int(q, 0, 3);
            l[16 * j + ii] = q;
        }
    }

    for (int j = 0; j < QK_K; j += 128) {
        for (int ii = 0; ii < 32; ++ii) {
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
    const int q2_k_threads = 128;
        const int64_t n_blocks = k / QK_K;
        const int blocks = (int)((n_blocks + q2_k_threads - 1) / q2_k_threads);
        quantize_block_q2_K<<<blocks, q2_k_threads, 0, stream>>>(x, (block_q2_K *)y, n_blocks);
}
