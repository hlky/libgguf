#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_quantize_common.cuh"
#include "cuda_quantize_kernels.h"

static __global__ void quantize_block_q6_K(const float * __restrict__ x, block_q6_K * __restrict__ y, int64_t n_blocks) {
    const int64_t iblock = blockDim.x * blockIdx.x + threadIdx.x;
    if (iblock >= n_blocks) {
        return;
    }

    const float * xb = x + iblock * QK_K;
    block_q6_K * yb = y + iblock;
    int8_t l[QK_K];
    float scales[QK_K / 16];
    int8_t quant_scales[QK_K / 16];

    float max_scale = 0.0f;
    float max_abs_scale = 0.0f;
    for (int ib = 0; ib < QK_K / 16; ++ib) {
        const float scale = gguf_cuda_make_qx_quants_rmse1(16, 32, xb + 16 * ib, l + 16 * ib);
        scales[ib] = scale;
        const float abs_scale = fabsf(scale);
        if (abs_scale > max_abs_scale) {
            max_abs_scale = abs_scale;
            max_scale = scale;
        }
    }

    if (max_abs_scale < GROUP_MAX_EPS) {
        memset(yb, 0, sizeof(block_q6_K));
        yb->d = gguf_cuda_compute_fp32_to_fp16(0.0f);
        return;
    }

    const float iscale = __fdiv_rn(-128.0f, max_scale);
    yb->d = gguf_cuda_compute_fp32_to_fp16(__fdiv_rn(1.0f, iscale));
    const float d_super = __half2float(gguf_cuda_load_half(yb->d));
    for (int ib = 0; ib < QK_K / 16; ++ib) {
        const int8_t qscale = min(127, gguf_cuda_nearest_int(__fmul_rn(iscale, scales[ib])));
        quant_scales[ib] = qscale;
        yb->scales[ib] = qscale;
    }

    for (int j = 0; j < QK_K / 16; ++j) {
        const float d = d_super * quant_scales[j];
        if (d == 0.0f) {
            continue;
        }
        for (int ii = 0; ii < 16; ++ii) {
            int q = gguf_cuda_nearest_int(__fdiv_rn(xb[16 * j + ii], d));
            q = max(-32, min(31, q));
            l[16 * j + ii] = q + 32;
        }
    }

    uint8_t * ql = yb->ql;
    uint8_t * qh = yb->qh;
    for (int j = 0; j < QK_K; j += 128) {
        for (int i = 0; i < 32; ++i) {
            const uint8_t q1 = l[j + i + 0] & 0x0f;
            const uint8_t q2 = l[j + i + 32] & 0x0f;
            const uint8_t q3 = l[j + i + 64] & 0x0f;
            const uint8_t q4 = l[j + i + 96] & 0x0f;
            ql[i + 0] = q1 | (q3 << 4);
            ql[i + 32] = q2 | (q4 << 4);
            qh[i] = (l[j + i] >> 4) | ((l[j + i + 32] >> 4) << 2) |
                    ((l[j + i + 64] >> 4) << 4) | ((l[j + i + 96] >> 4) << 6);
        }
        ql += 64;
        qh += 32;
    }
}


void gguf_cuda_quantize_launch_q6_k(
    const float * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    cudaStream_t stream
) {
    (void)quant_weights;
    (void)n_per_row;
    const int q6_k_threads = 96;
        const int64_t n_blocks = k / QK_K;
        const int blocks = (int)((n_blocks + q6_k_threads - 1) / q6_k_threads);
        quantize_block_q6_K<<<blocks, q6_k_threads, 0, stream>>>(x, (block_q6_K *)y, n_blocks);
}
