#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_quantize_common.cuh"
#include "cuda_quantize_kernels.h"

static __global__ void quantize_block_q3_K_warp(const float * __restrict__ x, block_q3_K * __restrict__ y, int64_t n_blocks) {
    constexpr unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t iblock = (int64_t)blockIdx.x * warps_per_block + warp;
    if (iblock >= n_blocks) {
        return;
    }

    extern __shared__ int8_t q3_smem[];
    int8_t * l = q3_smem + warp * QK_K;
    float * scales = (float *)(q3_smem + warps_per_block * QK_K) + warp * (QK_K / 16);
    const float * xb = x + iblock * QK_K;
    block_q3_K * yb = y + iblock;

    float scale = 0.0f;
    if (lane < QK_K / 16) {
        scale = gguf_cuda_make_q3_quants(16, 4, xb + 16 * lane, l + 16 * lane);
        scales[lane] = scale;
    }

    const float max_scale = gguf_cuda_warp_reduce_absmax_first(scale, lane);
    __syncwarp(mask);
    ggml_half d_h = gguf_cuda_compute_fp32_to_fp16(0.0f);
    if (max_scale != 0.0f) {
        const float iscale = -32.0f / max_scale;
        d_h = gguf_cuda_compute_fp32_to_fp16(1.0f / iscale);
        if (lane == 0) {
            memset(yb->scales, 0, sizeof(yb->scales));
            for (int j = 0; j < QK_K / 16; ++j) {
                int q = gguf_cuda_nearest_int(iscale * scales[j]);
                q = gguf_cuda_clamp_int(q, -32, 31) + 32;
                if (j < 8) {
                    yb->scales[j] = q & 0x0f;
                } else {
                    yb->scales[j - 8] |= (q & 0x0f) << 4;
                }
                yb->scales[j % 4 + 8] |= (q >> 4) << (2 * (j / 4));
            }
        }
        if (lane < QK_K / 16) {
            int q = gguf_cuda_nearest_int(iscale * scale);
            q = gguf_cuda_clamp_int(q, -32, 31) + 32;
            const int8_t qscale = q - 32;

            const float d = __half2float(gguf_cuda_load_half(d_h)) * qscale;
            if (d != 0.0f) {
                for (int ii = 0; ii < 16; ++ii) {
                    int qv = gguf_cuda_nearest_int(xb[16 * lane + ii] / d);
                    qv = gguf_cuda_clamp_int(qv, -4, 3);
                    l[16 * lane + ii] = qv + 4;
                }
            }
        }
    } else if (lane == 0) {
        memset(yb->scales, 0, sizeof(yb->scales));
    }
    if (lane == 0) {
        yb->d = d_h;
    }

    __syncwarp(mask);

    if (lane < QK_K / 8) {
        uint8_t hm = 0;
        for (int bit = 0; bit < 8; ++bit) {
            hm |= (l[lane + bit * (QK_K / 8)] > 3) << bit;
        }
        yb->hmask[lane] = hm;
    }

    for (int j = 0; j < QK_K; j += 128) {
        yb->qs[j / 4 + lane] = (l[j + lane] & 3) | ((l[j + lane + 32] & 3) << 2) |
                                ((l[j + lane + 64] & 3) << 4) | ((l[j + lane + 96] & 3) << 6);
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
    constexpr int q3_k_warp_threads = 128;
        constexpr int q3_k_warps_per_cta = q3_k_warp_threads / 32;
        const int64_t n_blocks = k / QK_K;
        const int blocks = (int)((n_blocks + q3_k_warps_per_cta - 1) / q3_k_warps_per_cta);
        const size_t smem = q3_k_warps_per_cta * (QK_K * sizeof(int8_t) + (QK_K / 16) * sizeof(float));
        quantize_block_q3_K_warp<<<blocks, q3_k_warp_threads, smem, stream>>>(x, (block_q3_K *)y, n_blocks);
}
