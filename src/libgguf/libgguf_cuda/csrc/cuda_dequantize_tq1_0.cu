#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_dequantize_common.cuh"
#include "cuda_dequantize_kernels.h"
#include "dispatch_utils.h"

template<typename dst_t>
static __global__ void dequantize_block_tq1_0(const void * __restrict__ vx, dst_t * __restrict__ y, const int k) {
    const int packed_index = blockDim.x*blockIdx.x + threadIdx.x;
    const int n_blocks = k / QK_K;
    if (packed_index >= n_blocks * (int)(sizeof(block_tq1_0::qs) + sizeof(block_tq1_0::qh))) {
        return;
    }

    const block_tq1_0 * x = (const block_tq1_0 *) vx;
    const int ib = packed_index / (int)(sizeof(block_tq1_0::qs) + sizeof(block_tq1_0::qh));
    const int j = packed_index - ib * (int)(sizeof(block_tq1_0::qs) + sizeof(block_tq1_0::qh));
    const float d = __half2float(gguf_cuda_load_half(x[ib].d));
    dst_t * yb = y + ib * QK_K;

    if (j < 32) {
        const uint8_t packed = x[ib].qs[j];
#pragma unroll
        for (int plane = 0; plane < 5; ++plane) {
            const int multiplier = plane == 0 ? 1 : plane == 1 ? 3 : plane == 2 ? 9 : plane == 3 ? 27 : 81;
            const uint8_t code = (uint8_t)(packed * multiplier);
            const int q = (((int)code * 3) >> 8) - 1;
            yb[j + plane * 32] = convert_from_float<dst_t>(d * q);
        }
    } else if (j < 48) {
        const int byte = j - 32;
        const uint8_t packed = x[ib].qs[j];
#pragma unroll
        for (int plane = 0; plane < 5; ++plane) {
            const int multiplier = plane == 0 ? 1 : plane == 1 ? 3 : plane == 2 ? 9 : plane == 3 ? 27 : 81;
            const uint8_t code = (uint8_t)(packed * multiplier);
            const int q = (((int)code * 3) >> 8) - 1;
            yb[160 + byte + plane * 16] = convert_from_float<dst_t>(d * q);
        }
    } else {
        const int byte = j - 48;
        const uint8_t packed = x[ib].qh[byte];
#pragma unroll
        for (int plane = 0; plane < 4; ++plane) {
            const int multiplier = plane == 0 ? 1 : plane == 1 ? 3 : plane == 2 ? 9 : 27;
            const uint8_t code = (uint8_t)(packed * multiplier);
            const int q = (((int)code * 3) >> 8) - 1;
            yb[240 + byte + plane * 4] = convert_from_float<dst_t>(d * q);
        }
    }
}


void gguf_cuda_dequantize_launch_tq1_0(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
) {
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "dequantize_tq1_0", [&] {
        const int n_blocks = k / QK_K;
        const int packed_per_block = (int)(sizeof(block_tq1_0::qs) + sizeof(block_tq1_0::qh));
        const int nb = (n_blocks * packed_per_block + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
                dequantize_block_tq1_0<<<nb, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(x, (scalar_t *)y, k);
    });
}
