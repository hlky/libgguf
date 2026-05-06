#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_dequantize_common.cuh"
#include "cuda_dequantize_kernels.h"
#include "dispatch_utils.h"

template<typename dst_t>
static __global__ void dequantize_block_tq2_0(const void * __restrict__ vx, dst_t * __restrict__ y, const int k) {
    const int packed_index = blockDim.x*blockIdx.x + threadIdx.x;
    const int n_blocks = k / QK_K;
    if (packed_index >= n_blocks * (int)sizeof(block_tq2_0::qs)) {
        return;
    }

    const block_tq2_0 * x = (const block_tq2_0 *) vx;
    const int ib = packed_index / (int)sizeof(block_tq2_0::qs);
    const int byte = packed_index - ib * (int)sizeof(block_tq2_0::qs);
    const int group = byte / 32;
    const int local = byte - group * 32;
    const uint8_t packed = x[ib].qs[byte];
    const float d = __half2float(gguf_cuda_load_half(x[ib].d));
    dst_t * yb = y + ib * QK_K + group * 128 + local;
#pragma unroll
    for (int plane = 0; plane < 4; ++plane) {
        const int q = ((packed >> (2 * plane)) & 3) - 1;
        yb[plane * 32] = convert_from_float<dst_t>(d * q);
    }
}


void gguf_cuda_dequantize_launch_tq2_0(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
) {
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "dequantize_tq2_0", [&] {
        const int n_blocks = k / QK_K;
        const int nb = (n_blocks * (int)sizeof(block_tq2_0::qs) + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
                dequantize_block_tq2_0<<<nb, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(x, (scalar_t *)y, k);
    });
}
