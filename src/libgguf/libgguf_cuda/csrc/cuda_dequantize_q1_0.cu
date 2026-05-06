#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_dequantize_common.cuh"
#include "cuda_dequantize_kernels.h"
#include "dispatch_utils.h"

template<typename dst_t>
static __global__ void dequantize_block_q1_0_pair(const void * __restrict__ vx, dst_t * __restrict__ y, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k / 2) {
        return;
    }

    const block_q1_0 * x = (const block_q1_0 *) vx;
    const int out = 2 * i;
    const int ib = out / QK1_0;
    const int j = out - ib * QK1_0;
    const int byte = j >> 3;
    const int bit = j & 7;
    const float d = __half2float(gguf_cuda_load_half(x[ib].d));
    const uint8_t packed = x[ib].qs[byte];
    dst_t * yb = y + ib * QK1_0;
    const int q0 = (packed >> bit) & 1;
    const int q1 = (packed >> (bit + 1)) & 1;
    yb[j] = convert_from_float<dst_t>(q0 ? d : -d);
    yb[j + 1] = convert_from_float<dst_t>(q1 ? d : -d);
}


void gguf_cuda_dequantize_launch_q1_0(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
) {
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "dequantize_q1_0", [&] {
        const int nb = (k / 2 + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
                dequantize_block_q1_0_pair<<<nb, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(x, (scalar_t *)y, k);
    });
}
