#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_dequantize_common.cuh"
#include "cuda_dequantize_kernels.h"
#include "dispatch_utils.h"

template<typename dst_t>
static __global__ void dequantize_block_bf16(const void * __restrict__ vx, dst_t * __restrict__ y, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }

    const uint16_t value = ((const uint16_t *) vx)[i];
    y[i] = convert_from_float<dst_t>(bitcast_u32_to_float((uint32_t)value << 16));
}


void gguf_cuda_dequantize_launch_bf16(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
) {
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "dequantize_bf16", [&] {
        const int nb = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
                dequantize_block_bf16<<<nb, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(x, (scalar_t *)y, k);
    });
}
