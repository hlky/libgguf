#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_dequantize_common.cuh"
#include "cuda_dequantize_kernels.h"
#include "dispatch_utils.h"

template<typename dst_t>
static __global__ void dequantize_block_tq2_0(const void * __restrict__ vx, dst_t * __restrict__ y, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }

    const block_tq2_0 * x = (const block_tq2_0 *) vx;
    const int ib = i / QK_K;
    const int j = i - ib * QK_K;
    const int group = j / 128;
    const int local = j - group * 128;
    const int plane = local / 32;
    const int byte = group * 32 + local - plane * 32;
    const int q = ((x[ib].qs[byte] >> (2 * plane)) & 3) - 1;
    y[i] = convert_from_float<dst_t>(__half2float(gguf_cuda_load_half(x[ib].d)) * q);
}


void gguf_cuda_dequantize_launch_tq2_0(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
) {
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "dequantize_tq2_0", [&] {
        const int nb = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
                dequantize_block_tq2_0<<<nb, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(x, (scalar_t *)y, k);
    });
}
