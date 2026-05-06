#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_dequantize_common.cuh"
#include "cuda_dequantize_kernels.h"
#include "dispatch_utils.h"

template<typename dst_t>
static __global__ void dequantize_block_mxfp4(const void * __restrict__ vx, dst_t * __restrict__ y, const int k) {
    const int packed_index = blockDim.x*blockIdx.x + threadIdx.x;
    const int n_blocks = k / QK_MXFP4;
    if (packed_index >= n_blocks * (QK_MXFP4 / 2)) {
        return;
    }

    const block_mxfp4 * x = (const block_mxfp4 *) vx;
    const int ib = packed_index / (QK_MXFP4 / 2);
    const int byte = packed_index - ib * (QK_MXFP4 / 2);
    const uint8_t packed = x[ib].qs[byte];
    const float d = e8m0_to_float(x[ib].e);
    dst_t * yb = y + ib * QK_MXFP4;
    yb[byte] = convert_from_float<dst_t>(d * kvalues_e2m1[packed & 15]);
    yb[byte + 16] = convert_from_float<dst_t>(d * kvalues_e2m1[packed >> 4]);
}


void gguf_cuda_dequantize_launch_mxfp4(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
) {
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "dequantize_mxfp4", [&] {
        const int n_blocks = k / QK_MXFP4;
        const int nb = (n_blocks * (QK_MXFP4 / 2) + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
                dequantize_block_mxfp4<<<nb, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(x, (scalar_t *)y, k);
    });
}
