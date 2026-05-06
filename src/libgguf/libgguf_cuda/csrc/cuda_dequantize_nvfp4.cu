#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_dequantize_common.cuh"
#include "cuda_dequantize_kernels.h"
#include "dispatch_utils.h"

template<typename dst_t>
static __global__ void dequantize_block_nvfp4(const void * __restrict__ vx, dst_t * __restrict__ y, const int k) {
    const int packed_index = blockDim.x*blockIdx.x + threadIdx.x;
    const int n_blocks = k / QK_NVFP4;
    if (packed_index >= n_blocks * (QK_NVFP4 / 2)) {
        return;
    }

    const block_nvfp4 * x = (const block_nvfp4 *) vx;
    const int ib = packed_index / (QK_NVFP4 / 2);
    const int byte = packed_index - ib * (QK_NVFP4 / 2);
    const int group = byte / 8;
    const int local = byte - group * 8;
    const uint8_t packed = x[ib].qs[byte];
    const float d = ue4m3_to_float(x[ib].d[group]);
    dst_t * yb = y + ib * QK_NVFP4 + group * 16 + local;
    yb[0] = convert_from_float<dst_t>(d * kvalues_e2m1[packed & 15]);
    yb[8] = convert_from_float<dst_t>(d * kvalues_e2m1[packed >> 4]);
}


void gguf_cuda_dequantize_launch_nvfp4(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
) {
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "dequantize_nvfp4", [&] {
        const int n_blocks = k / QK_NVFP4;
        const int nb = (n_blocks * (QK_NVFP4 / 2) + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
                dequantize_block_nvfp4<<<nb, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(x, (scalar_t *)y, k);
    });
}
