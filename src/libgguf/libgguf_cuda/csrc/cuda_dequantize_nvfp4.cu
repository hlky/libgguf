#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_dequantize_common.cuh"
#include "cuda_dequantize_kernels.h"
#include "dispatch_utils.h"

template<typename dst_t>
static __global__ void dequantize_block_nvfp4(const void * __restrict__ vx, dst_t * __restrict__ y, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }

    const block_nvfp4 * x = (const block_nvfp4 *) vx;
    const int ib = i / QK_NVFP4;
    const int j = i - ib * QK_NVFP4;
    const int group = j / 16;
    const int local = j - group * 16;
    const int byte = group * 8 + (local & 7);
    const int shift = local < 8 ? 0 : 4;
    const int q = (x[ib].qs[byte] >> shift) & 15;
    y[i] = convert_from_float<dst_t>(ue4m3_to_float(x[ib].d[group]) * kvalues_e2m1[q]);
}


void gguf_cuda_dequantize_launch_nvfp4(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
) {
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "dequantize_nvfp4", [&] {
        const int nb = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
                dequantize_block_nvfp4<<<nb, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(x, (scalar_t *)y, k);
    });
}
