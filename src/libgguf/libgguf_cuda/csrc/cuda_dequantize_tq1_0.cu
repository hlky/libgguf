#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_dequantize_common.cuh"
#include "cuda_dequantize_kernels.h"
#include "dispatch_utils.h"

template<typename dst_t>
static __global__ void dequantize_block_tq1_0(const void * __restrict__ vx, dst_t * __restrict__ y, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }

    const block_tq1_0 * x = (const block_tq1_0 *) vx;
    const int ib = i / QK_K;
    const int j = i - ib * QK_K;
    const uint8_t pow3[5] = {1, 3, 9, 27, 81};

    uint8_t code;
    if (j < 160) {
        const int plane = j / 32;
        const int byte = j - plane * 32;
        code = x[ib].qs[byte] * pow3[plane];
    } else if (j < 240) {
        const int local = j - 160;
        const int plane = local / 16;
        const int byte = 32 + local - plane * 16;
        code = x[ib].qs[byte] * pow3[plane];
    } else {
        const int local = j - 240;
        const int plane = local / 4;
        const int byte = local - plane * 4;
        code = x[ib].qh[byte] * pow3[plane];
    }

    const int q = (((int)code * 3) >> 8) - 1;
    y[i] = convert_from_float<dst_t>(__half2float(gguf_cuda_load_half(x[ib].d)) * q);
}


void gguf_cuda_dequantize_launch_tq1_0(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
) {
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "dequantize_tq1_0", [&] {
        const int nb = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
                dequantize_block_tq1_0<<<nb, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(x, (scalar_t *)y, k);
    });
}
