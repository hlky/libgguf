#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_dequantize_common.cuh"
#include "cuda_dequantize_kernels.h"
#include "dispatch_utils.h"

template<typename dst_t>
static __global__ void dequantize_block_q2_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const auto i   = blockIdx.x;
    const block_q2_K * x = (const block_q2_K *) vx;

    const auto tid = threadIdx.x;
    const int n   = tid/32;
    const int l   = tid - 32*n;
    const int is  = 8*n + l/16;

    const uint8_t q = x[i].qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;

    const float dall = __half2float(gguf_cuda_low_half(x[i].dm));
    const float dmin = __half2float(gguf_cuda_high_half(x[i].dm));
    y[l+ 0] = convert_from_float<dst_t>(dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4));
    y[l+32] = convert_from_float<dst_t>(dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4));
    y[l+64] = convert_from_float<dst_t>(dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4));
    y[l+96] = convert_from_float<dst_t>(dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4));
}


void gguf_cuda_dequantize_launch_q2_k(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
) {
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "dequantize_q2_k", [&] {
        const int nb = k / QK_K;
                dequantize_block_q2_K<<<nb, 64, 0, stream>>>(x, (scalar_t *)y);
    });
}
