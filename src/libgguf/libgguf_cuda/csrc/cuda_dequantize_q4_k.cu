#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_dequantize_common.cuh"
#include "cuda_dequantize_kernels.h"
#include "dispatch_utils.h"

template<typename dst_t>
static __global__ void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q4_K * x = (const block_q4_K *) vx;

    const auto i = blockIdx.x;

    // assume 32 threads
    const auto tid = threadIdx.x;
    const int il  = tid/8;
    const int ir  = tid%8;
    const int is  = 2*il;
    const int n   = 4;

    dst_t * y = yy + i*QK_K + 64*il + n*ir;

    const float dall = __half2float(gguf_cuda_low_half(x[i].dm));
    const float dmin = __half2float(gguf_cuda_high_half(x[i].dm));

    const uint8_t * q = x[i].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc;
    const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc;
    const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = convert_from_float<dst_t>(d1 * (q[l] & 0xF) - m1);
        y[l +32] = convert_from_float<dst_t>(d2 * (q[l] >> 4) - m2);
    }
}


void gguf_cuda_dequantize_launch_q4_k(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
) {
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "dequantize_q4_k", [&] {
        const int nb = k / QK_K;
        dequantize_block_q4_K<<<nb, 32, 0, stream>>>(x, (scalar_t *)y);
    });
}
