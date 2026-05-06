#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_dequantize_common.cuh"
#include "cuda_dequantize_kernels.h"
#include "dispatch_utils.h"

template<typename dst_t>
static __global__ void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q5_K * x = (const block_q5_K *) vx;

    const auto i = blockIdx.x;

    // assume 64 threads - this is very slightly better than the one below
    const auto tid = threadIdx.x;
    const int il  = tid/16;   // il is in 0...3
    const int ir  = tid%16;   // ir is in 0...15
    const int is  = 2*il;     // is is in 0...6

    dst_t * y = yy + i*QK_K + 64*il + 2*ir;

    const float dall = __half2float(gguf_cuda_low_half(x[i].dm));
    const float dmin = __half2float(gguf_cuda_high_half(x[i].dm));

    const uint8_t * ql = x[i].qs + 32*il + 2*ir;
    const uint8_t * qh = x[i].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = convert_from_float<dst_t>(d1 * ((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0)) - m1);
    y[ 1] = convert_from_float<dst_t>(d1 * ((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0)) - m1);
    hm <<= 1;
    y[32] = convert_from_float<dst_t>(d2 * ((ql[0] >>  4) + (qh[0] & hm ? 16 : 0)) - m2);
    y[33] = convert_from_float<dst_t>(d2 * ((ql[1] >>  4) + (qh[1] & hm ? 16 : 0)) - m2);
}


void gguf_cuda_dequantize_launch_q5_k(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
) {
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "dequantize_q5_k", [&] {
        const int nb = k / QK_K;
                dequantize_block_q5_K<<<nb, 64, 0, stream>>>(x, (scalar_t *)y);
    });
}
