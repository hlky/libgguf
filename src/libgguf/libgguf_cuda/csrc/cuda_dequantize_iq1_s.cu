#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_dequantize_common.cuh"
#include "cuda_dequantize_kernels.h"
#include "dispatch_utils.h"

template<typename dst_t>
static __global__ void dequantize_block_iq1_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq1_s * x = (const block_iq1_s  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const float delta = x[i].qh[ib] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
    const float d = __half2float(gguf_cuda_load_half(x[i].d)) * (2*((x[i].qh[ib] >> 12) & 7) + 1);
    const uint64_t grid = iq1s_grid[x[i].qs[4*ib+il] | (((x[i].qh[ib] >> 3*il) & 7) << 8)];
    const int8_t * q = (const int8_t *)&grid;
    for (int j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
}


void gguf_cuda_dequantize_launch_iq1_s(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
) {
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "dequantize_iq1_s", [&] {
        const int nb = k / QK_K;
                dequantize_block_iq1_s<<<nb, 32, 0, stream>>>(x, (scalar_t *)y);
    });
}
