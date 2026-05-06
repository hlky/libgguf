#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_dequantize_common.cuh"
#include "cuda_dequantize_kernels.h"
#include "dispatch_utils.h"

template<typename dst_t>
static __global__ void dequantize_block_iq4_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const auto i   = blockIdx.x;
    const block_iq4_xs * x = (const block_iq4_xs *)vx;

    const auto tid = threadIdx.x;
    const int il = tid/8; // 0...3
    const int ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[i].qs + 16*ib + 4*il;
    const float d = __half2float(gguf_cuda_load_half(x[i].d)) * ((((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }
}


void gguf_cuda_dequantize_launch_iq4_xs(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
) {
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "dequantize_iq4_xs", [&] {
        const int nb = (k + QK_K - 1) / QK_K;
                dequantize_block_iq4_xs<<<nb, 32, 0, stream>>>(x, (scalar_t *)y);
    });
}
