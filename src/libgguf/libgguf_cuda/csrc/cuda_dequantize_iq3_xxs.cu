#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_dequantize_common.cuh"
#include "cuda_dequantize_kernels.h"
#include "dispatch_utils.h"

template<typename dst_t>
static __global__ void dequantize_block_iq3_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const auto i   = blockIdx.x;
    const block_iq3_xxs * x = (const block_iq3_xxs  *) vx;

    const auto tid = threadIdx.x;
    const int il = tid/8; // 0...3
    const int ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t  * q3 = x[i].qs + 8*ib;
    const uint16_t * gas = (const uint16_t *)(x[i].qs + QK_K/4) + 2*ib;
    const uint8_t  * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*il+0]);
    const uint8_t  * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*il+1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = __half2float(gguf_cuda_load_half(x[i].d)) * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
}


void gguf_cuda_dequantize_launch_iq3_xxs(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
) {
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "dequantize_iq3_xxs", [&] {
        const int nb = k / QK_K;
                dequantize_block_iq3_xxs<<<nb, 32, 0, stream>>>(x, (scalar_t *)y);
    });
}
