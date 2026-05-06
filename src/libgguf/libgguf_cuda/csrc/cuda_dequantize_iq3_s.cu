#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_dequantize_common.cuh"
#include "cuda_dequantize_kernels.h"
#include "dispatch_utils.h"

template<typename dst_t>
static __global__ void dequantize_block_iq3_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const auto i   = blockIdx.x;
    const block_iq3_s * x = (const block_iq3_s *) vx;

    const auto tid = threadIdx.x;
    const int il = tid/8; // 0...3
    const int ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t * qs = x[i].qs + 8*ib;
    const uint8_t * grid1 = (const uint8_t *)(iq3xs_grid + (qs[2*il+0] | ((x[i].qh[ib] << (8-2*il)) & 256)));
    const uint8_t * grid2 = (const uint8_t *)(iq3xs_grid + (qs[2*il+1] | ((x[i].qh[ib] << (7-2*il)) & 256)));
    const float d = __half2float(gguf_cuda_load_half(x[i].d)) * (1.0f + 2.0f * ((x[i].scales[ib/2] >> 4*(ib%2)) & 0xf));
    const uint8_t signs = x[i].signs[4*ib + il];
    for (int j = 0; j < 4; ++j) {
        const float v1 = grid1[j];
        const float v2 = grid2[j];
        y[j+0] = d * v1 * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * v2 * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
}


void gguf_cuda_dequantize_launch_iq3_s(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
) {
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "dequantize_iq3_s", [&] {
        const int nb = k / QK_K;
                dequantize_block_iq3_s<<<nb, 32, 0, stream>>>(x, (scalar_t *)y);
    });
}
