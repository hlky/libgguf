#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_dequantize_common.cuh"
#include "cuda_dequantize_kernels.h"
#include "dispatch_utils.h"

template<typename dst_t>
static __global__ void dequantize_block_iq2_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const auto i   = blockIdx.x;
    const block_iq2_s * x = (const block_iq2_s *) vx;

    const auto tid = threadIdx.x;
    const int il = tid/8; // 0...3
    const int ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t * grid = (const uint8_t *)(iq2s_grid + (x[i].qs[4*ib+il] | ((x[i].qh[ib] << (8-2*il)) & 0x300)));
    const float d = __half2float(gguf_cuda_load_half(x[i].d)) * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[i].qs[QK_K/8+4*ib+il];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}


void gguf_cuda_dequantize_launch_iq2_s(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
) {
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "dequantize_iq2_s", [&] {
        const int nb = k / QK_K;
                dequantize_block_iq2_s<<<nb, 32, 0, stream>>>(x, (scalar_t *)y);
    });
}
