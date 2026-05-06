#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_dequantize_common.cuh"
#include "cuda_dequantize_kernels.h"
#include "dispatch_utils.h"

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const dfloat d = __half2float(gguf_cuda_low_half(x[ib].dm));
    const dfloat m = __half2float(gguf_cuda_high_half(x[ib].dm));

    const int vui = x[ib].qs[iqs];

    v.x = (vui & 0xF) * d + m;
    v.y = (vui >> 4) * d + m;
}


void gguf_cuda_dequantize_launch_q4_1(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
) {
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "dequantize_q4_1", [&] {
        dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>(x, (scalar_t *)y, k, stream);
    });
}
