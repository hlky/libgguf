#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_dequantize_common.cuh"
#include "cuda_dequantize_kernels.h"
#include "dispatch_utils.h"

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const dfloat d = __half2float(gguf_cuda_load_half(x[ib].d));

    const int vui = x[ib].qs[iqs];

    v.x = ((vui & 0xF) - 8) * d;
    v.y = ((vui >> 4) - 8) * d;
}


void gguf_cuda_dequantize_launch_q4_0(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
) {
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "dequantize_q4_0", [&] {
        dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>(x, (scalar_t *)y, k, stream);
    });
}
