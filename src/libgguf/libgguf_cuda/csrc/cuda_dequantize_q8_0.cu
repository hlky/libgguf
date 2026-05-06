#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_dequantize_common.cuh"
#include "cuda_dequantize_kernels.h"
#include "dispatch_utils.h"

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const dfloat d = __half2float(gguf_cuda_load_half(x[ib].d));

    v.x = x[ib].qs[iqs + 0] * d;
    v.y = x[ib].qs[iqs + 1] * d;
}


void gguf_cuda_dequantize_launch_q8_0(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
) {
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "dequantize_q8_0", [&] {
        dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>(x, (scalar_t *)y, k, stream);
    });
}
