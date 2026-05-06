#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_dequantize_common.cuh"
#include "cuda_dequantize_kernels.h"
#include "dispatch_utils.h"

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const dfloat d = __half2float(gguf_cuda_low_half(x[ib].dm));
    const dfloat m = __half2float(gguf_cuda_high_half(x[ib].dm));

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0) * d + m;
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1) * d + m;
}


void gguf_cuda_dequantize_launch_q5_1(
    const void * x,
    void * y,
    int k,
    at::ScalarType dtype,
    cudaStream_t stream
) {
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "dequantize_q5_1", [&] {
        dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>(x, (scalar_t *)y, k, stream);
    });
}
