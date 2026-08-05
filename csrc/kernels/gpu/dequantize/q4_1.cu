#include <libgguf/gpu/compat.h>
#include <libgguf/gpu/dequantize/common.cuh>
#include <libgguf/gpu/dequantize/kernels.h>
#include <libgguf/gpu/dispatch.h>

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const dfloat d = __half2float(low_half(x[ib].dm));
    const dfloat m = __half2float(high_half(x[ib].dm));

    const int vui = x[ib].qs[iqs];

    v.x = (vui & 0xF) * d + m;
    v.y = (vui >> 4) * d + m;
}


void dequantize_launch_q4_1(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
) {
    DISPATCH_FLOATING_TYPES(dtype, "dequantize_q4_1", [&] {
        dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>(x, (scalar_t *)y, k, stream);
    });
}
