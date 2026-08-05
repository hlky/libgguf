#include <libgguf/gpu/compat.h>
#include <libgguf/gpu/dequantize/common.cuh>
#include <libgguf/gpu/dequantize/kernels.h>
#include <libgguf/gpu/dispatch.h>

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const dfloat d = __half2float(load_half(x[ib].d));

    v.x = x[ib].qs[iqs + 0] * d;
    v.y = x[ib].qs[iqs + 1] * d;
}


void dequantize_launch_q8_0(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
) {
    DISPATCH_FLOATING_TYPES(dtype, "dequantize_q8_0", [&] {
        dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>(x, (scalar_t *)y, k, stream);
    });
}
