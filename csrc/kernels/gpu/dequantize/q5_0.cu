#include <libgguf/gpu/compat.h>
#include <libgguf/gpu/dequantize/common.cuh>
#include <libgguf/gpu/dequantize/kernels.h>
#include <libgguf/gpu/dispatch.h>

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const dfloat d = __half2float(load_half(x[ib].d));

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = (((x[ib].qs[iqs] & 0xf) | xh_0) - 16) * d;
    v.y = (((x[ib].qs[iqs] >>  4) | xh_1) - 16) * d;
}


void dequantize_launch_q5_0(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
) {
    DISPATCH_FLOATING_TYPES(dtype, "dequantize_q5_0", [&] {
        dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>(x, (scalar_t *)y, k, stream);
    });
}
