#include <libgguf/gpu/compat.h>
#include <libgguf/gpu/dequantize/common.cuh>
#include <libgguf/gpu/dequantize/kernels.h>
#include <libgguf/gpu/dispatch.h>
#include <libgguf/gpu/dequantize/q3_k.cuh>

void dequantize_q3_k(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
) {
    DISPATCH_FLOATING_TYPES(dtype, "dequantize_q3_k", [&] {
        const int nb = k / QK_K;
                libgguf::gpu::dequantize_block_q3_k<<<nb, 64, 0, stream>>>(x, (scalar_t *)y);
    });
}
