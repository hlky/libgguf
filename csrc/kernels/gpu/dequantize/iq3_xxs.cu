#include <libgguf/gpu/compat.h>
#include <libgguf/gpu/dequantize/common.cuh>
#include <libgguf/gpu/dequantize/kernels.h>
#include <libgguf/gpu/dispatch.h>
#include <libgguf/gpu/dequantize/iq3_xxs.cuh>

void dequantize_iq3_xxs(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
) {
    DISPATCH_FLOATING_TYPES(dtype, "dequantize_iq3_xxs", [&] {
        const int nb = k / QK_K;
                libgguf::gpu::dequantize_block_iq3_xxs<<<nb, 32, 0, stream>>>(x, (scalar_t *)y);
    });
}
