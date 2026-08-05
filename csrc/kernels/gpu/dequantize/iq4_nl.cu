#include <libgguf/gpu/compat.h>
#include <libgguf/gpu/dequantize/common.cuh>
#include <libgguf/gpu/dequantize/kernels.h>
#include <libgguf/gpu/dispatch.h>
#include <libgguf/gpu/dequantize/iq4_nl.cuh>

void dequantize_iq4_nl(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
) {
    DISPATCH_FLOATING_TYPES(dtype, "dequantize_iq4_nl", [&] {
        const int nb = (k + QK_K - 1) / QK_K;
                libgguf::gpu::dequantize_block_iq4_nl<<<nb, 32, 0, stream>>>(x, (scalar_t *)y);
    });
}
