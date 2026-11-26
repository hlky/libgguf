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
    if (k <= 0) return;
    DISPATCH_FLOATING_TYPES(dtype, "dequantize_iq4_nl", [&] {
        const int nb = 1 + (k - 1) / QK_K;
        libgguf::gpu::dequantize_block_iq4_nl<<<nb, 32, 0, stream>>>(x, static_cast<scalar_t *>(y), k / QK4_NL);
    });
}
