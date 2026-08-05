#include <libgguf/gpu/compat.h>
#include <libgguf/gpu/dequantize/common.cuh>
#include <libgguf/gpu/dequantize/kernels.h>
#include <libgguf/gpu/dispatch.h>
#include <libgguf/gpu/dequantize/q5_1.cuh>

void dequantize_q5_1(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
) {
    DISPATCH_FLOATING_TYPES(dtype, "dequantize_q5_1", [&] {
        const int nb = (k + 2*LIBGGUF_GPU_DEQUANTIZE_BLOCK_SIZE - 1) / (2*LIBGGUF_GPU_DEQUANTIZE_BLOCK_SIZE);
        libgguf::gpu::dequantize_block_q5_1<<<nb, LIBGGUF_GPU_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(x, (scalar_t *)y, k);
    });
}
