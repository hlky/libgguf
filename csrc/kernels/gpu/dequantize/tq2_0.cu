#include <libgguf/gpu/compat.h>
#include <libgguf/gpu/dequantize/common.cuh>
#include <libgguf/gpu/dequantize/kernels.h>
#include <libgguf/gpu/dispatch.h>
#include <libgguf/gpu/dequantize/tq2_0.cuh>

void dequantize_tq2_0(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
) {
    DISPATCH_FLOATING_TYPES(dtype, "dequantize_tq2_0", [&] {
        const int n_blocks = k / QK_K;
        const int nb = (n_blocks * (int)sizeof(block_tq2_0::qs) + LIBGGUF_GPU_DEQUANTIZE_BLOCK_SIZE - 1) / LIBGGUF_GPU_DEQUANTIZE_BLOCK_SIZE;
                libgguf::gpu::dequantize_block_tq2_0<<<nb, LIBGGUF_GPU_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(x, (scalar_t *)y, k);
    });
}
