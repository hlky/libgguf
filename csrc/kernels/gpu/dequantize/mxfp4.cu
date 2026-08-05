#include <libgguf/gpu/compat.h>
#include <libgguf/gpu/dequantize/common.cuh>
#include <libgguf/gpu/dequantize/kernels.h>
#include <libgguf/gpu/dispatch.h>
#include <libgguf/gpu/dequantize/mxfp4.cuh>

void dequantize_mxfp4(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
) {
    DISPATCH_FLOATING_TYPES(dtype, "dequantize_mxfp4", [&] {
        const int n_blocks = k / QK_MXFP4;
        const int nb = (n_blocks * (QK_MXFP4 / 2) + LIBGGUF_GPU_DEQUANTIZE_BLOCK_SIZE - 1) / LIBGGUF_GPU_DEQUANTIZE_BLOCK_SIZE;
                libgguf::gpu::dequantize_block_mxfp4<<<nb, LIBGGUF_GPU_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(x, (scalar_t *)y, k);
    });
}
