#include <libgguf/gpu/compat.h>
#include <libgguf/gpu/dequantize/common.cuh>
#include <libgguf/gpu/dequantize/kernels.h>
#include <libgguf/gpu/dispatch.h>
#include <libgguf/gpu/dequantize/tq1_0.cuh>

void dequantize_tq1_0(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
) {
    DISPATCH_FLOATING_TYPES(dtype, "dequantize_tq1_0", [&] {
        const int n_blocks = k / QK_K;
        const int packed_per_block = (int)(sizeof(block_tq1_0::qs) + sizeof(block_tq1_0::qh));
        const int nb = (n_blocks * packed_per_block + LIBGGUF_GPU_DEQUANTIZE_BLOCK_SIZE - 1) / LIBGGUF_GPU_DEQUANTIZE_BLOCK_SIZE;
                libgguf::gpu::dequantize_block_tq1_0<<<nb, LIBGGUF_GPU_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(x, (scalar_t *)y, k);
    });
}
