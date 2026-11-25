#include <libgguf/gpu/compat.h>
#include <libgguf/gpu/dequantize/kernels.h>
#include <libgguf/gpu/dequantize/prism.cuh>

void dequantize_pq2_0(const void *x, void *y, int k,
                      libgguf_gpu_dtype dtype, StreamType stream)
{
    DISPATCH_FLOATING_TYPES(dtype, "dequantize_pq2_0", [&] {
        const int grid = ((k / 128) * 32 + LIBGGUF_GPU_DEQUANTIZE_BLOCK_SIZE - 1) /
                         LIBGGUF_GPU_DEQUANTIZE_BLOCK_SIZE;
        libgguf::gpu::dequantize_block_prism<scalar_t, false>
            <<<grid, LIBGGUF_GPU_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(x, (scalar_t *)y, k);
    });
}

void dequantize_ptq1_0(const void *x, void *y, int k,
                       libgguf_gpu_dtype dtype, StreamType stream)
{
    DISPATCH_FLOATING_TYPES(dtype, "dequantize_ptq1_0", [&] {
        const int grid = ((k / 128) * 26 + LIBGGUF_GPU_DEQUANTIZE_BLOCK_SIZE - 1) /
                         LIBGGUF_GPU_DEQUANTIZE_BLOCK_SIZE;
        libgguf::gpu::dequantize_block_prism<scalar_t, true>
            <<<grid, LIBGGUF_GPU_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(x, (scalar_t *)y, k);
    });
}
