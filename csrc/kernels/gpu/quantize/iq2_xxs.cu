#include <libgguf/gpu/dispatch.h>
#include <libgguf/gpu/quantize/iq2_xxs.cuh>
#include <libgguf/gpu/quantize/kernels.h>

void quantize_iq2_xxs(
    const void * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    libgguf_gpu_dtype dtype,
    StreamType stream
) {
    const int threads = 128;
    const int64_t n_blocks = k / QK_K;
    const int blocks = (int)((n_blocks + threads - 1) / threads);
    const int64_t n_blocks_per_row = n_per_row / QK_K;
    DISPATCH_FLOATING_TYPES(dtype, "quantize_iq2_xxs", [&] {
        libgguf::gpu::quantize_block_iq2_xxs<<<blocks, threads, 0, stream>>>(
            static_cast<const scalar_t *>(x), quant_weights, static_cast<block_iq2_xxs *>(y),
            n_blocks, n_blocks_per_row);
    });
}
