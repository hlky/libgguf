#include <libgguf/gpu/dispatch.h>
#include <libgguf/gpu/quantize/kernels.h>
#include <libgguf/gpu/quantize/tq2_0.cuh>

void quantize_tq2_0(
    const void * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    libgguf_gpu_dtype dtype,
    StreamType stream
) {
    (void)quant_weights;
    (void)n_per_row;
    constexpr int tq2_threads = 128;
    constexpr int tq2_blocks_per_cta = tq2_threads / 32;
    const int64_t n_blocks = k / QK_K;
    const int blocks = (int)((n_blocks + tq2_blocks_per_cta - 1) / tq2_blocks_per_cta);
    DISPATCH_FLOATING_TYPES(dtype, "quantize_tq2_0", [&] {
        libgguf::gpu::quantize_block_tq2_0<<<blocks, tq2_threads, 0, stream>>>(
            static_cast<const scalar_t *>(x), static_cast<block_tq2_0 *>(y), n_blocks);
    });
}
