#include <libgguf/gpu/dispatch.h>
#include <libgguf/gpu/quantize/kernels.h>
#include <libgguf/gpu/quantize/q2_k.cuh>

void quantize_q2_k(
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
    constexpr int q2_k_threads = 64;
    constexpr int q2_k_blocks_per_cta = (q2_k_threads / 32) * 2;
    const int64_t n_blocks = k / QK_K;
    const int blocks = (int)((n_blocks + q2_k_blocks_per_cta - 1) / q2_k_blocks_per_cta);
    const size_t smem = q2_k_blocks_per_cta * (QK_K * sizeof(uint8_t) + 2 * (QK_K / 16) * sizeof(float));
    DISPATCH_FLOATING_TYPES(dtype, "quantize_q2_k", [&] {
        libgguf::gpu::quantize_block_q2_k<<<blocks, q2_k_threads, smem, stream>>>(
            static_cast<const scalar_t *>(x), static_cast<block_q2_K *>(y), n_blocks);
    });
}
