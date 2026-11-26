#include <libgguf/gpu/dispatch.h>
#include <libgguf/gpu/quantize/weighted_k.cuh>
#include <libgguf/gpu/quantize/kernels.h>
#include <libgguf/gpu/quantize/q5_k.cuh>

void quantize_q5_k(
    const void * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    libgguf_gpu_dtype dtype,
    StreamType stream
) {
    if (quant_weights && k > 0 && n_per_row >= QK_K) {
        DISPATCH_FLOATING_TYPES(dtype, "quantize_weighted_q5_k", [&] {
            libgguf::gpu::quantize_weighted_k<5><<<k / QK_K, 64, 0, stream>>>(
                static_cast<const scalar_t *>(x), quant_weights, static_cast<block_q5_K *>(y), n_per_row / QK_K);
        });
        return;
    }
    constexpr int q5_k_threads = 64;
    constexpr int q5_k_blocks_per_cta = (q5_k_threads / 32) * 4;
    const int64_t n_blocks = k / QK_K;
    const int blocks = (int)((n_blocks + q5_k_blocks_per_cta - 1) / q5_k_blocks_per_cta);
    const size_t smem = q5_k_blocks_per_cta *
        (QK_K * sizeof(uint8_t) + 2 * (QK_K / 32) * sizeof(float));
    DISPATCH_FLOATING_TYPES(dtype, "quantize_q5_k", [&] {
        libgguf::gpu::quantize_block_q5_k<<<blocks, q5_k_threads, smem, stream>>>(
            static_cast<const scalar_t *>(x), static_cast<block_q5_K *>(y), n_blocks);
    });
}
