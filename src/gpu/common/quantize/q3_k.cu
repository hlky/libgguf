#include <libgguf/gpu/dispatch.h>
#include <libgguf/gpu/quantize/kernels.h>
#include <libgguf/gpu/quantize/q3_k.cuh>

void quantize_q3_k(
    const void * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    libgguf_gpu_dtype dtype,
    StreamType stream
) {
    if (quant_weights && k > 0 && n_per_row >= QK_K) {
        DISPATCH_FLOATING_TYPES(dtype, "quantize_weighted_q3_k", [&] {
            libgguf::gpu::quantize_weighted_q3_k<<<k / QK_K, 64, 0, stream>>>(
                static_cast<const scalar_t *>(x), quant_weights, static_cast<block_q3_K *>(y), n_per_row / QK_K);
        });
        return;
    }
    constexpr int q3_k_threads = 64;
    constexpr int q3_k_blocks_per_cta = (q3_k_threads / 32) * 2;
    const int64_t n_blocks = k / QK_K;
    const int blocks = (int)((n_blocks + q3_k_blocks_per_cta - 1) / q3_k_blocks_per_cta);
    const size_t smem = q3_k_blocks_per_cta *
        (QK_K * sizeof(int8_t) + (QK_K / 16) * (sizeof(float) + sizeof(int8_t)));
    DISPATCH_FLOATING_TYPES(dtype, "quantize_q3_k", [&] {
        libgguf::gpu::quantize_block_q3_k<<<blocks, q3_k_threads, smem, stream>>>(
            static_cast<const scalar_t *>(x), static_cast<block_q3_K *>(y), n_blocks);
    });
}
