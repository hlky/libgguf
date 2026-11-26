#include <libgguf/gpu/dispatch.h>
#include <libgguf/gpu/quantize/weighted_qx.cuh>
#include <libgguf/gpu/quantize/kernels.h>
#include <libgguf/gpu/quantize/q5_0.cuh>

void quantize_q5_0(
    const void * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    libgguf_gpu_dtype dtype,
    StreamType stream
) {
    if (quant_weights && k > 0 && n_per_row > 0) {
        DISPATCH_FLOATING_TYPES(dtype, "quantize_weighted_q5_0", [&] {
            libgguf::gpu::quantize_weighted_qx<5><<<dim3(k / n_per_row, libgguf::gpu::weighted_qx_groups(n_per_row)), 64, 0, stream>>>(
                static_cast<const scalar_t *>(x), quant_weights, static_cast<block_q5_0 *>(y), n_per_row);
        });
        return;
    }
    constexpr int q_warp_threads = 128;
    constexpr int q_blocks_per_cta = q_warp_threads / 32;
    const int64_t n_blocks = k / QK5_0;
    const int blocks = (int)((n_blocks + q_blocks_per_cta - 1) / q_blocks_per_cta);
    DISPATCH_FLOATING_TYPES(dtype, "quantize_q5_0", [&] {
        libgguf::gpu::quantize_block_q5_0<<<blocks, q_warp_threads, 0, stream>>>(
            static_cast<const scalar_t *>(x), static_cast<block_q5_0 *>(y), n_blocks);
    });
}
