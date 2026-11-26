#include <libgguf/gpu/dispatch.h>
#include <libgguf/gpu/quantize/weighted_qx.cuh>
#include <libgguf/gpu/quantize/kernels.h>
#include <libgguf/gpu/quantize/q4_1.cuh>

void quantize_q4_1(
    const void * x,
    const float * quant_weights,
    void * y,
    int64_t k,
    int64_t n_per_row,
    libgguf_gpu_dtype dtype,
    StreamType stream
) {
    if (quant_weights && k > 0 && n_per_row > 0) {
        DISPATCH_FLOATING_TYPES(dtype, "quantize_weighted_q4_1", [&] {
            libgguf::gpu::quantize_weighted_qx<4, true><<<dim3(k / n_per_row, libgguf::gpu::weighted_qx_groups(n_per_row)), 64, 0, stream>>>(
                static_cast<const scalar_t *>(x), quant_weights, static_cast<block_q4_1 *>(y), n_per_row);
        });
        return;
    }
    const int threads = 256;
    const int64_t n_blocks = k / QK4_1;
    const int blocks = (int)((n_blocks + threads - 1) / threads);
    DISPATCH_FLOATING_TYPES(dtype, "quantize_q4_1", [&] {
        libgguf::gpu::quantize_block_q4_1<<<blocks, threads, 0, stream>>>(
            static_cast<const scalar_t *>(x), static_cast<block_q4_1 *>(y), n_blocks);
    });
}
