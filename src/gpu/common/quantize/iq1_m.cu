#include <libgguf/gpu/dispatch.h>
#include <libgguf/gpu/quantize/iq1_m.cuh>
#include <libgguf/gpu/quantize/kernels.h>

void quantize_iq1_m(
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
    DISPATCH_FLOATING_TYPES(dtype, "quantize_iq1_m", [&] {
        libgguf::gpu::quantize_block_iq1_m<<<blocks, threads, 0, stream>>>(
            static_cast<const scalar_t *>(x), quant_weights, static_cast<block_iq1_m *>(y),
            n_blocks, n_blocks_per_row);
    });
}
