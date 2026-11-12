#include <libgguf/gpu/dispatch.h>
#include <libgguf/gpu/quantize/iq3_s.cuh>
#include <libgguf/gpu/quantize/kernels.h>

void quantize_iq3_s(
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
    const int threads = 128;
    const int64_t n_blocks = k / QK_K;
    const int blocks = (int)((n_blocks + threads - 1) / threads);
    DISPATCH_FLOATING_TYPES(dtype, "quantize_iq3_s", [&] {
        libgguf::gpu::quantize_block_iq3_s<<<blocks, threads, 0, stream>>>(
            static_cast<const scalar_t *>(x), static_cast<block_iq3_s *>(y), n_blocks);
    });
}
