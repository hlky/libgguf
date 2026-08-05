#include <libgguf/gpu/dispatch.h>
#include <libgguf/gpu/quantize/kernels.h>
#include <libgguf/gpu/quantize/mxfp4.cuh>

void quantize_mxfp4(
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
    constexpr int mxfp4_threads = 128;
    constexpr int mxfp4_blocks_per_cta = mxfp4_threads / 32;
    const int64_t n_blocks = k / QK_MXFP4;
    const int blocks = (int)((n_blocks + mxfp4_blocks_per_cta - 1) / mxfp4_blocks_per_cta);
    DISPATCH_FLOATING_TYPES(dtype, "quantize_mxfp4", [&] {
        libgguf::gpu::quantize_block_mxfp4<<<blocks, mxfp4_threads, 0, stream>>>(
            static_cast<const scalar_t *>(x), static_cast<block_mxfp4 *>(y), n_blocks);
    });
}
