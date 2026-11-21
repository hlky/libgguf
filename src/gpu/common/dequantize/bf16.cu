#include <type_traits>
#include <libgguf/gpu/compat.h>
#include <libgguf/gpu/dispatch.h>
#include <libgguf/gpu/dequantize/common.cuh>
#include <libgguf/gpu/dequantize/kernels.h>
#include <libgguf/gpu/dequantize/bf16.cuh>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
extern "C" hipError_t libgguf_gfx942_bf16_cast(const void *x, void *y, int count,
    libgguf_gpu_dtype dtype, int threads, int items, hipStream_t stream) {
    if (count < 0 || (count && (!x || !y)) ||
        (threads != 64 && threads != 128 && threads != 256 && threads != 512) ||
        (items != 1 && items != 2 && items != 4 && items != 8) ||
        (dtype != libgguf_gpu_dtype::f32 && dtype != libgguf_gpu_dtype::f16 && dtype != libgguf_gpu_dtype::bf16))
        return hipErrorInvalidValue;
    if (!count) return hipSuccess;
    const int blocks = 1 + (count - 1) / (threads * items);
    DISPATCH_FLOATING_TYPES(dtype, "gfx942_bf16_cast", [&] {
        auto launch = [&](auto width) {
            libgguf::gpu::dequantize_block_bf16<scalar_t, decltype(width)::value>
                <<<blocks, threads, 0, stream>>>(x, static_cast<scalar_t *>(y), count);
        };
        switch (items) {
            case 1: launch(std::integral_constant<int, 1>{}); break;
            case 2: launch(std::integral_constant<int, 2>{}); break;
            case 4: launch(std::integral_constant<int, 4>{}); break;
            case 8: launch(std::integral_constant<int, 8>{}); break;
        }
    });
    return hipPeekAtLastError();
}
#endif

void dequantize_bf16(
    const void * x,
    void * y,
    int k,
    libgguf_gpu_dtype dtype,
    StreamType stream
) {
    DISPATCH_FLOATING_TYPES(dtype, "dequantize_bf16", [&] {
        const int nb = (k + LIBGGUF_GPU_DEQUANTIZE_BLOCK_SIZE - 1) / LIBGGUF_GPU_DEQUANTIZE_BLOCK_SIZE;
                libgguf::gpu::dequantize_block_bf16<<<nb, LIBGGUF_GPU_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(x, (scalar_t *)y, k);
    });
}
