#pragma once

#include <cstdint>

#include <libgguf/formats.h>

#if defined(__clang__) && \
    (defined(__HIP__) || defined(__HIPCC__) || defined(__CUDA__))
#define LIBGGUF_QUANTIZE_FRAGMENT_INLINE                               \
    __attribute__((host)) __attribute__((device)) inline              \
        __attribute__((always_inline))
#elif defined(__CUDACC__)
#define LIBGGUF_QUANTIZE_FRAGMENT_INLINE __host__ __device__ __forceinline__
#else
#define LIBGGUF_QUANTIZE_FRAGMENT_INLINE inline
#endif

namespace libgguf
{
namespace gpu
{

inline constexpr std::uint32_t block_fragment_quantizer_api_revision = 1;

// Operations owns the platform-specific converged wave implementation. Its
// reduce_absmax operation must return the same absolute maximum to every lane.
// All 32 lanes represented by one Q8_0 block must participate.
template <ggml_type Format>
struct BlockFragmentQuantizer;

template <>
struct BlockFragmentQuantizer<GGML_TYPE_Q8_0>
{
    static constexpr int values_per_block = QK8_0;
    static constexpr int encoded_bytes = sizeof(block_q8_0);

    template <typename Src, typename Operations>
    static LIBGGUF_QUANTIZE_FRAGMENT_INLINE float quantize_lane(
        Src value, std::int8_t & quant, const Operations & operations)
    {
        const float input = static_cast<float>(value);
        const float absolute_maximum = operations.reduce_absmax(input);
        const float scale = operations.divide(absolute_maximum, 127.0f);
        const float inverse_scale = scale != 0.0f
            ? operations.divide(1.0f, scale)
            : 0.0f;
        quant = operations.round_to_int8(
            operations.multiply(input, inverse_scale));
        return scale;
    }
};

}
}

#undef LIBGGUF_QUANTIZE_FRAGMENT_INLINE
