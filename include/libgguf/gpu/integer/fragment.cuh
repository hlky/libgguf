#pragma once

#include <cstdint>

#include <libgguf/gpu/dequantize/fragment.cuh>

#if defined(__clang__) && \
    (defined(__HIP__) || defined(__HIPCC__) || defined(__CUDA__))
#define LIBGGUF_INTEGER_FRAGMENT_INLINE                                \
    __attribute__((host)) __attribute__((device)) inline              \
        __attribute__((always_inline))
#elif defined(__CUDACC__)
#define LIBGGUF_INTEGER_FRAGMENT_INLINE __host__ __device__ __forceinline__
#else
#define LIBGGUF_INTEGER_FRAGMENT_INLINE inline
#endif

namespace libgguf
{
namespace gpu
{

inline constexpr std::uint32_t block_integer_fragment_api_revision = 1;

template <ggml_type Format>
struct BlockIntegerFragment;

template <>
struct BlockIntegerFragment<GGML_TYPE_Q8_0>
{
    using Block = block_q8_0;
    static constexpr int values_per_block = QK8_0;
    static constexpr int encoded_bytes = sizeof(Block);

    static LIBGGUF_INTEGER_FRAGMENT_INLINE float decode_raw_scale(
        const std::uint8_t * encoded)
    {
        return fragment_detail::fp16_to_float(
            fragment_detail::load_u16_le(encoded));
    }

    static LIBGGUF_INTEGER_FRAGMENT_INLINE std::int8_t decode_raw_quant(
        const std::uint8_t * encoded, int value_index)
    {
        return static_cast<std::int8_t>(encoded[2 + value_index]);
    }
};

template <>
struct BlockIntegerFragment<GGML_TYPE_Q4_0>
{
    using Block = block_q4_0;
    static constexpr int values_per_block = QK4_0;
    static constexpr int encoded_bytes = sizeof(Block);

    static LIBGGUF_INTEGER_FRAGMENT_INLINE float decode_raw_scale(
        const std::uint8_t * encoded)
    {
        return fragment_detail::fp16_to_float(
            fragment_detail::load_u16_le(encoded));
    }

    static LIBGGUF_INTEGER_FRAGMENT_INLINE std::int8_t decode_raw_quant(
        const std::uint8_t * encoded, int value_index)
    {
        return static_cast<std::int8_t>(
            BlockFragmentDecoder<GGML_TYPE_Q4_0>::decode_raw_quant(
                encoded, value_index));
    }
};

}
}

#undef LIBGGUF_INTEGER_FRAGMENT_INLINE
