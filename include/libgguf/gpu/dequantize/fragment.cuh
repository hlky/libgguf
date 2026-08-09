#pragma once

#include <cstdint>
#include <cstring>

#include <libgguf/formats.h>

#if defined(__clang__) && \
    (defined(__HIP__) || defined(__HIPCC__) || defined(__CUDA__))
#define LIBGGUF_FRAGMENT_INLINE                                        \
    __attribute__((host)) __attribute__((device)) inline              \
        __attribute__((always_inline))
#elif defined(__CUDACC__)
#define LIBGGUF_FRAGMENT_INLINE __host__ __device__ __forceinline__
#else
#define LIBGGUF_FRAGMENT_INLINE inline
#endif

namespace libgguf
{
namespace gpu
{

inline constexpr std::uint32_t block_fragment_decoder_api_revision = 1;

namespace fragment_detail
{

LIBGGUF_FRAGMENT_INLINE float float_from_bits(std::uint32_t bits)
{
#if (defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __builtin_bit_cast(float, bits);
#else
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
#endif
}

LIBGGUF_FRAGMENT_INLINE float fp16_to_float(std::uint16_t value)
{
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16;
    std::uint32_t exponent = (value >> 10) & 0x1fu;
    std::uint32_t mantissa = value & 0x03ffu;
    if (exponent == 0)
    {
        if (mantissa == 0) return float_from_bits(sign);
        std::int32_t unbiased_exponent = -14;
        while ((mantissa & 0x0400u) == 0)
        {
            mantissa <<= 1;
            --unbiased_exponent;
        }
        mantissa &= 0x03ffu;
        return float_from_bits(
            sign |
            (static_cast<std::uint32_t>(unbiased_exponent + 127) << 23) |
            (mantissa << 13));
    }
    if (exponent == 0x1fu)
        return float_from_bits(sign | 0x7f800000u | (mantissa << 13));
    exponent += 127u - 15u;
    return float_from_bits(sign | (exponent << 23) | (mantissa << 13));
}

}

template <ggml_type Format>
struct BlockFragmentDecoder;

template <>
struct BlockFragmentDecoder<GGML_TYPE_Q8_0>
{
    using Block = block_q8_0;
    static constexpr int values_per_block = QK8_0;

    static LIBGGUF_FRAGMENT_INLINE float decode_value(const Block & block, int value_index)
    {
        return fragment_detail::fp16_to_float(block.d) * static_cast<float>(block.qs[value_index]);
    }

    template <typename SetValue>
    static LIBGGUF_FRAGMENT_INLINE void decode_8(
        const Block & block, int value_index, const SetValue & set_value)
    {
        const float scale = fragment_detail::fp16_to_float(block.d);
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            set_value(i, scale * static_cast<float>(block.qs[value_index + i]));
        }
    }
};

template <>
struct BlockFragmentDecoder<GGML_TYPE_Q4_0>
{
    using Block = block_q4_0;
    static constexpr int values_per_block = QK4_0;

    static LIBGGUF_FRAGMENT_INLINE float decode_value(const Block & block, int value_index)
    {
        const uint8_t packed = block.qs[value_index & 15];
        const int quant = value_index < 16
            ? static_cast<int>(packed & 0x0f)
            : static_cast<int>(packed >> 4);
        return fragment_detail::fp16_to_float(block.d) * static_cast<float>(quant - 8);
    }

    template <typename SetValue>
    static LIBGGUF_FRAGMENT_INLINE void decode_8(
        const Block & block, int value_index, const SetValue & set_value)
    {
        const float scale = fragment_detail::fp16_to_float(block.d);
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            const uint8_t packed = block.qs[(value_index + i) & 15];
            const int quant = value_index + i >= 16
                ? static_cast<int>(packed >> 4)
                : static_cast<int>(packed & 0x0f);
            set_value(i, scale * static_cast<float>(quant - 8));
        }
    }
};

template <>
struct BlockFragmentDecoder<GGML_TYPE_Q4_1>
{
    using Block = block_q4_1;
    static constexpr int values_per_block = QK4_1;

    static LIBGGUF_FRAGMENT_INLINE float decode_value(const Block & block, int value_index)
    {
        const uint8_t packed = block.qs[value_index & 15];
        const int quant = value_index < 16
            ? static_cast<int>(packed & 0x0f)
            : static_cast<int>(packed >> 4);
        return fragment_detail::fp16_to_float(block.d) * static_cast<float>(quant) +
            fragment_detail::fp16_to_float(block.m);
    }

    template <typename SetValue>
    static LIBGGUF_FRAGMENT_INLINE void decode_8(
        const Block & block, int value_index, const SetValue & set_value)
    {
        const float scale = fragment_detail::fp16_to_float(block.d);
        const float minimum = fragment_detail::fp16_to_float(block.m);
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            const uint8_t packed = block.qs[(value_index + i) & 15];
            const int quant = value_index + i >= 16
                ? static_cast<int>(packed >> 4)
                : static_cast<int>(packed & 0x0f);
            set_value(i, scale * static_cast<float>(quant) + minimum);
        }
    }
};

template <>
struct BlockFragmentDecoder<GGML_TYPE_Q5_0>
{
    using Block = block_q5_0;
    static constexpr int values_per_block = QK5_0;

    static LIBGGUF_FRAGMENT_INLINE uint32_t high_bits(const Block & block)
    {
        return static_cast<uint32_t>(block.qh[0]) |
            (static_cast<uint32_t>(block.qh[1]) << 8) |
            (static_cast<uint32_t>(block.qh[2]) << 16) |
            (static_cast<uint32_t>(block.qh[3]) << 24);
    }

    static LIBGGUF_FRAGMENT_INLINE int decode_quant(
        const Block & block, uint32_t high, int value_index)
    {
        const uint8_t packed = block.qs[value_index & 15];
        const int low = value_index < 16
            ? static_cast<int>(packed & 0x0f)
            : static_cast<int>(packed >> 4);
        const int high_bit = static_cast<int>((high >> value_index) & 1u) << 4;
        return (low | high_bit) - 16;
    }

    static LIBGGUF_FRAGMENT_INLINE float decode_value(const Block & block, int value_index)
    {
        return fragment_detail::fp16_to_float(block.d) *
            static_cast<float>(decode_quant(block, high_bits(block), value_index));
    }

    template <typename SetValue>
    static LIBGGUF_FRAGMENT_INLINE void decode_8(
        const Block & block, int value_index, const SetValue & set_value)
    {
        const float scale = fragment_detail::fp16_to_float(block.d);
        const uint32_t high = high_bits(block);
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            set_value(
                i,
                scale * static_cast<float>(decode_quant(block, high, value_index + i)));
        }
    }
};

template <>
struct BlockFragmentDecoder<GGML_TYPE_Q5_1>
{
    using Block = block_q5_1;
    static constexpr int values_per_block = QK5_1;

    static LIBGGUF_FRAGMENT_INLINE uint32_t high_bits(const Block & block)
    {
        return static_cast<uint32_t>(block.qh[0]) |
            (static_cast<uint32_t>(block.qh[1]) << 8) |
            (static_cast<uint32_t>(block.qh[2]) << 16) |
            (static_cast<uint32_t>(block.qh[3]) << 24);
    }

    static LIBGGUF_FRAGMENT_INLINE int decode_quant(
        const Block & block, uint32_t high, int value_index)
    {
        const uint8_t packed = block.qs[value_index & 15];
        const int low = value_index < 16
            ? static_cast<int>(packed & 0x0f)
            : static_cast<int>(packed >> 4);
        const int high_bit = static_cast<int>((high >> value_index) & 1u) << 4;
        return low | high_bit;
    }

    static LIBGGUF_FRAGMENT_INLINE float decode_value(const Block & block, int value_index)
    {
        return fragment_detail::fp16_to_float(block.d) *
            static_cast<float>(decode_quant(block, high_bits(block), value_index)) +
            fragment_detail::fp16_to_float(block.m);
    }

    template <typename SetValue>
    static LIBGGUF_FRAGMENT_INLINE void decode_8(
        const Block & block, int value_index, const SetValue & set_value)
    {
        const float scale = fragment_detail::fp16_to_float(block.d);
        const float minimum = fragment_detail::fp16_to_float(block.m);
        const uint32_t high = high_bits(block);
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            set_value(
                i,
                scale * static_cast<float>(decode_quant(block, high, value_index + i)) + minimum);
        }
    }
};

}
}

#undef LIBGGUF_FRAGMENT_INLINE
