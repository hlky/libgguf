#pragma once

#include <libgguf/formats.h>
#include <libgguf/gpu/common.cuh>
#include <libgguf/types.h>

namespace libgguf
{
namespace gpu
{

template <ggml_type Format>
struct BlockFragmentDecoder;

template <>
struct BlockFragmentDecoder<GGML_TYPE_Q8_0>
{
    using Block = block_q8_0;
    static constexpr int values_per_block = QK8_0;

    static __device__ __forceinline__ float decode_value(const Block & block, int value_index)
    {
        return __half2float(load_half(block.d)) * static_cast<float>(block.qs[value_index]);
    }

    template <typename SetValue>
    static __device__ __forceinline__ void decode_8(
        const Block & block, int value_index, const SetValue & set_value)
    {
        const float scale = __half2float(load_half(block.d));
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

    static __device__ __forceinline__ float decode_value(const Block & block, int value_index)
    {
        const uint8_t packed = block.qs[value_index & 15];
        const int quant = value_index < 16
            ? static_cast<int>(packed & 0x0f)
            : static_cast<int>(packed >> 4);
        return __half2float(load_half(block.d)) * static_cast<float>(quant - 8);
    }

    template <typename SetValue>
    static __device__ __forceinline__ void decode_8(
        const Block & block, int value_index, const SetValue & set_value)
    {
        const float scale = __half2float(load_half(block.d));
        const bool high_nibble = value_index >= 16;
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            const uint8_t packed = block.qs[(value_index + i) & 15];
            const int quant = high_nibble
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

    static __device__ __forceinline__ float decode_value(const Block & block, int value_index)
    {
        const uint8_t packed = block.qs[value_index & 15];
        const int quant = value_index < 16
            ? static_cast<int>(packed & 0x0f)
            : static_cast<int>(packed >> 4);
        return __half2float(load_half(block.d)) * static_cast<float>(quant) +
            __half2float(load_half(block.m));
    }

    template <typename SetValue>
    static __device__ __forceinline__ void decode_8(
        const Block & block, int value_index, const SetValue & set_value)
    {
        const float scale = __half2float(load_half(block.d));
        const float minimum = __half2float(load_half(block.m));
        const bool high_nibble = value_index >= 16;
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            const uint8_t packed = block.qs[(value_index + i) & 15];
            const int quant = high_nibble
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

    static __device__ __forceinline__ uint32_t high_bits(const Block & block)
    {
        return static_cast<uint32_t>(block.qh[0]) |
            (static_cast<uint32_t>(block.qh[1]) << 8) |
            (static_cast<uint32_t>(block.qh[2]) << 16) |
            (static_cast<uint32_t>(block.qh[3]) << 24);
    }

    static __device__ __forceinline__ int decode_quant(
        const Block & block, uint32_t high, int value_index)
    {
        const uint8_t packed = block.qs[value_index & 15];
        const int low = value_index < 16
            ? static_cast<int>(packed & 0x0f)
            : static_cast<int>(packed >> 4);
        const int high_bit = static_cast<int>((high >> value_index) & 1u) << 4;
        return (low | high_bit) - 16;
    }

    static __device__ __forceinline__ float decode_value(const Block & block, int value_index)
    {
        return __half2float(load_half(block.d)) *
            static_cast<float>(decode_quant(block, high_bits(block), value_index));
    }

    template <typename SetValue>
    static __device__ __forceinline__ void decode_8(
        const Block & block, int value_index, const SetValue & set_value)
    {
        const float scale = __half2float(load_half(block.d));
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

    static __device__ __forceinline__ uint32_t high_bits(const Block & block)
    {
        return static_cast<uint32_t>(block.qh[0]) |
            (static_cast<uint32_t>(block.qh[1]) << 8) |
            (static_cast<uint32_t>(block.qh[2]) << 16) |
            (static_cast<uint32_t>(block.qh[3]) << 24);
    }

    static __device__ __forceinline__ int decode_quant(
        const Block & block, uint32_t high, int value_index)
    {
        const uint8_t packed = block.qs[value_index & 15];
        const int low = value_index < 16
            ? static_cast<int>(packed & 0x0f)
            : static_cast<int>(packed >> 4);
        const int high_bit = static_cast<int>((high >> value_index) & 1u) << 4;
        return low | high_bit;
    }

    static __device__ __forceinline__ float decode_value(const Block & block, int value_index)
    {
        return __half2float(load_half(block.d)) *
            static_cast<float>(decode_quant(block, high_bits(block), value_index)) +
            __half2float(load_half(block.m));
    }

    template <typename SetValue>
    static __device__ __forceinline__ void decode_8(
        const Block & block, int value_index, const SetValue & set_value)
    {
        const float scale = __half2float(load_half(block.d));
        const float minimum = __half2float(load_half(block.m));
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
