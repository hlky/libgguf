#pragma once

#include <libgguf/common/native_formats.h>
#include <libgguf/gpu/dequantize/fragment.cuh>

#if defined(__HIPCC__) || defined(__HIP__)
#define LIBGGUF_NATIVE_HD __attribute__((host, device)) inline
#elif defined(__CUDACC__)
#define LIBGGUF_NATIVE_HD __host__ __device__ inline
#else
#define LIBGGUF_NATIVE_HD inline
#endif

namespace libgguf {

inline constexpr std::uint32_t native_fragment_api_revision = 1;

namespace native_detail {

LIBGGUF_NATIVE_HD int signed_nibble(std::uint8_t value) {
    const int bits = value & 15;
    return bits < 8 ? bits : bits - 16;
}

LIBGGUF_NATIVE_HD float half_bits_to_float(std::uint16_t bits) {
#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
    // Use the target conversion for finite scales, retaining the scalar bit
    // expansion for infinity/NaN payloads. Exhaustive device parity covers
    // subnormals and signed zero as well as ordinary finite scales.
    if ((bits & 0x7c00u) != 0x7c00u)
        return static_cast<float>(__builtin_bit_cast(_Float16, bits));
#endif
    return gpu::fragment_detail::fp16_to_float(bits);
}

LIBGGUF_NATIVE_HD float read_half(const std::uint8_t* bytes) {
    return half_bits_to_float(gpu::fragment_detail::load_u16_le(bytes));
}

LIBGGUF_NATIVE_HD int quant(NativeFormat format, const std::uint8_t* block, int k) {
    if (format == NativeFormat::s8g32) {
        const int bits = block[2 + k];
        return bits < 128 ? bits : bits - 256;
    }
    const int start = format == NativeFormat::s4h256 ? 10 : 2;
    return signed_nibble(block[start + k / 2] >> ((k & 1) * 4));
}

LIBGGUF_NATIVE_HD float scale(NativeFormat format, const std::uint8_t* block, int k) {
    const float base = read_half(block);
    return format == NativeFormat::s4h256
        ? base * static_cast<float>(block[2 + k / 32]) : base;
}

LIBGGUF_NATIVE_HD bool valid_block(NativeFormat format, const std::uint8_t* block) {
    const auto info = native_format_info(format);
    if (!info.name) return false;
    const auto half = gpu::fragment_detail::load_u16_le(block);
    // Canonical native scales are finite, nonnegative, and use positive zero.
    if ((half & 0x8000u) || (half & 0x7c00u) == 0x7c00u) return false;
    for (unsigned k = 0; k < info.values_per_block; ++k) {
        const int q = quant(format, block, static_cast<int>(k));
        if (format == NativeFormat::s8g32 && q == -128) return false;
        if ((half == 0 || (format == NativeFormat::s4h256 && block[2 + k / 32] == 0))
            && q != 0) return false;
    }
    if (half == 0 && format == NativeFormat::s4h256) {
        for (int g = 0; g < 8; ++g) if (block[2 + g] != 0) return false;
    }
    return true;
}

} // namespace native_detail

template <NativeFormat Format>
struct NativeBlockCodec {
    static constexpr auto geometry = native_format_info(Format);
    static_assert(geometry.values_per_block != 0, "unknown native format");
    static constexpr int values_per_block = geometry.values_per_block;
    static constexpr int values_per_scale_group = geometry.values_per_group;
    static constexpr int encoded_bytes = geometry.bytes_per_block;
    struct Block { std::uint8_t bytes[encoded_bytes]; };
    static constexpr int values_per_decode_group = values_per_block;
    static constexpr int threads_per_decode_group = 32;
    template <class T>
    static LIBGGUF_NATIVE_HD void decode_group(const void* row, T* output,
        std::int64_t group, int thread) {
        const auto* encoded = static_cast<const std::uint8_t*>(row) + group * encoded_bytes;
        for (int k = thread; k < values_per_block; k += threads_per_decode_group)
            output[k] = static_cast<T>(decode_raw_value(encoded, k));
    }
    static LIBGGUF_NATIVE_HD float decode_value(const Block& block, int k) {
        return decode_raw_value(block.bytes, k);
    }

    static LIBGGUF_NATIVE_HD int decode_raw_quant(const std::uint8_t* block, int k) {
        return native_detail::quant(Format, block, k);
    }
    static LIBGGUF_NATIVE_HD float decode_raw_scale(const std::uint8_t* block, int k) {
        return native_detail::scale(Format, block, k);
    }
    static LIBGGUF_NATIVE_HD float decode_raw_value(const std::uint8_t* block, int k) {
        return decode_raw_scale(block, k) * static_cast<float>(decode_raw_quant(block, k));
    }
    template <class SetValue>
    static LIBGGUF_NATIVE_HD void decode_raw_8(
        const std::uint8_t* block, int k, const SetValue& set) {
        for (int i = 0; i < 8; ++i) set(i, decode_raw_value(block, k + i));
    }
    template <class SetValue>
    static LIBGGUF_NATIVE_HD void decode_8(const Block& block, int k, const SetValue& set) {
        decode_raw_8(block.bytes, k, set);
    }
};

// One complete K256 decode group can contain several physical native blocks.
// The caller supplies the complete encoded group, including explicit K padding.
template<NativeFormat Format>
struct NativeCooperativeBlockDecoder : NativeBlockCodec<Format> {
    using Codec=NativeBlockCodec<Format>;
    static constexpr int values_per_decode_group=256;
    static constexpr int threads_per_decode_group=32;
    template<class T>
    static LIBGGUF_NATIVE_HD void decode_group(const void* source,T* output,std::int64_t group,int thread) {
        const auto* bytes=static_cast<const std::uint8_t*>(source)+group*(256/Codec::values_per_block)*Codec::encoded_bytes;
        for(int k=thread;k<256;k+=threads_per_decode_group)
            output[k]=static_cast<T>(Codec::decode_raw_value(bytes+(k/Codec::values_per_block)*Codec::encoded_bytes,k%Codec::values_per_block));
    }
};

} // namespace libgguf
