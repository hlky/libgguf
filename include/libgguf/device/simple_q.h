// SPDX-License-Identifier: Apache-2.0
// Runtime-free fragment decoding for the GGUF/GGML Q4_0 and Q8_0 wire formats.
// Format behavior follows the libgguf CPU reference implementation, which is
// adapted from llama.cpp/GGML. See the repository NOTICE for provenance.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__clang__) && \
    (defined(__HIP__) || defined(__HIPCC__) || defined(__CUDA__))
#define LIBGGUF_SIMPLE_Q_INLINE                                          \
  __attribute__((host)) __attribute__((device)) inline                  \
      __attribute__((always_inline))
#elif defined(__CUDACC__)
#define LIBGGUF_SIMPLE_Q_INLINE __host__ __device__ __forceinline__
#else
#define LIBGGUF_SIMPLE_Q_INLINE inline
#endif

namespace libgguf::device {

inline constexpr std::uint32_t simple_q_api_revision = 2;
inline constexpr std::uint32_t simple_q_block_values = 32;

struct q4_0_block {
  std::uint16_t scale_bits;
  std::uint8_t quants[simple_q_block_values / 2];
};

struct q8_0_block {
  std::uint16_t scale_bits;
  std::int8_t quants[simple_q_block_values];
};

inline constexpr std::size_t q4_0_block_bytes = 18;
inline constexpr std::size_t q8_0_block_bytes = 34;

static_assert(sizeof(q4_0_block) == q4_0_block_bytes);
static_assert(sizeof(q8_0_block) == q8_0_block_bytes);
static_assert(alignof(q4_0_block) == alignof(std::uint16_t));
static_assert(alignof(q8_0_block) == alignof(std::uint16_t));
static_assert(std::is_standard_layout_v<q4_0_block>);
static_assert(std::is_standard_layout_v<q8_0_block>);
static_assert(std::is_trivially_copyable_v<q4_0_block>);
static_assert(std::is_trivially_copyable_v<q8_0_block>);
static_assert(offsetof(q4_0_block, scale_bits) == 0);
static_assert(offsetof(q4_0_block, quants) == 2);
static_assert(offsetof(q8_0_block, scale_bits) == 0);
static_assert(offsetof(q8_0_block, quants) == 2);

namespace detail {

LIBGGUF_SIMPLE_Q_INLINE float float_from_bits(std::uint32_t bits) {
#if (defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__) || \
    defined(__CUDA_ARCH__)
#if defined(__clang__) || defined(__GNUC__)
  return __builtin_bit_cast(float, bits);
#else
  union {
    std::uint32_t bits;
    float value;
  } result{bits};
  return result.value;
#endif
#else
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
#endif
}

// IEEE binary16 to binary32 conversion expressed only in fixed-width integer
// operations. This keeps the header usable in ordinary C++, CUDA, HIP, and
// runtime-free generated device translation units.
LIBGGUF_SIMPLE_Q_INLINE float fp16_bits_to_float(std::uint16_t value) {
  const std::uint32_t sign =
      static_cast<std::uint32_t>(value & 0x8000u) << 16;
  std::uint32_t exponent = (value >> 10) & 0x1fu;
  std::uint32_t mantissa = value & 0x03ffu;

  if (exponent == 0) {
    if (mantissa == 0) return float_from_bits(sign);
    std::int32_t unbiased_exponent = -14;
    while ((mantissa & 0x0400u) == 0) {
      mantissa <<= 1;
      --unbiased_exponent;
    }
    mantissa &= 0x03ffu;
    const std::uint32_t fp32_exponent =
        static_cast<std::uint32_t>(unbiased_exponent + 127);
    return float_from_bits(sign | (fp32_exponent << 23) | (mantissa << 13));
  }

  if (exponent == 0x1fu) {
    return float_from_bits(sign | 0x7f800000u | (mantissa << 13));
  }

  exponent += 127u - 15u;
  return float_from_bits(sign | (exponent << 23) | (mantissa << 13));
}

}  // namespace detail

LIBGGUF_SIMPLE_Q_INLINE float q4_0_scale(const q4_0_block& block) {
  return detail::fp16_bits_to_float(block.scale_bits);
}

// Raw-wire accessors avoid materializing a typed block when a device kernel
// consumes only one fragment. encoded points to at least q4_0_block_bytes.
LIBGGUF_SIMPLE_Q_INLINE float q4_0_raw_scale(const std::byte* encoded) {
  const std::uint16_t bits =
      static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(encoded[0])) |
      static_cast<std::uint16_t>(
          static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(encoded[1]))
          << 8);
  return detail::fp16_bits_to_float(bits);
}

LIBGGUF_SIMPLE_Q_INLINE int q4_0_raw_quant(const std::byte* encoded,
                                          std::uint32_t value_index) {
  const std::uint8_t packed =
      std::to_integer<std::uint8_t>(encoded[2 + (value_index & 15u)]);
  const int quant = value_index < 16u ? static_cast<int>(packed & 0x0fu)
                                     : static_cast<int>(packed >> 4);
  return quant - 8;
}

// Preconditions: value_index < simple_q_block_values.
LIBGGUF_SIMPLE_Q_INLINE int q4_0_quant(const q4_0_block& block,
                                      std::uint32_t value_index) {
  const std::uint8_t packed = block.quants[value_index & 15u];
  const int quant = value_index < 16u ? static_cast<int>(packed & 0x0fu)
                                     : static_cast<int>(packed >> 4);
  return quant - 8;
}

LIBGGUF_SIMPLE_Q_INLINE float decode_q4_0(const q4_0_block& block,
                                         std::uint32_t value_index) {
  return q4_0_scale(block) * static_cast<float>(q4_0_quant(block, value_index));
}

// Preconditions: value_index <= simple_q_block_values - 8 and output points
// to at least eight floats. Encoded block addresses must be two-byte aligned.
LIBGGUF_SIMPLE_Q_INLINE void decode_q4_0_8(const q4_0_block& block,
                                          std::uint32_t value_index,
                                          float* output) {
  const float scale = q4_0_scale(block);
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i) {
    output[i] = scale * static_cast<float>(q4_0_quant(block, value_index + i));
  }
}

LIBGGUF_SIMPLE_Q_INLINE float q8_0_scale(const q8_0_block& block) {
  return detail::fp16_bits_to_float(block.scale_bits);
}

// encoded points to at least q8_0_block_bytes.
LIBGGUF_SIMPLE_Q_INLINE float q8_0_raw_scale(const std::byte* encoded) {
  const std::uint16_t bits =
      static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(encoded[0])) |
      static_cast<std::uint16_t>(
          static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(encoded[1]))
          << 8);
  return detail::fp16_bits_to_float(bits);
}

LIBGGUF_SIMPLE_Q_INLINE int q8_0_raw_quant(const std::byte* encoded,
                                          std::uint32_t value_index) {
  return static_cast<int>(static_cast<std::int8_t>(
      std::to_integer<std::uint8_t>(encoded[2 + value_index])));
}

// Preconditions: value_index < simple_q_block_values.
LIBGGUF_SIMPLE_Q_INLINE int q8_0_quant(const q8_0_block& block,
                                      std::uint32_t value_index) {
  return static_cast<int>(block.quants[value_index]);
}

LIBGGUF_SIMPLE_Q_INLINE float decode_q8_0(const q8_0_block& block,
                                         std::uint32_t value_index) {
  return q8_0_scale(block) * static_cast<float>(q8_0_quant(block, value_index));
}

// Preconditions: value_index <= simple_q_block_values - 8 and output points
// to at least eight floats. Encoded block addresses must be two-byte aligned.
LIBGGUF_SIMPLE_Q_INLINE void decode_q8_0_8(const q8_0_block& block,
                                          std::uint32_t value_index,
                                          float* output) {
  const float scale = q8_0_scale(block);
#pragma unroll
  for (std::uint32_t i = 0; i < 8; ++i) {
    output[i] = scale *
                static_cast<float>(q8_0_quant(block, value_index + i));
  }
}

}  // namespace libgguf::device

#undef LIBGGUF_SIMPLE_Q_INLINE
