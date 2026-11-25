#pragma once

#include <libgguf/common/formats.h>
#include <libgguf/common/types.h>
#include <cstdint>

namespace libgguf { namespace cpu {

// Header-only block readers for bounded CPU consumers. The caller supplies its
// existing half conversion; payload fields are read explicitly little endian.
template <ggml_type Format> struct BlockDecoder;

template <> struct BlockDecoder<GGML_TYPE_PQ2_0>
{
  static constexpr int values_per_block = 128;
  static constexpr int bytes_per_block = 34;
  template <typename HalfToFloat>
  static void decode(const void *data, float *out, HalfToFloat half_to_float)
  {
    const auto *p = static_cast<const uint8_t *>(data);
    const float d = half_to_float(uint16_t(p[0]) | (uint16_t(p[1]) << 8));
    for (int j = 0; j < 128; ++j)
      out[j] = (int((p[2 + j / 4] >> (2 * (j % 4))) & 3) - 1) * d;
  }
};

template <> struct BlockDecoder<GGML_TYPE_PTQ1_0>
{
  static constexpr int values_per_block = 128;
  static constexpr int bytes_per_block = 28;
  template <typename HalfToFloat>
  static void decode(const void *data, float *out, HalfToFloat half_to_float)
  {
    const auto *p = static_cast<const uint8_t *>(data);
    const float d = half_to_float(uint16_t(p[26]) | (uint16_t(p[27]) << 8));
    const uint8_t pow3[5] = {1, 3, 9, 27, 81};
    for (int stage = 0; stage < 3; ++stage)
    {
      const int width = stage == 0 ? 16 : stage == 1 ? 8 : 2;
      const int start = stage == 0 ? 0 : stage == 1 ? 16 : 24;
      const int planes = stage == 2 ? 4 : 5;
      for (int plane = 0; plane < planes; ++plane)
        for (int m = 0; m < width; ++m)
        {
          const uint8_t q = p[start + m] * pow3[plane];
          *out++ = ((int(q) * 3 >> 8) - 1) * d;
        }
    }
  }
};

template <> struct BlockDecoder<GGML_TYPE_Q8_0>
{
  static constexpr int values_per_block = 32;
  static constexpr int bytes_per_block = 34;
  template <typename HalfToFloat>
  static void decode(const void *data, float *out, HalfToFloat half_to_float)
  {
    const auto *p = static_cast<const uint8_t *>(data);
    const float d = half_to_float(uint16_t(p[0]) | (uint16_t(p[1]) << 8));
    for (int j = 0; j < 32; ++j)
      out[j] = (p[2 + j] < 128 ? int(p[2 + j]) : int(p[2 + j]) - 256) * d;
  }
};

} }
