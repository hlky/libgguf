#include "libgguf_storage.h"

#if defined(__SSE4_1__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
#include <smmintrin.h>

static inline __m128i libgguf_bf16_from_fp32_4_sse4_1(__m128 x)
{
  const __m128i bits = _mm_castps_si128(x);
  const __m128i abs_bits = _mm_and_si128(bits, _mm_set1_epi32(0x7fffffff));
  const __m128i nan_mask = _mm_cmpgt_epi32(abs_bits, _mm_set1_epi32(0x7f800000));
  const __m128i high = _mm_srli_epi32(bits, 16);
  const __m128i rounding = _mm_add_epi32(_mm_set1_epi32(0x7fff), _mm_and_si128(high, _mm_set1_epi32(1)));
  const __m128i rounded = _mm_srli_epi32(_mm_add_epi32(bits, rounding), 16);
  const __m128i nan = _mm_or_si128(high, _mm_set1_epi32(64));
  return _mm_or_si128(_mm_and_si128(nan_mask, nan), _mm_andnot_si128(nan_mask, rounded));
}

extern "C" void libgguf_store_bf16_sse4_1(const float *src, ggml_bf16_t *dst, size_t n)
{
  size_t i = 0;
  for (; i + 4 <= n; i += 4)
  {
    const __m128i bf16 = libgguf_bf16_from_fp32_4_sse4_1(_mm_loadu_ps(src + i));
    _mm_storel_epi64((__m128i *)(dst + i), _mm_packus_epi32(bf16, bf16));
  }
  for (; i < n; ++i)
  {
    dst[i] = GGML_FP32_TO_BF16(src[i]);
  }
}

#else

extern "C" void libgguf_store_bf16_sse4_1(const float *src, ggml_bf16_t *dst, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    dst[i] = GGML_FP32_TO_BF16(src[i]);
  }
}

#endif
