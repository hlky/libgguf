#include "libgguf_storage.h"

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>

static inline __m128i libgguf_bf16_from_fp32_4_sse2(__m128 x)
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

static inline void libgguf_store_bf16_4_sse2(ggml_bf16_t *dst, __m128i values)
{
  const __m128i lo16 = _mm_and_si128(values, _mm_set1_epi32(0xffff));
  const __m128i low_pair = _mm_and_si128(
      _mm_shufflelo_epi16(lo16, _MM_SHUFFLE(0, 0, 2, 0)),
      _mm_set_epi32(0, 0, 0, -1));
  const __m128i high_pair = _mm_slli_si128(
      _mm_shufflelo_epi16(_mm_srli_si128(lo16, 8), _MM_SHUFFLE(0, 0, 2, 0)),
      4);
  _mm_storel_epi64((__m128i *)dst, _mm_or_si128(low_pair, high_pair));
}

extern "C" void libgguf_store_bf16_sse2(const float *src, ggml_bf16_t *dst, size_t n)
{
  size_t i = 0;
  for (; i + 4 <= n; i += 4)
  {
    libgguf_store_bf16_4_sse2(dst + i, libgguf_bf16_from_fp32_4_sse2(_mm_loadu_ps(src + i)));
  }
  for (; i < n; ++i)
  {
    dst[i] = GGML_FP32_TO_BF16(src[i]);
  }
}

#else

extern "C" void libgguf_store_bf16_sse2(const float *src, ggml_bf16_t *dst, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    dst[i] = GGML_FP32_TO_BF16(src[i]);
  }
}

#endif
