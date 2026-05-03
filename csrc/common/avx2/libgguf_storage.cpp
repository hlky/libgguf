#include "libgguf_storage.h"

#if defined(__AVX2__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
#include <immintrin.h>

static inline __m256i libgguf_bf16_from_fp32_8_avx2(__m256 x)
{
  const __m256i bits = _mm256_castps_si256(x);
  const __m256i abs_bits = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffffff));
  const __m256i nan_mask = _mm256_cmpgt_epi32(abs_bits, _mm256_set1_epi32(0x7f800000));
  const __m256i high = _mm256_srli_epi32(bits, 16);
  const __m256i rounding = _mm256_add_epi32(_mm256_set1_epi32(0x7fff), _mm256_and_si256(high, _mm256_set1_epi32(1)));
  const __m256i rounded = _mm256_srli_epi32(_mm256_add_epi32(bits, rounding), 16);
  const __m256i nan = _mm256_or_si256(high, _mm256_set1_epi32(64));
  return _mm256_or_si256(_mm256_and_si256(nan_mask, nan), _mm256_andnot_si256(nan_mask, rounded));
}

extern "C" void libgguf_store_bf16_avx2(const float *src, ggml_bf16_t *dst, size_t n)
{
  size_t i = 0;
  for (; i + 8 <= n; i += 8)
  {
    const __m256i bf16 = libgguf_bf16_from_fp32_8_avx2(_mm256_loadu_ps(src + i));
    const __m256i packed = _mm256_packus_epi32(bf16, bf16);
    _mm_storel_epi64((__m128i *)(dst + i), _mm256_castsi256_si128(packed));
    _mm_storel_epi64((__m128i *)(dst + i + 4), _mm256_extracti128_si256(packed, 1));
  }
  for (; i < n; ++i)
  {
    dst[i] = GGML_FP32_TO_BF16(src[i]);
  }
}

#else

extern "C" void libgguf_store_bf16_avx2(const float *src, ggml_bf16_t *dst, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    dst[i] = GGML_FP32_TO_BF16(src[i]);
  }
}

#endif
