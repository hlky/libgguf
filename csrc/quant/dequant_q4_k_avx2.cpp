#include "common/libgguf_common.h"

#if defined(__AVX2__)
#include <immintrin.h>
#define LIBGGUF_BUILD_AVX2 1
#endif

static inline void libgguf_q4_k_store16_avx2(float *RESTRICT y, __m128i q8, __m256 d, __m256 m)
{
  const __m256i q16 = _mm256_cvtepu8_epi16(q8);
  const __m256i q32a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(q16));
  const __m256i q32b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(q16, 1));
  _mm256_storeu_ps(y + 0, _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q32a), d), m));
  _mm256_storeu_ps(y + 8, _mm256_sub_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(q32b), d), m));
}

extern "C" void dequantize_row_q4_K_avx2(const block_q4_K *RESTRICT x, float *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_AVX2)
  assert(k % QK_K == 0);
  static_assert(QK_K == 256, "QK_K must be 256");

  const int nb = (int)(k / QK_K);
  const __m128i low_mask = _mm_set1_epi8(0x0f);

  for (int i = 0; i < nb; ++i)
  {
    const float d_all = GGML_FP16_TO_FP32(x[i].d);
    const float min_all = GGML_FP16_TO_FP32(x[i].dmin);
    const uint8_t *RESTRICT q = x[i].qs;
    float *RESTRICT yb = y + i * QK_K;

    int is = 0;
    for (int j = 0; j < QK_K; j += 64)
    {
      uint8_t sc, minv;
      get_scale_min_k4(is + 0, x[i].scales, &sc, &minv);
      const __m256 d1 = _mm256_set1_ps(d_all * sc);
      const __m256 m1 = _mm256_set1_ps(min_all * minv);
      get_scale_min_k4(is + 1, x[i].scales, &sc, &minv);
      const __m256 d2 = _mm256_set1_ps(d_all * sc);
      const __m256 m2 = _mm256_set1_ps(min_all * minv);

      const __m128i packed0 = _mm_loadu_si128((const __m128i *)(q + 0));
      const __m128i packed1 = _mm_loadu_si128((const __m128i *)(q + 16));
      const __m128i lo0 = _mm_and_si128(packed0, low_mask);
      const __m128i lo1 = _mm_and_si128(packed1, low_mask);
      const __m128i hi0 = _mm_and_si128(_mm_srli_epi16(packed0, 4), low_mask);
      const __m128i hi1 = _mm_and_si128(_mm_srli_epi16(packed1, 4), low_mask);

      libgguf_q4_k_store16_avx2(yb + j + 0, lo0, d1, m1);
      libgguf_q4_k_store16_avx2(yb + j + 16, lo1, d1, m1);
      libgguf_q4_k_store16_avx2(yb + j + 32, hi0, d2, m2);
      libgguf_q4_k_store16_avx2(yb + j + 48, hi1, d2, m2);

      q += 32;
      is += 2;
    }
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
