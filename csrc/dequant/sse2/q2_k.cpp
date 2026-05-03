#include "common/libgguf_common.h"

#if defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(__SSE2__)
#include <emmintrin.h>
#define LIBGGUF_BUILD_SSE2 1
#endif

static inline void libgguf_q2_k_store16_sse2(float *RESTRICT y, __m128i q8, __m128 d, __m128 m)
{
  const __m128i zero = _mm_setzero_si128();
  const __m128i q16a = _mm_unpacklo_epi8(q8, zero);
  const __m128i q16b = _mm_unpackhi_epi8(q8, zero);
  const __m128i q32a = _mm_unpacklo_epi16(q16a, zero);
  const __m128i q32b = _mm_unpackhi_epi16(q16a, zero);
  const __m128i q32c = _mm_unpacklo_epi16(q16b, zero);
  const __m128i q32d = _mm_unpackhi_epi16(q16b, zero);
  _mm_storeu_ps(y + 0, _mm_sub_ps(_mm_mul_ps(_mm_cvtepi32_ps(q32a), d), m));
  _mm_storeu_ps(y + 4, _mm_sub_ps(_mm_mul_ps(_mm_cvtepi32_ps(q32b), d), m));
  _mm_storeu_ps(y + 8, _mm_sub_ps(_mm_mul_ps(_mm_cvtepi32_ps(q32c), d), m));
  _mm_storeu_ps(y + 12, _mm_sub_ps(_mm_mul_ps(_mm_cvtepi32_ps(q32d), d), m));
}

extern "C" void dequantize_row_q2_K_sse2(const block_q2_K *RESTRICT x, float *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_SSE2)
  assert(k % QK_K == 0);
  static_assert(QK_K == 256, "QK_K must be 256");

  const int nb = (int)(k / QK_K);
  const __m128i mask = _mm_set1_epi8(3);

  for (int i = 0; i < nb; ++i)
  {
    const float d_all = GGML_FP16_TO_FP32(x[i].d);
    const float min_all = GGML_FP16_TO_FP32(x[i].dmin);
    const uint8_t *RESTRICT q = x[i].qs;
    float *RESTRICT yb = y + i * QK_K;

    int is = 0;
    for (int n = 0; n < QK_K; n += 128)
    {
      for (int shift = 0; shift < 8; shift += 2)
      {
        uint8_t sc = x[i].scales[is++];
        const __m128 d1 = _mm_set1_ps(d_all * (float)(sc & 0x0f));
        const __m128 m1 = _mm_set1_ps(min_all * (float)(sc >> 4));
        const __m128i q1 = _mm_and_si128(_mm_srli_epi16(_mm_loadu_si128((const __m128i *)(q + 0)), shift), mask);
        libgguf_q2_k_store16_sse2(yb + n + (shift / 2) * 32 + 0, q1, d1, m1);

        sc = x[i].scales[is++];
        const __m128 d2 = _mm_set1_ps(d_all * (float)(sc & 0x0f));
        const __m128 m2 = _mm_set1_ps(min_all * (float)(sc >> 4));
        const __m128i q2 = _mm_and_si128(_mm_srli_epi16(_mm_loadu_si128((const __m128i *)(q + 16)), shift), mask);
        libgguf_q2_k_store16_sse2(yb + n + (shift / 2) * 32 + 16, q2, d2, m2);
      }
      q += 32;
    }
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
