#include "common/libgguf_common.h"

#if defined(_MSC_VER)
#include <smmintrin.h>
#define LIBGGUF_BUILD_SSE4_1 1
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#define LIBGGUF_BUILD_SSE4_1 1
#endif

static inline void libgguf_q5_k_store16_sse4_1(float *RESTRICT y, __m128i q8, __m128 d, __m128 m)
{
  const __m128i q16a = _mm_cvtepu8_epi16(q8);
  const __m128i q16b = _mm_cvtepu8_epi16(_mm_srli_si128(q8, 8));
  const __m128i q32a = _mm_cvtepi16_epi32(q16a);
  const __m128i q32b = _mm_cvtepi16_epi32(_mm_srli_si128(q16a, 8));
  const __m128i q32c = _mm_cvtepi16_epi32(q16b);
  const __m128i q32d = _mm_cvtepi16_epi32(_mm_srli_si128(q16b, 8));
  _mm_storeu_ps(y + 0, _mm_sub_ps(_mm_mul_ps(_mm_cvtepi32_ps(q32a), d), m));
  _mm_storeu_ps(y + 4, _mm_sub_ps(_mm_mul_ps(_mm_cvtepi32_ps(q32b), d), m));
  _mm_storeu_ps(y + 8, _mm_sub_ps(_mm_mul_ps(_mm_cvtepi32_ps(q32c), d), m));
  _mm_storeu_ps(y + 12, _mm_sub_ps(_mm_mul_ps(_mm_cvtepi32_ps(q32d), d), m));
}

extern "C" void dequantize_row_q5_K_sse4_1(const block_q5_K *RESTRICT x, float *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_SSE4_1)
  assert(k % QK_K == 0);
  static_assert(QK_K == 256, "QK_K must be 256");

  const int nb = (int)(k / QK_K);
  const __m128i low_mask = _mm_set1_epi8(0x0f);

  for (int i = 0; i < nb; ++i)
  {
    const float d_all = GGML_FP16_TO_FP32(x[i].d);
    const float min_all = GGML_FP16_TO_FP32(x[i].dmin);
    const uint8_t *RESTRICT ql = x[i].qs;
    const uint8_t *RESTRICT qh = x[i].qh;
    float *RESTRICT yb = y + i * QK_K;

    int is = 0;
    uint8_t u1 = 1;
    uint8_t u2 = 2;
    for (int j = 0; j < QK_K; j += 64)
    {
      uint8_t high_lo[32];
      uint8_t high_hi[32];
      for (int l = 0; l < 32; ++l)
      {
        high_lo[l] = (qh[l] & u1) ? 16 : 0;
        high_hi[l] = (qh[l] & u2) ? 16 : 0;
      }

      uint8_t sc, minv;
      get_scale_min_k4(is + 0, x[i].scales, &sc, &minv);
      const __m128 d1 = _mm_set1_ps(d_all * sc);
      const __m128 m1 = _mm_set1_ps(min_all * minv);
      get_scale_min_k4(is + 1, x[i].scales, &sc, &minv);
      const __m128 d2 = _mm_set1_ps(d_all * sc);
      const __m128 m2 = _mm_set1_ps(min_all * minv);

      const __m128i packed0 = _mm_loadu_si128((const __m128i *)(ql + 0));
      const __m128i packed1 = _mm_loadu_si128((const __m128i *)(ql + 16));
      const __m128i lo0 = _mm_or_si128(_mm_and_si128(packed0, low_mask), _mm_loadu_si128((const __m128i *)(high_lo + 0)));
      const __m128i lo1 = _mm_or_si128(_mm_and_si128(packed1, low_mask), _mm_loadu_si128((const __m128i *)(high_lo + 16)));
      const __m128i hi0 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(packed0, 4), low_mask), _mm_loadu_si128((const __m128i *)(high_hi + 0)));
      const __m128i hi1 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(packed1, 4), low_mask), _mm_loadu_si128((const __m128i *)(high_hi + 16)));

      libgguf_q5_k_store16_sse4_1(yb + j + 0, lo0, d1, m1);
      libgguf_q5_k_store16_sse4_1(yb + j + 16, lo1, d1, m1);
      libgguf_q5_k_store16_sse4_1(yb + j + 32, hi0, d2, m2);
      libgguf_q5_k_store16_sse4_1(yb + j + 48, hi1, d2, m2);

      ql += 32;
      is += 2;
      u1 <<= 2;
      u2 <<= 2;
    }
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
