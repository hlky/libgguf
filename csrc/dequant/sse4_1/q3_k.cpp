#include "common/libgguf_common.h"

#if defined(__SSE4_1__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <smmintrin.h>
#define LIBGGUF_BUILD_SSE41 1
#endif

static inline void libgguf_q3_k_store16_sse41(float *RESTRICT y, __m128i q8, __m128 d)
{
  _mm_storeu_ps(y + 0, _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(q8)), d));
  _mm_storeu_ps(y + 4, _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(q8, 4))), d));
  _mm_storeu_ps(y + 8, _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(q8, 8))), d));
  _mm_storeu_ps(y + 12, _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(q8, 12))), d));
}

extern "C" void dequantize_row_q3_K_sse4_1(const block_q3_K *RESTRICT x, float *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_SSE41)
  assert(k % QK_K == 0);
  static_assert(QK_K == 256, "QK_K must be 256");

  const int nb = (int)(k / QK_K);
  const __m128i mask = _mm_set1_epi8(3);
  const __m128i four = _mm_set1_epi8(4);
  const __m128i zero = _mm_setzero_si128();
  const uint32_t kmask1 = 0x03030303;
  const uint32_t kmask2 = 0x0f0f0f0f;

  for (int i = 0; i < nb; ++i)
  {
    const float d_all = GGML_FP16_TO_FP32(x[i].d);
    const uint8_t *RESTRICT q = x[i].qs;
    const uint8_t *RESTRICT hm = x[i].hmask;
    float *RESTRICT yb = y + i * QK_K;
    uint32_t aux[4];
    memcpy(aux, x[i].scales, 12);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t *RESTRICT scales = (const int8_t *)aux;

    int is = 0;
    uint8_t m = 1;
    for (int n = 0; n < QK_K; n += 128)
    {
      for (int shift = 0; shift < 8; shift += 2)
      {
        const __m128 d1 = _mm_set1_ps(d_all * (float)(scales[is++] - 32));
        const __m128i h1 = _mm_and_si128(_mm_loadu_si128((const __m128i *)(hm + 0)), _mm_set1_epi8((char)m));
        const __m128i sub1 = _mm_and_si128(_mm_cmpeq_epi8(h1, zero), four);
        const __m128i q1 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(_mm_loadu_si128((const __m128i *)(q + 0)), shift), mask), sub1);
        libgguf_q3_k_store16_sse41(yb + n + (shift / 2) * 32 + 0, q1, d1);

        const __m128 d2 = _mm_set1_ps(d_all * (float)(scales[is++] - 32));
        const __m128i h2 = _mm_and_si128(_mm_loadu_si128((const __m128i *)(hm + 16)), _mm_set1_epi8((char)m));
        const __m128i sub2 = _mm_and_si128(_mm_cmpeq_epi8(h2, zero), four);
        const __m128i q2 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(_mm_loadu_si128((const __m128i *)(q + 16)), shift), mask), sub2);
        libgguf_q3_k_store16_sse41(yb + n + (shift / 2) * 32 + 16, q2, d2);

        m <<= 1;
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
