#include "common/libgguf_common.h"

#if defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(__SSE2__)
#include <emmintrin.h>
#define LIBGGUF_BUILD_SSE2 1
#endif

static inline void libgguf_tq1_0_store16_sse2(float *RESTRICT y, __m128i q8, __m128 d)
{
  const __m128i zero = _mm_setzero_si128();
  const __m128i sign = _mm_cmpgt_epi8(zero, q8);
  const __m128i q16a = _mm_unpacklo_epi8(q8, sign);
  const __m128i q16b = _mm_unpackhi_epi8(q8, sign);
  const __m128i s16a = _mm_cmpgt_epi16(zero, q16a);
  const __m128i s16b = _mm_cmpgt_epi16(zero, q16b);
  _mm_storeu_ps(y + 0, _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(q16a, s16a)), d));
  _mm_storeu_ps(y + 4, _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(q16a, s16a)), d));
  _mm_storeu_ps(y + 8, _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(q16b, s16b)), d));
  _mm_storeu_ps(y + 12, _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(q16b, s16b)), d));
}

static inline void libgguf_tq1_0_store4_sse2(float *RESTRICT y, __m128i q8, __m128 d)
{
  const __m128i zero = _mm_setzero_si128();
  const __m128i sign = _mm_cmpgt_epi8(zero, q8);
  const __m128i q16 = _mm_unpacklo_epi8(q8, sign);
  const __m128i s16 = _mm_cmpgt_epi16(zero, q16);
  _mm_storeu_ps(y, _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(q16, s16)), d));
}

static inline __m128i libgguf_tq1_0_digits16_sse2(__m128i bytes, int pow3)
{
  const __m128i zero = _mm_setzero_si128();
  const __m128i mult = _mm_set1_epi16((short)pow3);
  const __m128i three = _mm_set1_epi16(3);
  const __m128i one = _mm_set1_epi16(1);
  __m128i lo = _mm_mullo_epi16(_mm_unpacklo_epi8(bytes, zero), mult);
  __m128i hi = _mm_mullo_epi16(_mm_unpackhi_epi8(bytes, zero), mult);
  lo = _mm_srli_epi16(_mm_mullo_epi16(_mm_and_si128(lo, _mm_set1_epi16(0x00ff)), three), 8);
  hi = _mm_srli_epi16(_mm_mullo_epi16(_mm_and_si128(hi, _mm_set1_epi16(0x00ff)), three), 8);
  return _mm_packs_epi16(_mm_sub_epi16(lo, one), _mm_sub_epi16(hi, one));
}

extern "C" void dequantize_row_tq1_0_sse2(const block_tq1_0 *RESTRICT x, float *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_SSE2)
  assert(k % QK_K == 0);
  static_assert(QK_K == 256, "QK_K must be 256");

  const int nb = (int)(k / QK_K);
  const int pow3[5] = {1, 3, 9, 27, 81};
  const int qh_pow3[4] = {1, 3, 9, 27};

  for (int i = 0; i < nb; ++i)
  {
    const __m128 d = _mm_set1_ps(GGML_FP16_TO_FP32(x[i].d));
    float *RESTRICT yb = y + i * QK_K;
    int out = 0;

    for (int j = 0; j < 32; j += 32)
    {
      const __m128i p0 = _mm_loadu_si128((const __m128i *)(x[i].qs + j + 0));
      const __m128i p1 = _mm_loadu_si128((const __m128i *)(x[i].qs + j + 16));
      for (int n = 0; n < 5; ++n)
      {
        libgguf_tq1_0_store16_sse2(yb + out + 0, libgguf_tq1_0_digits16_sse2(p0, pow3[n]), d);
        libgguf_tq1_0_store16_sse2(yb + out + 16, libgguf_tq1_0_digits16_sse2(p1, pow3[n]), d);
        out += 32;
      }
    }

    const __m128i tail = _mm_loadu_si128((const __m128i *)(x[i].qs + 32));
    for (int n = 0; n < 5; ++n)
    {
      libgguf_tq1_0_store16_sse2(yb + out, libgguf_tq1_0_digits16_sse2(tail, pow3[n]), d);
      out += 16;
    }

    uint8_t high[16] = {};
    memcpy(high, x[i].qh, sizeof(x->qh));
    const __m128i qh = _mm_loadu_si128((const __m128i *)high);
    for (int n = 0; n < 4; ++n)
    {
      libgguf_tq1_0_store4_sse2(yb + out, libgguf_tq1_0_digits16_sse2(qh, qh_pow3[n]), d);
      out += 4;
    }
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
