#include "common/libgguf_common.h"

#if defined(__SSE4_1__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <smmintrin.h>
#define LIBGGUF_BUILD_SSE41 1
#endif

static inline void libgguf_tq2_0_store16_sse41(float *RESTRICT y, __m128i q8, __m128 d)
{
  _mm_storeu_ps(y + 0, _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(q8)), d));
  _mm_storeu_ps(y + 4, _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(q8, 4))), d));
  _mm_storeu_ps(y + 8, _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(q8, 8))), d));
  _mm_storeu_ps(y + 12, _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(q8, 12))), d));
}

extern "C" void dequantize_row_tq2_0_sse4_1(const block_tq2_0 *RESTRICT x, float *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_SSE41)
  assert(k % QK_K == 0);
  static_assert(QK_K == 256, "QK_K must be 256");

  const int nb = (int)(k / QK_K);
  const __m128i mask = _mm_set1_epi8(3);
  const __m128i one = _mm_set1_epi8(1);

  for (int i = 0; i < nb; ++i)
  {
    const __m128 d = _mm_set1_ps(GGML_FP16_TO_FP32(x[i].d));
    float *RESTRICT yb = y + i * QK_K;

    for (int j = 0; j < (int)sizeof(x->qs); j += 32)
    {
      const __m128i p0 = _mm_loadu_si128((const __m128i *)(x[i].qs + j + 0));
      const __m128i p1 = _mm_loadu_si128((const __m128i *)(x[i].qs + j + 16));
      for (int shift = 0; shift < 8; shift += 2)
      {
        const __m128i q0 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(p0, shift), mask), one);
        const __m128i q1 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(p1, shift), mask), one);
        const int out = j * 4 + (shift / 2) * 32;
        libgguf_tq2_0_store16_sse41(yb + out + 0, q0, d);
        libgguf_tq2_0_store16_sse41(yb + out + 16, q1, d);
      }
    }
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
