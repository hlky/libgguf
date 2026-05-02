#include "common/libgguf_common.h"

#if defined(__SSE4_1__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <smmintrin.h>
#define LIBGGUF_BUILD_SSE41 1
#endif

static inline void libgguf_q6_k_store16_sse41(float *RESTRICT y, __m128i q8, __m128 d)
{
  _mm_storeu_ps(y + 0, _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(q8)), d));
  _mm_storeu_ps(y + 4, _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(q8, 4))), d));
  _mm_storeu_ps(y + 8, _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(q8, 8))), d));
  _mm_storeu_ps(y + 12, _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(q8, 12))), d));
}

static inline __m128i libgguf_q6_k_vals_sse41(const uint8_t *RESTRICT ql, const uint8_t *RESTRICT qh, int ql_shift, int qh_shift)
{
  const __m128i low_mask = _mm_set1_epi8(0x0f);
  const __m128i high_mask = _mm_set1_epi8(0x30);
  const __m128i bias = _mm_set1_epi8(32);
  const __m128i l = ql_shift == 0 ? _mm_and_si128(_mm_loadu_si128((const __m128i *)ql), low_mask) : _mm_and_si128(_mm_srli_epi16(_mm_loadu_si128((const __m128i *)ql), 4), low_mask);
  const __m128i h = _mm_and_si128(_mm_slli_epi16(_mm_srli_epi16(_mm_loadu_si128((const __m128i *)qh), qh_shift), 4), high_mask);
  return _mm_sub_epi8(_mm_or_si128(l, h), bias);
}

extern "C" void dequantize_row_q6_K_sse4_1(const block_q6_K *RESTRICT x, float *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_SSE41)
  assert(k % QK_K == 0);
  static_assert(QK_K == 256, "QK_K must be 256");

  const int nb = (int)(k / QK_K);
  for (int i = 0; i < nb; ++i)
  {
    const float d_all = GGML_FP16_TO_FP32(x[i].d);
    const uint8_t *RESTRICT ql = x[i].ql;
    const uint8_t *RESTRICT qh = x[i].qh;
    const int8_t *RESTRICT sc = x[i].scales;
    float *RESTRICT yb = y + i * QK_K;

    for (int n = 0; n < QK_K; n += 128)
    {
      for (int l = 0; l < 32; l += 16)
      {
        const int is = l / 16;
        libgguf_q6_k_store16_sse41(yb + n + l + 0, libgguf_q6_k_vals_sse41(ql + l + 0, qh + l, 0, 0), _mm_set1_ps(d_all * (float)sc[is + 0]));
        libgguf_q6_k_store16_sse41(yb + n + l + 32, libgguf_q6_k_vals_sse41(ql + l + 32, qh + l, 0, 2), _mm_set1_ps(d_all * (float)sc[is + 2]));
        libgguf_q6_k_store16_sse41(yb + n + l + 64, libgguf_q6_k_vals_sse41(ql + l + 0, qh + l, 4, 4), _mm_set1_ps(d_all * (float)sc[is + 4]));
        libgguf_q6_k_store16_sse41(yb + n + l + 96, libgguf_q6_k_vals_sse41(ql + l + 32, qh + l, 4, 6), _mm_set1_ps(d_all * (float)sc[is + 6]));
      }
      ql += 64;
      qh += 32;
      sc += 8;
    }
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
