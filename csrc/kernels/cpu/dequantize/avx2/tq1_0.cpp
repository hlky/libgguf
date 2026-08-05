#include "kernels/cpu/common/common.h"
#include "kernels/cpu/dequantize/kernels.h"
#include "quantize/iq_tables.h"


#include <immintrin.h>

static inline void libgguf_tq1_0_store16(float *LIBGGUF_RESTRICT y, __m128i q8, __m256 d)
{
  _mm256_storeu_ps(y + 0, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8)), d));
  _mm256_storeu_ps(y + 8, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q8, 8))), d));
}

static inline void libgguf_tq1_0_store4(float *LIBGGUF_RESTRICT y, __m128i q8, __m128 d)
{
  _mm_storeu_ps(y, _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(q8)), d));
}

static inline __m128i libgguf_tq1_0_digits16(__m128i bytes, int pow3)
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

extern "C" void dequantize_block_tq1_0(const block_tq1_0 *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK_K == 0);
  static_assert(QK_K == 256, "QK_K must be 256");

  const int nb = (int)(k / QK_K);
  const int pow3[5] = {1, 3, 9, 27, 81};
  const int qh_pow3[4] = {1, 3, 9, 27};

  for (int i = 0; i < nb; ++i)
  {
    const float df = LIBGGUF_FP16_TO_FP32(x[i].d);
    const __m256 d = _mm256_set1_ps(df);
    const __m128 d4 = _mm_set1_ps(df);
    float *LIBGGUF_RESTRICT yb = y + i * QK_K;
    int out = 0;

    const __m128i p0 = _mm_loadu_si128((const __m128i *)(x[i].qs + 0));
    const __m128i p1 = _mm_loadu_si128((const __m128i *)(x[i].qs + 16));
    for (int n = 0; n < 5; ++n)
    {
      libgguf_tq1_0_store16(yb + out + 0, libgguf_tq1_0_digits16(p0, pow3[n]), d);
      libgguf_tq1_0_store16(yb + out + 16, libgguf_tq1_0_digits16(p1, pow3[n]), d);
      out += 32;
    }

    const __m128i tail = _mm_loadu_si128((const __m128i *)(x[i].qs + 32));
    for (int n = 0; n < 5; ++n)
    {
      libgguf_tq1_0_store16(yb + out, libgguf_tq1_0_digits16(tail, pow3[n]), d);
      out += 16;
    }

    uint8_t high[16] = {};
    memcpy(high, x[i].qh, sizeof(x->qh));
    const __m128i qh = _mm_loadu_si128((const __m128i *)high);
    for (int n = 0; n < 4; ++n)
    {
      libgguf_tq1_0_store4(yb + out, libgguf_tq1_0_digits16(qh, qh_pow3[n]), d4);
      out += 4;
    }
  }
}


void dequantize_tq1_0(const block_tq1_0 *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  dequantize_block_tq1_0(x, y, k);
}

