#include "kernels/cpu/common/common.h"
#include "kernels/cpu/dequantize/kernels.h"
#include "quantize/iq_tables.h"

#include "kernels/cpu/common/tables.h"

#include <emmintrin.h>

static inline void libgguf_iq1_store8(float *LIBGGUF_RESTRICT y, const int8_t *LIBGGUF_RESTRICT grid, __m128 dl, __m128 delta)
{
  const __m128i zero = _mm_setzero_si128();
  const __m128i g8 = _mm_loadl_epi64((const __m128i *)grid);
  const __m128i sign8 = _mm_cmpgt_epi8(zero, g8);
  const __m128i g16 = _mm_unpacklo_epi8(g8, sign8);
  const __m128i sign16 = _mm_cmpgt_epi16(zero, g16);
  const __m128 f0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(g16, sign16));
  const __m128 f1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(g16, sign16));
  _mm_storeu_ps(y + 0, _mm_mul_ps(_mm_add_ps(f0, delta), dl));
  _mm_storeu_ps(y + 4, _mm_mul_ps(_mm_add_ps(f1, delta), dl));
}

extern "C" void dequantize_block_iq1_s(const block_iq1_s *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK_K == 0);
  const int nb = (int)(k / QK_K);
  for (int i = 0; i < nb; ++i)
  {
    const float d = LIBGGUF_FP16_TO_FP32(x[i].d);
    float *LIBGGUF_RESTRICT yb = y + i * QK_K;
    for (int ib = 0; ib < QK_K / 32; ++ib)
    {
      const uint16_t qh = x[i].qh[ib];
      const __m128 dl = _mm_set1_ps(d * (float)(2 * ((qh >> 12) & 7) + 1));
      const __m128 delta = _mm_set1_ps((qh & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA);
      const uint8_t *LIBGGUF_RESTRICT qs = x[i].qs + 4 * ib;
      for (int l = 0; l < 4; ++l)
      {
        const int8_t *grid = (const int8_t *)(iq1s_grid + (qs[l] | (((qh >> (3 * l)) & 7) << 8)));
        libgguf_iq1_store8(yb + 32 * ib + 8 * l, grid, dl, delta);
      }
    }
  }
}


void dequantize_iq1_s(const block_iq1_s *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  dequantize_block_iq1_s(x, y, k);
}

