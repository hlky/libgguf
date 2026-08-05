#include "kernels/cpu/common/common.h"
#include "kernels/cpu/dequantize/kernels.h"
#include "quantize/iq_tables.h"

#include "kernels/cpu/common/tables.h"

#include <immintrin.h>

static inline void libgguf_iq1_store8(float *LIBGGUF_RESTRICT y, const int8_t *LIBGGUF_RESTRICT grid, __m256 dl, __m256 delta)
{
  const __m128i g8 = _mm_loadl_epi64((const __m128i *)grid);
  const __m256 f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g8));
  _mm256_storeu_ps(y, _mm256_mul_ps(_mm256_add_ps(f, delta), dl));
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
      const __m256 dl = _mm256_set1_ps(d * (float)(2 * ((qh >> 12) & 7) + 1));
      const __m256 delta = _mm256_set1_ps((qh & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA);
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

