#include "kernels/cpu/common/common.h"
#include "kernels/cpu/dequantize/kernels.h"
#include "quantize/iq_tables.h"

#include "kernels/cpu/common/tables.h"

#include <smmintrin.h>

static inline void libgguf_iq2_s_store8(float *LIBGGUF_RESTRICT y, const uint8_t *LIBGGUF_RESTRICT grid, uint8_t signs, __m128 db)
{
  const __m128i g8 = _mm_loadl_epi64((const __m128i *)grid);
  const uint32_t m0[4] = {(signs & 1) ? 0x80000000u : 0u, (signs & 2) ? 0x80000000u : 0u, (signs & 4) ? 0x80000000u : 0u, (signs & 8) ? 0x80000000u : 0u};
  const uint32_t m1[4] = {(signs & 16) ? 0x80000000u : 0u, (signs & 32) ? 0x80000000u : 0u, (signs & 64) ? 0x80000000u : 0u, (signs & 128) ? 0x80000000u : 0u};
  _mm_storeu_ps(y + 0, _mm_xor_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(g8)), db), _mm_castsi128_ps(_mm_loadu_si128((const __m128i *)m0))));
  _mm_storeu_ps(y + 4, _mm_xor_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(g8, 4))), db), _mm_castsi128_ps(_mm_loadu_si128((const __m128i *)m1))));
}

extern "C" void dequantize_block_iq2_s(const block_iq2_s *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK_K == 0);
  const int nb = (int)(k / QK_K);
  for (int i = 0; i < nb; ++i)
  {
    const float d = LIBGGUF_FP16_TO_FP32(x[i].d);
    const uint8_t *LIBGGUF_RESTRICT qs = x[i].qs;
    const uint8_t *LIBGGUF_RESTRICT signs = qs + QK_K / 8;
    float *LIBGGUF_RESTRICT yb = y + i * QK_K;
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32)
    {
      const __m128 db0 = _mm_set1_ps(d * (0.5f + (float)(x[i].scales[ib32] & 0xf)) * 0.25f);
      const __m128 db1 = _mm_set1_ps(d * (0.5f + (float)(x[i].scales[ib32] >> 4)) * 0.25f);
      for (int l = 0; l < 4; ++l)
      {
        const uint16_t idx = (uint16_t)(qs[l] | ((x[i].qh[ib32] << (8 - 2 * l)) & 0x300));
        const uint8_t *grid = (const uint8_t *)(iq2s_grid + idx);
        libgguf_iq2_s_store8(yb + 32 * ib32 + 8 * l, grid, signs[l], l < 2 ? db0 : db1);
      }
      qs += 4;
      signs += 4;
    }
  }
}


void dequantize_iq2_s(const block_iq2_s *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  dequantize_block_iq2_s(x, y, k);
}

