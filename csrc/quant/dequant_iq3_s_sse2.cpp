#include "common/libgguf_common.h"
#include "common/libgguf_tables.h"

#include <string.h>

#if defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(__SSE2__)
#include <emmintrin.h>
#define LIBGGUF_BUILD_SSE2 1
#endif

static inline uint32_t libgguf_load_u32(const void *p)
{
  uint32_t v;
  memcpy(&v, p, sizeof(v));
  return v;
}

static inline void libgguf_iq3_s_store8_sse2(float *RESTRICT y, const uint8_t *RESTRICT grid1, const uint8_t *RESTRICT grid2, uint8_t signs, __m128 db)
{
  const __m128i zero = _mm_setzero_si128();
  const __m128i g = _mm_unpacklo_epi32(_mm_cvtsi32_si128((int)libgguf_load_u32(grid1)), _mm_cvtsi32_si128((int)libgguf_load_u32(grid2)));
  const __m128i g16 = _mm_unpacklo_epi8(g, zero);
  const uint32_t m0[4] = {(signs & 1) ? 0x80000000u : 0u, (signs & 2) ? 0x80000000u : 0u, (signs & 4) ? 0x80000000u : 0u, (signs & 8) ? 0x80000000u : 0u};
  const uint32_t m1[4] = {(signs & 16) ? 0x80000000u : 0u, (signs & 32) ? 0x80000000u : 0u, (signs & 64) ? 0x80000000u : 0u, (signs & 128) ? 0x80000000u : 0u};
  _mm_storeu_ps(y + 0, _mm_xor_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(g16, zero)), db), _mm_castsi128_ps(_mm_loadu_si128((const __m128i *)m0))));
  _mm_storeu_ps(y + 4, _mm_xor_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(g16, zero)), db), _mm_castsi128_ps(_mm_loadu_si128((const __m128i *)m1))));
}

extern "C" void dequantize_row_iq3_s_sse2(const block_iq3_s *RESTRICT x, float *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_SSE2)
  assert(k % QK_K == 0);
  const int nb = (int)(k / QK_K);
  for (int i = 0; i < nb; ++i)
  {
    const float d = GGML_FP16_TO_FP32(x[i].d);
    float *RESTRICT yb = y + i * QK_K;
    for (int pair = 0; pair < QK_K / 64; ++pair)
    {
      const uint8_t *RESTRICT qs0 = x[i].qs + 16 * pair;
      const uint8_t *RESTRICT qs1 = qs0 + 8;
      const uint8_t qh0 = x[i].qh[2 * pair + 0];
      const uint8_t qh1 = x[i].qh[2 * pair + 1];
      const uint8_t *RESTRICT signs0 = x[i].signs + 8 * pair;
      const uint8_t *RESTRICT signs1 = signs0 + 4;
      const __m128 db1 = _mm_set1_ps(d * (float)(1 + 2 * (x[i].scales[pair] & 0xf)));
      const __m128 db2 = _mm_set1_ps(d * (float)(1 + 2 * (x[i].scales[pair] >> 4)));
      float *RESTRICT yd = yb + 64 * pair;
      for (int l = 0; l < 4; ++l)
      {
        const uint8_t *grid1 = (const uint8_t *)(iq3s_grid + (qs0[2 * l + 0] | ((qh0 << (8 - 2 * l)) & 256)));
        const uint8_t *grid2 = (const uint8_t *)(iq3s_grid + (qs0[2 * l + 1] | ((qh0 << (7 - 2 * l)) & 256)));
        libgguf_iq3_s_store8_sse2(yd + 8 * l, grid1, grid2, signs0[l], db1);
      }
      for (int l = 0; l < 4; ++l)
      {
        const uint8_t *grid1 = (const uint8_t *)(iq3s_grid + (qs1[2 * l + 0] | ((qh1 << (8 - 2 * l)) & 256)));
        const uint8_t *grid2 = (const uint8_t *)(iq3s_grid + (qs1[2 * l + 1] | ((qh1 << (7 - 2 * l)) & 256)));
        libgguf_iq3_s_store8_sse2(yd + 32 + 8 * l, grid1, grid2, signs1[l], db2);
      }
    }
  }
#else
  GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(k); GGML_UNREACHABLE();
#endif
}
