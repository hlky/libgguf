#include "kernels/cpu/common/common.h"
#include "kernels/cpu/dequantize/kernels.h"
#include "quantize/iq_tables.h"

#include "kernels/cpu/common/tables.h"

#include <string.h>

#include <immintrin.h>

static inline uint32_t libgguf_load_u32(const void *p)
{
  uint32_t v;
  memcpy(&v, p, sizeof(v));
  return v;
}

static inline void libgguf_iq3_s_store8(float *LIBGGUF_RESTRICT y, const uint8_t *LIBGGUF_RESTRICT grid1, const uint8_t *LIBGGUF_RESTRICT grid2, uint8_t signs, __m256 db)
{
  const __m128i g = _mm_unpacklo_epi32(_mm_cvtsi32_si128((int)libgguf_load_u32(grid1)), _mm_cvtsi32_si128((int)libgguf_load_u32(grid2)));
  const uint32_t m[8] = {(signs & 1) ? 0x80000000u : 0u, (signs & 2) ? 0x80000000u : 0u, (signs & 4) ? 0x80000000u : 0u, (signs & 8) ? 0x80000000u : 0u,
                         (signs & 16) ? 0x80000000u : 0u, (signs & 32) ? 0x80000000u : 0u, (signs & 64) ? 0x80000000u : 0u, (signs & 128) ? 0x80000000u : 0u};
  _mm256_storeu_ps(y, _mm256_xor_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g)), db), _mm256_castsi256_ps(_mm256_loadu_si256((const __m256i *)m))));
}

extern "C" void dequantize_block_iq3_s(const block_iq3_s *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK_K == 0);
  const int nb = (int)(k / QK_K);
  for (int i = 0; i < nb; ++i)
  {
    const float d = LIBGGUF_FP16_TO_FP32(x[i].d);
    float *LIBGGUF_RESTRICT yb = y + i * QK_K;
    for (int pair = 0; pair < QK_K / 64; ++pair)
    {
      const uint8_t *LIBGGUF_RESTRICT qs0 = x[i].qs + 16 * pair;
      const uint8_t *LIBGGUF_RESTRICT qs1 = qs0 + 8;
      const uint8_t qh0 = x[i].qh[2 * pair + 0];
      const uint8_t qh1 = x[i].qh[2 * pair + 1];
      const uint8_t *LIBGGUF_RESTRICT signs0 = x[i].signs + 8 * pair;
      const uint8_t *LIBGGUF_RESTRICT signs1 = signs0 + 4;
      const __m256 db1 = _mm256_set1_ps(d * (float)(1 + 2 * (x[i].scales[pair] & 0xf)));
      const __m256 db2 = _mm256_set1_ps(d * (float)(1 + 2 * (x[i].scales[pair] >> 4)));
      float *LIBGGUF_RESTRICT yd = yb + 64 * pair;
      for (int l = 0; l < 4; ++l)
      {
        const uint8_t *grid1 = (const uint8_t *)(iq3s_grid + (qs0[2 * l + 0] | ((qh0 << (8 - 2 * l)) & 256)));
        const uint8_t *grid2 = (const uint8_t *)(iq3s_grid + (qs0[2 * l + 1] | ((qh0 << (7 - 2 * l)) & 256)));
        libgguf_iq3_s_store8(yd + 8 * l, grid1, grid2, signs0[l], db1);
      }
      for (int l = 0; l < 4; ++l)
      {
        const uint8_t *grid1 = (const uint8_t *)(iq3s_grid + (qs1[2 * l + 0] | ((qh1 << (8 - 2 * l)) & 256)));
        const uint8_t *grid2 = (const uint8_t *)(iq3s_grid + (qs1[2 * l + 1] | ((qh1 << (7 - 2 * l)) & 256)));
        libgguf_iq3_s_store8(yd + 32 + 8 * l, grid1, grid2, signs1[l], db2);
      }
    }
  }
}


void dequantize_iq3_s(const block_iq3_s *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  dequantize_block_iq3_s(x, y, k);
}

