#include "common/libgguf_common.h"
#include "common/libgguf_tables.h"

#include <string.h>

#if defined(__AVX2__) || defined(_MSC_VER)
#include <immintrin.h>
#define LIBGGUF_BUILD_AVX2 1
#endif

static inline uint32_t libgguf_load_u32(const void *p)
{
  uint32_t v;
  memcpy(&v, p, sizeof(v));
  return v;
}

static inline void libgguf_iq3_xxs_store8_avx2(float *RESTRICT y, const uint8_t *RESTRICT grid1, const uint8_t *RESTRICT grid2, uint8_t signs, __m256 db)
{
  const __m128i g = _mm_unpacklo_epi32(_mm_cvtsi32_si128((int)libgguf_load_u32(grid1)), _mm_cvtsi32_si128((int)libgguf_load_u32(grid2)));
  const uint32_t m[8] = {(signs & 1) ? 0x80000000u : 0u, (signs & 2) ? 0x80000000u : 0u, (signs & 4) ? 0x80000000u : 0u, (signs & 8) ? 0x80000000u : 0u,
                         (signs & 16) ? 0x80000000u : 0u, (signs & 32) ? 0x80000000u : 0u, (signs & 64) ? 0x80000000u : 0u, (signs & 128) ? 0x80000000u : 0u};
  _mm256_storeu_ps(y, _mm256_xor_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g)), db), _mm256_castsi256_ps(_mm256_loadu_si256((const __m256i *)m))));
}

extern "C" void dequantize_row_iq3_xxs_avx2(const block_iq3_xxs *RESTRICT x, float *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_AVX2)
  assert(k % QK_K == 0);
  const int nb = (int)(k / QK_K);
  for (int i = 0; i < nb; ++i)
  {
    const float d = GGML_FP16_TO_FP32(x[i].d);
    const uint8_t *RESTRICT scales_and_signs = x[i].qs + QK_K / 4;
    float *RESTRICT yb = y + i * QK_K;
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32)
    {
      const uint32_t aux32 = libgguf_load_u32(scales_and_signs + 4 * ib32);
      const __m256 db = _mm256_set1_ps(d * (0.5f + (float)(aux32 >> 28)) * 0.5f);
      const uint8_t *RESTRICT qs = x[i].qs + 8 * ib32;
      for (int l = 0; l < 4; ++l)
      {
        const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
        const uint8_t *grid1 = (const uint8_t *)(iq3xxs_grid + qs[2 * l + 0]);
        const uint8_t *grid2 = (const uint8_t *)(iq3xxs_grid + qs[2 * l + 1]);
        libgguf_iq3_xxs_store8_avx2(yb + 32 * ib32 + 8 * l, grid1, grid2, signs, db);
      }
    }
  }
#else
  GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(k); GGML_UNREACHABLE();
#endif
}
