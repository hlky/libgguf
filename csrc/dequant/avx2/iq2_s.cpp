#include "common/libgguf_common.h"
#include "common/libgguf_tables.h"

#if defined(__AVX2__) || defined(_MSC_VER)
#include <immintrin.h>
#define LIBGGUF_BUILD_AVX2 1
#endif

static inline void libgguf_iq2_s_store8_avx2(float *RESTRICT y, const uint8_t *RESTRICT grid, uint8_t signs, __m256 db)
{
  const __m128i g8 = _mm_loadl_epi64((const __m128i *)grid);
  const uint32_t m[8] = {(signs & 1) ? 0x80000000u : 0u, (signs & 2) ? 0x80000000u : 0u, (signs & 4) ? 0x80000000u : 0u, (signs & 8) ? 0x80000000u : 0u, (signs & 16) ? 0x80000000u : 0u, (signs & 32) ? 0x80000000u : 0u, (signs & 64) ? 0x80000000u : 0u, (signs & 128) ? 0x80000000u : 0u};
  _mm256_storeu_ps(y, _mm256_xor_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(g8)), db), _mm256_castsi256_ps(_mm256_loadu_si256((const __m256i *)m))));
}

extern "C" void dequantize_row_iq2_s_avx2(const block_iq2_s *RESTRICT x, float *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_AVX2)
  assert(k % QK_K == 0);
  const int nb = (int)(k / QK_K);
  for (int i = 0; i < nb; ++i)
  {
    const float d = GGML_FP16_TO_FP32(x[i].d);
    const uint8_t *RESTRICT qs = x[i].qs;
    const uint8_t *RESTRICT signs = qs + QK_K / 8;
    float *RESTRICT yb = y + i * QK_K;
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32)
    {
      const __m256 db0 = _mm256_set1_ps(d * (0.5f + (float)(x[i].scales[ib32] & 0xf)) * 0.25f);
      const __m256 db1 = _mm256_set1_ps(d * (0.5f + (float)(x[i].scales[ib32] >> 4)) * 0.25f);
      for (int l = 0; l < 4; ++l)
      {
        const uint16_t idx = (uint16_t)(qs[l] | ((x[i].qh[ib32] << (8 - 2 * l)) & 0x300));
        const uint8_t *grid = (const uint8_t *)(iq2s_grid + idx);
        libgguf_iq2_s_store8_avx2(yb + 32 * ib32 + 8 * l, grid, signs[l], l < 2 ? db0 : db1);
      }
      qs += 4;
      signs += 4;
    }
  }
#else
  GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(k); GGML_UNREACHABLE();
#endif
}
