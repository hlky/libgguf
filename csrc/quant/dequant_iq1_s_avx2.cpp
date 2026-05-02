#include "common/libgguf_common.h"
#include "common/libgguf_iq_tables.h"
#include "common/libgguf_tables.h"

#if defined(__AVX2__) || defined(_MSC_VER)
#include <immintrin.h>
#define LIBGGUF_BUILD_AVX2 1
#endif

static inline void libgguf_iq1_store8_avx2(float *RESTRICT y, const int8_t *RESTRICT grid, __m256 dl, __m256 delta)
{
  const __m128i g8 = _mm_loadl_epi64((const __m128i *)grid);
  const __m256 f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g8));
  _mm256_storeu_ps(y, _mm256_mul_ps(_mm256_add_ps(f, delta), dl));
}

extern "C" void dequantize_row_iq1_s_avx2(const block_iq1_s *RESTRICT x, float *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_AVX2)
  assert(k % QK_K == 0);
  const int nb = (int)(k / QK_K);
  for (int i = 0; i < nb; ++i)
  {
    const float d = GGML_FP16_TO_FP32(x[i].d);
    float *RESTRICT yb = y + i * QK_K;
    for (int ib = 0; ib < QK_K / 32; ++ib)
    {
      const uint16_t qh = x[i].qh[ib];
      const __m256 dl = _mm256_set1_ps(d * (float)(2 * ((qh >> 12) & 7) + 1));
      const __m256 delta = _mm256_set1_ps((qh & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA);
      const uint8_t *RESTRICT qs = x[i].qs + 4 * ib;
      for (int l = 0; l < 4; ++l)
      {
        const int8_t *grid = (const int8_t *)(iq1s_grid + (qs[l] | (((qh >> (3 * l)) & 7) << 8)));
        libgguf_iq1_store8_avx2(yb + 32 * ib + 8 * l, grid, dl, delta);
      }
    }
  }
#else
  GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(k); GGML_UNREACHABLE();
#endif
}
