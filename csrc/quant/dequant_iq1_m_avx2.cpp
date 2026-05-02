#include "common/libgguf_common.h"
#include "common/libgguf_iq_tables.h"
#include "common/libgguf_tables.h"

#include <string.h>

#if defined(__AVX2__) || defined(_MSC_VER)
#include <immintrin.h>
#define LIBGGUF_BUILD_AVX2 1
#endif

static inline uint16_t libgguf_load_u16(const void *p)
{
  uint16_t v;
  memcpy(&v, p, sizeof(v));
  return v;
}

static inline void libgguf_iq1_m_store8_avx2(float *RESTRICT y, const int8_t *RESTRICT grid, __m256 dl, __m256 delta)
{
  const __m128i g8 = _mm_loadl_epi64((const __m128i *)grid);
  const __m256 f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(g8));
  _mm256_storeu_ps(y, _mm256_mul_ps(_mm256_add_ps(f, delta), dl));
}

extern "C" void dequantize_row_iq1_m_avx2(const block_iq1_m *RESTRICT x, float *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_AVX2)
  assert(k % QK_K == 0);
  const int nb = (int)(k / QK_K);
  for (int i = 0; i < nb; ++i)
  {
    const uint8_t *RESTRICT scales = x[i].scales;
    const uint16_t sc0 = libgguf_load_u16(scales + 0);
    const uint16_t sc1 = libgguf_load_u16(scales + 2);
    const uint16_t sc2 = libgguf_load_u16(scales + 4);
    const uint16_t sc3 = libgguf_load_u16(scales + 6);
    iq1m_scale_t scale;
    scale.u16 = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
    const float d = GGML_FP16_TO_FP32(scale.f16);
    float *RESTRICT yb = y + i * QK_K;
    for (int ib = 0; ib < QK_K / 32; ++ib)
    {
      const uint16_t sc = libgguf_load_u16(scales + 2 * (ib / 2));
      const __m256 dl1 = _mm256_set1_ps(d * (float)(2 * ((sc >> (6 * (ib % 2) + 0)) & 7) + 1));
      const __m256 dl2 = _mm256_set1_ps(d * (float)(2 * ((sc >> (6 * (ib % 2) + 3)) & 7) + 1));
      const uint8_t *RESTRICT qs = x[i].qs + 4 * ib;
      const uint8_t *RESTRICT qh = x[i].qh + 2 * ib;
      const uint16_t idx0 = qs[0] | ((qh[0] << 8) & 0x700);
      const uint16_t idx1 = qs[1] | ((qh[0] << 4) & 0x700);
      const uint16_t idx2 = qs[2] | ((qh[1] << 8) & 0x700);
      const uint16_t idx3 = qs[3] | ((qh[1] << 4) & 0x700);
      libgguf_iq1_m_store8_avx2(yb + 32 * ib + 0, (const int8_t *)(iq1s_grid + idx0), dl1, _mm256_set1_ps((qh[0] & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA));
      libgguf_iq1_m_store8_avx2(yb + 32 * ib + 8, (const int8_t *)(iq1s_grid + idx1), dl1, _mm256_set1_ps((qh[0] & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA));
      libgguf_iq1_m_store8_avx2(yb + 32 * ib + 16, (const int8_t *)(iq1s_grid + idx2), dl2, _mm256_set1_ps((qh[1] & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA));
      libgguf_iq1_m_store8_avx2(yb + 32 * ib + 24, (const int8_t *)(iq1s_grid + idx3), dl2, _mm256_set1_ps((qh[1] & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA));
    }
  }
#else
  GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(k); GGML_UNREACHABLE();
#endif
}
