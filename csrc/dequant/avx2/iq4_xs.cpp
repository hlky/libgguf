#include "common/libgguf_common.h"
#include "common/libgguf_tables.h"

#if defined(__AVX2__) || defined(_MSC_VER)
#include <immintrin.h>
#define LIBGGUF_BUILD_AVX2 1
#endif

static inline void libgguf_iq4_xs_store8_avx2(float *RESTRICT y, int v0, int v1, int v2, int v3, int v4, int v5, int v6, int v7, __m256 dl)
{
  const __m256 q = _mm256_set_ps((float)v7, (float)v6, (float)v5, (float)v4, (float)v3, (float)v2, (float)v1, (float)v0);
  _mm256_storeu_ps(y, _mm256_mul_ps(q, dl));
}

extern "C" void dequantize_row_iq4_xs_avx2(const block_iq4_xs *RESTRICT x, float *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_AVX2)
  assert(k % QK_K == 0);
  const int nb = (int)(k / QK_K);
  for (int i = 0; i < nb; ++i)
  {
    const float d = GGML_FP16_TO_FP32(x[i].d);
    const uint8_t *RESTRICT qs = x[i].qs;
    float *RESTRICT yb = y + i * QK_K;
    for (int ib = 0; ib < QK_K / 32; ++ib)
    {
      const int ls = ((x[i].scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((x[i].scales_h >> (2 * ib)) & 3) << 4);
      const __m256 dl = _mm256_set1_ps(d * (float)(ls - 32));
      float *RESTRICT yd = yb + 32 * ib;
      for (int j = 0; j < 16; j += 8)
      {
        libgguf_iq4_xs_store8_avx2(yd + j, kvalues_iq4nl[qs[j + 0] & 15], kvalues_iq4nl[qs[j + 1] & 15],
                                  kvalues_iq4nl[qs[j + 2] & 15], kvalues_iq4nl[qs[j + 3] & 15],
                                  kvalues_iq4nl[qs[j + 4] & 15], kvalues_iq4nl[qs[j + 5] & 15],
                                  kvalues_iq4nl[qs[j + 6] & 15], kvalues_iq4nl[qs[j + 7] & 15], dl);
        libgguf_iq4_xs_store8_avx2(yd + j + 16, kvalues_iq4nl[qs[j + 0] >> 4], kvalues_iq4nl[qs[j + 1] >> 4],
                                  kvalues_iq4nl[qs[j + 2] >> 4], kvalues_iq4nl[qs[j + 3] >> 4],
                                  kvalues_iq4nl[qs[j + 4] >> 4], kvalues_iq4nl[qs[j + 5] >> 4],
                                  kvalues_iq4nl[qs[j + 6] >> 4], kvalues_iq4nl[qs[j + 7] >> 4], dl);
      }
      qs += 16;
    }
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
