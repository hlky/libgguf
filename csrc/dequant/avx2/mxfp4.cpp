#include "common/libgguf_common.h"

#if defined(__AVX2__)
#include <immintrin.h>
#define LIBGGUF_BUILD_AVX2 1
#endif

static inline void libgguf_mxfp4_store8_avx2(float *RESTRICT y, int v0, int v1, int v2, int v3, int v4, int v5, int v6, int v7, __m256 d)
{
  const __m256 q = _mm256_set_ps((float)v7, (float)v6, (float)v5, (float)v4, (float)v3, (float)v2, (float)v1, (float)v0);
  _mm256_storeu_ps(y, _mm256_mul_ps(q, d));
}

extern "C" void dequantize_row_mxfp4_avx2(const block_mxfp4 *RESTRICT x, float *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_AVX2)
  assert(k % QK_MXFP4 == 0);
  static_assert(QK_MXFP4 == 32, "QK_MXFP4 must be 32");

  const int nb = (int)(k / QK_MXFP4);
  for (int i = 0; i < nb; ++i)
  {
    const __m256 d = _mm256_set1_ps(GGML_E8M0_TO_FP32_HALF(x[i].e));
    float *RESTRICT yb = y + i * QK_MXFP4;
    for (int j = 0; j < QK_MXFP4 / 2; j += 8)
    {
      libgguf_mxfp4_store8_avx2(yb + j, kvalues_mxfp4[x[i].qs[j + 0] & 15], kvalues_mxfp4[x[i].qs[j + 1] & 15],
                                kvalues_mxfp4[x[i].qs[j + 2] & 15], kvalues_mxfp4[x[i].qs[j + 3] & 15],
                                kvalues_mxfp4[x[i].qs[j + 4] & 15], kvalues_mxfp4[x[i].qs[j + 5] & 15],
                                kvalues_mxfp4[x[i].qs[j + 6] & 15], kvalues_mxfp4[x[i].qs[j + 7] & 15], d);
      libgguf_mxfp4_store8_avx2(yb + j + QK_MXFP4 / 2, kvalues_mxfp4[x[i].qs[j + 0] >> 4], kvalues_mxfp4[x[i].qs[j + 1] >> 4],
                                kvalues_mxfp4[x[i].qs[j + 2] >> 4], kvalues_mxfp4[x[i].qs[j + 3] >> 4],
                                kvalues_mxfp4[x[i].qs[j + 4] >> 4], kvalues_mxfp4[x[i].qs[j + 5] >> 4],
                                kvalues_mxfp4[x[i].qs[j + 6] >> 4], kvalues_mxfp4[x[i].qs[j + 7] >> 4], d);
    }
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
