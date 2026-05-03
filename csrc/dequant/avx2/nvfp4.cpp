#include "common/libgguf_common.h"

#if defined(__AVX2__)
#include <immintrin.h>
#define LIBGGUF_BUILD_AVX2 1
#endif

static inline void libgguf_nvfp4_store8_avx2(float *RESTRICT y, int v0, int v1, int v2, int v3, int v4, int v5, int v6, int v7, __m256 d)
{
  const __m256 q = _mm256_set_ps((float)v7, (float)v6, (float)v5, (float)v4, (float)v3, (float)v2, (float)v1, (float)v0);
  _mm256_storeu_ps(y, _mm256_mul_ps(q, d));
}

extern "C" void dequantize_row_nvfp4_avx2(const block_nvfp4 *RESTRICT x, float *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_AVX2)
  assert(k % QK_NVFP4 == 0);
  const int nb = (int)(k / QK_NVFP4);
  for (int i = 0; i < nb; ++i)
  {
    for (int s = 0; s < QK_NVFP4 / QK_NVFP4_SUB; ++s)
    {
      const __m256 d = _mm256_set1_ps(ggml_ue4m3_to_fp32(x[i].d[s]));
      const uint8_t *RESTRICT qs = x[i].qs + s * (QK_NVFP4_SUB / 2);
      float *RESTRICT yb = y + i * QK_NVFP4 + s * QK_NVFP4_SUB;
      libgguf_nvfp4_store8_avx2(yb, kvalues_mxfp4[qs[0] & 15], kvalues_mxfp4[qs[1] & 15], kvalues_mxfp4[qs[2] & 15],
                                kvalues_mxfp4[qs[3] & 15], kvalues_mxfp4[qs[4] & 15], kvalues_mxfp4[qs[5] & 15],
                                kvalues_mxfp4[qs[6] & 15], kvalues_mxfp4[qs[7] & 15], d);
      libgguf_nvfp4_store8_avx2(yb + QK_NVFP4_SUB / 2, kvalues_mxfp4[qs[0] >> 4], kvalues_mxfp4[qs[1] >> 4],
                                kvalues_mxfp4[qs[2] >> 4], kvalues_mxfp4[qs[3] >> 4], kvalues_mxfp4[qs[4] >> 4],
                                kvalues_mxfp4[qs[5] >> 4], kvalues_mxfp4[qs[6] >> 4], kvalues_mxfp4[qs[7] >> 4], d);
    }
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
