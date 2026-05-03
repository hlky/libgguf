#include "common/libgguf_common.h"

#if defined(_MSC_VER)
#include <smmintrin.h>
#define LIBGGUF_BUILD_SSE4_1 1
#elif defined(__SSE4_1__)
#include <smmintrin.h>
#define LIBGGUF_BUILD_SSE4_1 1
#endif

static inline void libgguf_nvfp4_store4_sse4_1(float *RESTRICT y, int v0, int v1, int v2, int v3, __m128 d)
{
  const __m128 q = _mm_set_ps((float)v3, (float)v2, (float)v1, (float)v0);
  _mm_storeu_ps(y, _mm_mul_ps(q, d));
}

extern "C" void dequantize_row_nvfp4_sse4_1(const block_nvfp4 *RESTRICT x, float *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_SSE4_1)
  assert(k % QK_NVFP4 == 0);
  const int nb = (int)(k / QK_NVFP4);
  for (int i = 0; i < nb; ++i)
  {
    for (int s = 0; s < QK_NVFP4 / QK_NVFP4_SUB; ++s)
    {
      const __m128 d = _mm_set1_ps(ggml_ue4m3_to_fp32(x[i].d[s]));
      const uint8_t *RESTRICT qs = x[i].qs + s * (QK_NVFP4_SUB / 2);
      float *RESTRICT yb = y + i * QK_NVFP4 + s * QK_NVFP4_SUB;
      for (int j = 0; j < QK_NVFP4_SUB / 2; j += 4)
      {
        libgguf_nvfp4_store4_sse4_1(yb + j, kvalues_mxfp4[qs[j + 0] & 15], kvalues_mxfp4[qs[j + 1] & 15],
                                    kvalues_mxfp4[qs[j + 2] & 15], kvalues_mxfp4[qs[j + 3] & 15], d);
        libgguf_nvfp4_store4_sse4_1(yb + j + QK_NVFP4_SUB / 2, kvalues_mxfp4[qs[j + 0] >> 4], kvalues_mxfp4[qs[j + 1] >> 4],
                                    kvalues_mxfp4[qs[j + 2] >> 4], kvalues_mxfp4[qs[j + 3] >> 4], d);
      }
    }
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
