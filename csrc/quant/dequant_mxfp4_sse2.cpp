#include "common/libgguf_common.h"

#if defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(__SSE2__)
#include <emmintrin.h>
#define LIBGGUF_BUILD_SSE2 1
#endif

static inline void libgguf_mxfp4_store4_sse2(float *RESTRICT y, int v0, int v1, int v2, int v3, __m128 d)
{
  const __m128 q = _mm_set_ps((float)v3, (float)v2, (float)v1, (float)v0);
  _mm_storeu_ps(y, _mm_mul_ps(q, d));
}

extern "C" void dequantize_row_mxfp4_sse2(const block_mxfp4 *RESTRICT x, float *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_SSE2)
  assert(k % QK_MXFP4 == 0);
  static_assert(QK_MXFP4 == 32, "QK_MXFP4 must be 32");

  const int nb = (int)(k / QK_MXFP4);
  for (int i = 0; i < nb; ++i)
  {
    const __m128 d = _mm_set1_ps(GGML_E8M0_TO_FP32_HALF(x[i].e));
    float *RESTRICT yb = y + i * QK_MXFP4;
    for (int j = 0; j < QK_MXFP4 / 2; j += 4)
    {
      libgguf_mxfp4_store4_sse2(yb + j, kvalues_mxfp4[x[i].qs[j + 0] & 15], kvalues_mxfp4[x[i].qs[j + 1] & 15],
                                kvalues_mxfp4[x[i].qs[j + 2] & 15], kvalues_mxfp4[x[i].qs[j + 3] & 15], d);
      libgguf_mxfp4_store4_sse2(yb + j + QK_MXFP4 / 2, kvalues_mxfp4[x[i].qs[j + 0] >> 4], kvalues_mxfp4[x[i].qs[j + 1] >> 4],
                                kvalues_mxfp4[x[i].qs[j + 2] >> 4], kvalues_mxfp4[x[i].qs[j + 3] >> 4], d);
    }
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
