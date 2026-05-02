#include "common/libgguf_common.h"

#if defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(__SSE2__)
#include <emmintrin.h>
#define LIBGGUF_BUILD_SSE2 1
#endif

static inline void libgguf_iq4_nl_store4_sse2(float *RESTRICT y, int v0, int v1, int v2, int v3, __m128 d)
{
  const __m128 q = _mm_set_ps((float)v3, (float)v2, (float)v1, (float)v0);
  _mm_storeu_ps(y, _mm_mul_ps(q, d));
}

extern "C" void dequantize_row_iq4_nl_sse2(const block_iq4_nl *RESTRICT x, float *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_SSE2)
  assert(k % QK4_NL == 0);
  static_assert(QK4_NL == 32, "QK4_NL must be 32");

  const int nb = (int)(k / QK4_NL);
  for (int i = 0; i < nb; ++i)
  {
    const __m128 d = _mm_set1_ps(GGML_FP16_TO_FP32(x[i].d));
    float *RESTRICT yb = y + i * QK4_NL;
    for (int j = 0; j < QK4_NL / 2; j += 4)
    {
      libgguf_iq4_nl_store4_sse2(yb + j, kvalues_iq4nl[x[i].qs[j + 0] & 15], kvalues_iq4nl[x[i].qs[j + 1] & 15],
                                 kvalues_iq4nl[x[i].qs[j + 2] & 15], kvalues_iq4nl[x[i].qs[j + 3] & 15], d);
      libgguf_iq4_nl_store4_sse2(yb + j + QK4_NL / 2, kvalues_iq4nl[x[i].qs[j + 0] >> 4], kvalues_iq4nl[x[i].qs[j + 1] >> 4],
                                 kvalues_iq4nl[x[i].qs[j + 2] >> 4], kvalues_iq4nl[x[i].qs[j + 3] >> 4], d);
    }
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
