#include "kernels/cpu/common/common.h"
#include "kernels/cpu/dequantize/kernels.h"
#include "quantize/iq_tables.h"


#include <emmintrin.h>

static inline void libgguf_iq4_nl_store4(float *LIBGGUF_RESTRICT y, int v0, int v1, int v2, int v3, __m128 d)
{
  const __m128 q = _mm_set_ps((float)v3, (float)v2, (float)v1, (float)v0);
  _mm_storeu_ps(y, _mm_mul_ps(q, d));
}

extern "C" void dequantize_block_iq4_nl(const block_iq4_nl *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK4_NL == 0);
  static_assert(QK4_NL == 32, "QK4_NL must be 32");

  const int nb = (int)(k / QK4_NL);
  for (int i = 0; i < nb; ++i)
  {
    const __m128 d = _mm_set1_ps(LIBGGUF_FP16_TO_FP32(x[i].d));
    float *LIBGGUF_RESTRICT yb = y + i * QK4_NL;
    for (int j = 0; j < QK4_NL / 2; j += 4)
    {
      libgguf_iq4_nl_store4(yb + j, kvalues_iq4nl[x[i].qs[j + 0] & 15], kvalues_iq4nl[x[i].qs[j + 1] & 15],
                                 kvalues_iq4nl[x[i].qs[j + 2] & 15], kvalues_iq4nl[x[i].qs[j + 3] & 15], d);
      libgguf_iq4_nl_store4(yb + j + QK4_NL / 2, kvalues_iq4nl[x[i].qs[j + 0] >> 4], kvalues_iq4nl[x[i].qs[j + 1] >> 4],
                                 kvalues_iq4nl[x[i].qs[j + 2] >> 4], kvalues_iq4nl[x[i].qs[j + 3] >> 4], d);
    }
  }
}


void dequantize_iq4_nl(const block_iq4_nl *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  dequantize_block_iq4_nl(x, y, k);
}

