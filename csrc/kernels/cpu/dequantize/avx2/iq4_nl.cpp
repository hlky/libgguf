#include "kernels/cpu/common/common.h"
#include "kernels/cpu/dequantize/kernels.h"
#include "quantize/iq_tables.h"


#include <immintrin.h>

static inline void libgguf_iq4_nl_store8(float *LIBGGUF_RESTRICT y, int v0, int v1, int v2, int v3, int v4, int v5, int v6, int v7, __m256 d)
{
  const __m256 q = _mm256_set_ps((float)v7, (float)v6, (float)v5, (float)v4, (float)v3, (float)v2, (float)v1, (float)v0);
  _mm256_storeu_ps(y, _mm256_mul_ps(q, d));
}

extern "C" void dequantize_block_iq4_nl(const block_iq4_nl *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK4_NL == 0);
  static_assert(QK4_NL == 32, "QK4_NL must be 32");

  const int nb = (int)(k / QK4_NL);
  for (int i = 0; i < nb; ++i)
  {
    const __m256 d = _mm256_set1_ps(LIBGGUF_FP16_TO_FP32(x[i].d));
    float *LIBGGUF_RESTRICT yb = y + i * QK4_NL;
    for (int j = 0; j < QK4_NL / 2; j += 8)
    {
      libgguf_iq4_nl_store8(yb + j, kvalues_iq4nl[x[i].qs[j + 0] & 15], kvalues_iq4nl[x[i].qs[j + 1] & 15],
                                 kvalues_iq4nl[x[i].qs[j + 2] & 15], kvalues_iq4nl[x[i].qs[j + 3] & 15],
                                 kvalues_iq4nl[x[i].qs[j + 4] & 15], kvalues_iq4nl[x[i].qs[j + 5] & 15],
                                 kvalues_iq4nl[x[i].qs[j + 6] & 15], kvalues_iq4nl[x[i].qs[j + 7] & 15], d);
      libgguf_iq4_nl_store8(yb + j + QK4_NL / 2, kvalues_iq4nl[x[i].qs[j + 0] >> 4], kvalues_iq4nl[x[i].qs[j + 1] >> 4],
                                 kvalues_iq4nl[x[i].qs[j + 2] >> 4], kvalues_iq4nl[x[i].qs[j + 3] >> 4],
                                 kvalues_iq4nl[x[i].qs[j + 4] >> 4], kvalues_iq4nl[x[i].qs[j + 5] >> 4],
                                 kvalues_iq4nl[x[i].qs[j + 6] >> 4], kvalues_iq4nl[x[i].qs[j + 7] >> 4], d);
    }
  }
}


void dequantize_iq4_nl(const block_iq4_nl *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  dequantize_block_iq4_nl(x, y, k);
}

