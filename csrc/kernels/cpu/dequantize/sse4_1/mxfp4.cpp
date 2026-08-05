#include "kernels/cpu/common/common.h"
#include "kernels/cpu/dequantize/kernels.h"
#include "quantize/iq_tables.h"


#include <smmintrin.h>

static inline void libgguf_mxfp4_store4(float *LIBGGUF_RESTRICT y, int v0, int v1, int v2, int v3, __m128 d)
{
  const __m128 q = _mm_set_ps((float)v3, (float)v2, (float)v1, (float)v0);
  _mm_storeu_ps(y, _mm_mul_ps(q, d));
}

extern "C" void dequantize_block_mxfp4(const block_mxfp4 *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK_MXFP4 == 0);
  static_assert(QK_MXFP4 == 32, "QK_MXFP4 must be 32");

  const int nb = (int)(k / QK_MXFP4);
  for (int i = 0; i < nb; ++i)
  {
    const __m128 d = _mm_set1_ps(LIBGGUF_E8M0_TO_FP32_HALF(x[i].e));
    float *LIBGGUF_RESTRICT yb = y + i * QK_MXFP4;
    for (int j = 0; j < QK_MXFP4 / 2; j += 4)
    {
      libgguf_mxfp4_store4(yb + j, kvalues_mxfp4[x[i].qs[j + 0] & 15], kvalues_mxfp4[x[i].qs[j + 1] & 15],
                                  kvalues_mxfp4[x[i].qs[j + 2] & 15], kvalues_mxfp4[x[i].qs[j + 3] & 15], d);
      libgguf_mxfp4_store4(yb + j + QK_MXFP4 / 2, kvalues_mxfp4[x[i].qs[j + 0] >> 4], kvalues_mxfp4[x[i].qs[j + 1] >> 4],
                                  kvalues_mxfp4[x[i].qs[j + 2] >> 4], kvalues_mxfp4[x[i].qs[j + 3] >> 4], d);
    }
  }
}


void dequantize_mxfp4(const block_mxfp4 *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  dequantize_block_mxfp4(x, y, k);
}

