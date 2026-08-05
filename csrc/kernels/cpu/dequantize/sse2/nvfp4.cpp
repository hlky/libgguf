#include "kernels/cpu/common/common.h"
#include "kernels/cpu/dequantize/kernels.h"
#include "quantize/iq_tables.h"


#include <emmintrin.h>

static inline void libgguf_nvfp4_store4(float *LIBGGUF_RESTRICT y, int v0, int v1, int v2, int v3, __m128 d)
{
  const __m128 q = _mm_set_ps((float)v3, (float)v2, (float)v1, (float)v0);
  _mm_storeu_ps(y, _mm_mul_ps(q, d));
}

extern "C" void dequantize_block_nvfp4(const block_nvfp4 *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK_NVFP4 == 0);
  const int nb = (int)(k / QK_NVFP4);
  for (int i = 0; i < nb; ++i)
  {
    for (int s = 0; s < QK_NVFP4 / QK_NVFP4_SUB; ++s)
    {
      const __m128 d = _mm_set1_ps(libgguf_ue4m3_to_fp32(x[i].d[s]));
      const uint8_t *LIBGGUF_RESTRICT qs = x[i].qs + s * (QK_NVFP4_SUB / 2);
      float *LIBGGUF_RESTRICT yb = y + i * QK_NVFP4 + s * QK_NVFP4_SUB;
      for (int j = 0; j < QK_NVFP4_SUB / 2; j += 4)
      {
        libgguf_nvfp4_store4(yb + j, kvalues_mxfp4[qs[j + 0] & 15], kvalues_mxfp4[qs[j + 1] & 15],
                                  kvalues_mxfp4[qs[j + 2] & 15], kvalues_mxfp4[qs[j + 3] & 15], d);
        libgguf_nvfp4_store4(yb + j + QK_NVFP4_SUB / 2, kvalues_mxfp4[qs[j + 0] >> 4], kvalues_mxfp4[qs[j + 1] >> 4],
                                  kvalues_mxfp4[qs[j + 2] >> 4], kvalues_mxfp4[qs[j + 3] >> 4], d);
      }
    }
  }
}


void dequantize_nvfp4(const block_nvfp4 *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  dequantize_block_nvfp4(x, y, k);
}

