#include "kernels/cpu/common/common.h"
#include "kernels/cpu/dequantize/kernels.h"
#include "quantize/iq_tables.h"

#include "kernels/cpu/common/tables.h"

#include <emmintrin.h>

static inline void libgguf_iq4_xs_store4(float *LIBGGUF_RESTRICT y, int v0, int v1, int v2, int v3, __m128 dl)
{
  const __m128 q = _mm_set_ps((float)v3, (float)v2, (float)v1, (float)v0);
  _mm_storeu_ps(y, _mm_mul_ps(q, dl));
}

extern "C" void dequantize_block_iq4_xs(const block_iq4_xs *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK_K == 0);
  const int nb = (int)(k / QK_K);
  for (int i = 0; i < nb; ++i)
  {
    const float d = LIBGGUF_FP16_TO_FP32(x[i].d);
    const uint8_t *LIBGGUF_RESTRICT qs = x[i].qs;
    float *LIBGGUF_RESTRICT yb = y + i * QK_K;
    for (int ib = 0; ib < QK_K / 32; ++ib)
    {
      const int ls = ((x[i].scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((x[i].scales_h >> (2 * ib)) & 3) << 4);
      const __m128 dl = _mm_set1_ps(d * (float)(ls - 32));
      float *LIBGGUF_RESTRICT yd = yb + 32 * ib;
      for (int j = 0; j < 16; j += 4)
      {
        libgguf_iq4_xs_store4(yd + j, kvalues_iq4nl[qs[j + 0] & 15], kvalues_iq4nl[qs[j + 1] & 15],
                                  kvalues_iq4nl[qs[j + 2] & 15], kvalues_iq4nl[qs[j + 3] & 15], dl);
        libgguf_iq4_xs_store4(yd + j + 16, kvalues_iq4nl[qs[j + 0] >> 4], kvalues_iq4nl[qs[j + 1] >> 4],
                                  kvalues_iq4nl[qs[j + 2] >> 4], kvalues_iq4nl[qs[j + 3] >> 4], dl);
      }
      qs += 16;
    }
  }
}


void dequantize_iq4_xs(const block_iq4_xs *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT y, int64_t k)
{
  dequantize_block_iq4_xs(x, y, k);
}

