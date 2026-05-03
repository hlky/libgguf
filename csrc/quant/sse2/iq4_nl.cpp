#include "libgguf_common.h"

#if defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(__SSE2__)
#include <emmintrin.h>
#include <xmmintrin.h>
#define LIBGGUF_BUILD_SSE2 1
#endif

#if defined(LIBGGUF_BUILD_SSE2)
static inline void iq4_nl_indices_4_sse2(const float *RESTRICT x, float id, uint8_t *RESTRICT L)
{
  const __m128 xv = _mm_mul_ps(_mm_loadu_ps(x), _mm_set1_ps(id));
  int i0 = 0;
  int i1 = 0;
  int i2 = 0;
  int i3 = 0;
  for (int i = 1; i < 16; ++i)
  {
    const float midpoint = 0.5f * ((float)kvalues_iq4nl[i - 1] + (float)kvalues_iq4nl[i]);
    const int mask = _mm_movemask_ps(_mm_cmpge_ps(xv, _mm_set1_ps(midpoint)));
    i0 += (mask >> 0) & 1;
    i1 += (mask >> 1) & 1;
    i2 += (mask >> 2) & 1;
    i3 += (mask >> 3) & 1;
  }
  L[0] = (uint8_t)i0;
  L[1] = (uint8_t)i1;
  L[2] = (uint8_t)i2;
  L[3] = (uint8_t)i3;
}

static bool iq4_nl_quantize_block_sse2(const float *RESTRICT x, block_iq4_nl *RESTRICT y)
{
  uint8_t L[QK4_NL];
  float amax = 0.0f;
  float max = 0.0f;
  for (int j = 0; j < QK4_NL; ++j)
  {
    if (!(x[j] == x[j]))
      return false;
    const float ax = fabsf(x[j]);
    if (ax > amax)
    {
      amax = ax;
      max = x[j];
    }
  }
  if (amax < GROUP_MAX_EPS)
    return false;

  const float id = 1.0f / (max / kvalues_iq4nl[0]);
  for (int j = 0; j < QK4_NL; j += 4)
    iq4_nl_indices_4_sse2(x + j, id, L + j);

  float sumqx = 0.0f;
  float sumq2 = 0.0f;
  for (int j = 0; j < QK4_NL; ++j)
  {
    const float q = (float)kvalues_iq4nl[L[j]];
    const float w = x[j] * x[j];
    sumqx += w * q * x[j];
    sumq2 += w * q * q;
  }

  y->d = GGML_FP32_TO_FP16(sumq2 > 0.0f ? sumqx / sumq2 : 0.0f);
  for (int j = 0; j < QK4_NL / 2; ++j)
    y->qs[j] = L[j] | (L[j + QK4_NL / 2] << 4);
  return true;
}
#endif

extern "C" void quantize_row_iq4_nl_sse2(const float *RESTRICT x, block_iq4_nl *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_SSE2)
  assert(k % QK4_NL == 0);
  const int64_t nblock = k / QK4_NL;
  for (int64_t i = 0; i < nblock; ++i)
  {
    if (!iq4_nl_quantize_block_sse2(x + i * QK4_NL, y + i))
    {
      quantize_row_iq4_nl(x, y, k);
      return;
    }
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
