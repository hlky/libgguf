#include "libgguf_common.h"
#include "libgguf_tables.h"

#if defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(__SSE2__)
#include <emmintrin.h>
#include <xmmintrin.h>
#define LIBGGUF_BUILD_IQ4_XS_SSE2 1
#endif

extern "C" int libgguf_best_index_int8_sse2(int n, const int8_t *val, float x);

#if defined(LIBGGUF_BUILD_IQ4_XS_SSE2)
static inline void iq4_xs_fill_square_32_sse2(const float *RESTRICT x, float *RESTRICT weight)
{
  for (int j = 0; j < 32; j += 4)
  {
    const __m128 v = _mm_loadu_ps(x + j);
    _mm_storeu_ps(weight + j, _mm_mul_ps(v, v));
  }
}

static inline void iq4_xs_pack_32_sse2(const uint8_t *RESTRICT L, uint8_t *RESTRICT q)
{
  const __m128i low_mask = _mm_set1_epi8(0x0F);
  const __m128i lo = _mm_and_si128(_mm_loadu_si128((const __m128i *)L), low_mask);
  const __m128i hi = _mm_slli_epi16(_mm_and_si128(_mm_loadu_si128((const __m128i *)(L + 16)), low_mask), 4);
  _mm_storeu_si128((__m128i *)q, _mm_or_si128(lo, hi));
}

static bool quantize_one_iq4_xs_sse2(const float *RESTRICT x, block_iq4_xs *RESTRICT y)
{
  uint8_t L[QK_K];
  float weight[32];
  float scales[QK_K / 32];
  const int super_block_size = QK_K;
  const int block_size = 32;
  const int ntry = 7;
  const int8_t *values = kvalues_iq4nl;

  for (int j = 0; j < super_block_size; ++j)
  {
    if (!std::isfinite(x[j]))
    {
      return false;
    }
  }

  memset(y->qs, 0, super_block_size / 2);
  y->d = GGML_FP32_TO_FP16(0.0f);

  float max_scale = 0.0f;
  float amax_scale = 0.0f;
  for (int ib = 0; ib < super_block_size / block_size; ++ib)
  {
    const float *xb = x + ib * block_size;
    uint8_t *Lb = L + ib * block_size;
    iq4_xs_fill_square_32_sse2(xb, weight);

    float amax = 0.0f;
    float max = 0.0f;
    for (int j = 0; j < block_size; ++j)
    {
      const float ax = fabsf(xb[j]);
      if (ax > amax)
      {
        amax = ax;
        max = xb[j];
      }
    }
    if (amax < GROUP_MAX_EPS)
    {
      scales[ib] = 0.0f;
      continue;
    }

    float d = -max / values[0];
    float id = 1.0f / d;
    float sumqx = 0.0f;
    float sumq2 = 0.0f;
    for (int j = 0; j < block_size; ++j)
    {
      const int l = libgguf_best_index_int8_sse2(16, values, id * xb[j]);
      Lb[j] = (uint8_t)l;
      const float q = (float)values[l];
      const float w = weight[j];
      sumqx += w * q * xb[j];
      sumq2 += w * q * q;
    }
    d = sumq2 > 0.0f ? sumqx / sumq2 : 0.0f;
    float best = d * sumqx;

    for (int itry = -ntry; itry <= ntry; ++itry)
    {
      id = (itry + values[0]) / max;
      sumqx = 0.0f;
      sumq2 = 0.0f;
      for (int j = 0; j < block_size; ++j)
      {
        const int l = libgguf_best_index_int8_sse2(16, values, id * xb[j]);
        const float q = (float)values[l];
        const float w = weight[j];
        sumqx += w * q * xb[j];
        sumq2 += w * q * q;
      }
      if (sumq2 > 0.0f && sumqx * sumqx > best * sumq2)
      {
        d = sumqx / sumq2;
        best = d * sumqx;
      }
    }

    scales[ib] = d;
    const float abs_d = fabsf(d);
    if (abs_d > amax_scale)
    {
      amax_scale = abs_d;
      max_scale = d;
    }
  }

  memset(&y->scales_h, 0, sizeof(y->scales_h));
  const float d = -max_scale / 32.0f;
  y->d = GGML_FP32_TO_FP16(d);
  const float id = d ? 1.0f / d : 0.0f;
  for (int ib = 0; ib < super_block_size / block_size; ++ib)
  {
    int l = nearest_int(id * scales[ib]);
    l = MAX(-32, MIN(31, l));
    const float dl = d * l;
    const float idl = dl ? 1.0f / dl : 0.0f;
    uint8_t *Lb = L + ib * block_size;
    const float *xb = x + ib * block_size;
    for (int j = 0; j < block_size; ++j)
    {
      Lb[j] = (uint8_t)libgguf_best_index_int8_sse2(16, values, idl * xb[j]);
    }

    l += 32;
    const uint8_t l_l = (uint8_t)(l & 0x0f);
    const uint8_t l_h = (uint8_t)(l >> 4);
    if (ib % 2 == 0)
    {
      y->scales_l[ib / 2] = l_l;
    }
    else
    {
      y->scales_l[ib / 2] |= (uint8_t)(l_l << 4);
    }
    y->scales_h |= (uint16_t)(l_h << (2 * ib));
  }

  for (int i = 0; i < super_block_size / 32; ++i)
  {
    iq4_xs_pack_32_sse2(L + 32 * i, y->qs + 16 * i);
  }
  return true;
}
#endif

extern "C" void quantize_row_iq4_xs_sse2(const float *RESTRICT x, block_iq4_xs *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_IQ4_XS_SSE2)
  assert(k % QK_K == 0);
  const int64_t nblock = k / QK_K;
  for (int64_t i = 0; i < nblock; ++i)
  {
    if (!quantize_one_iq4_xs_sse2(x + i * QK_K, y + i))
    {
      quantize_row_iq4_xs_ref(x, y, k);
      return;
    }
  }
#else
  quantize_row_iq4_xs_ref(x, y, k);
#endif
}
