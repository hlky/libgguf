#include "libgguf_common.h"

#if defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(__SSE2__)
#include <emmintrin.h>
#include <xmmintrin.h>
#define LIBGGUF_BUILD_SSE2 1
#endif

#if defined(LIBGGUF_BUILD_SSE2)
static inline __m128 q5_0_blendv_ps_sse2(__m128 a, __m128 b, __m128 mask)
{
  return _mm_or_ps(_mm_and_ps(mask, b), _mm_andnot_ps(mask, a));
}

static inline float q5_0_signed_absmax_sse2(const float *RESTRICT x)
{
  const __m128 sign_mask = _mm_set1_ps(-0.0f);
  const __m128 step = _mm_set1_ps(4.0f);
  __m128 idx = _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f);
  __m128 best_idx = _mm_set1_ps(2147483648.0f);
  __m128 best_abs = _mm_setzero_ps();
  __m128 best_val = _mm_setzero_ps();

  for (int j = 0; j < QK5_0; j += 4)
  {
    const __m128 v = _mm_loadu_ps(x + j);
    const __m128 av = _mm_andnot_ps(sign_mask, v);
    const __m128 mask = _mm_cmpgt_ps(av, best_abs);
    best_abs = q5_0_blendv_ps_sse2(best_abs, av, mask);
    best_val = q5_0_blendv_ps_sse2(best_val, v, mask);
    best_idx = q5_0_blendv_ps_sse2(best_idx, idx, mask);
    idx = _mm_add_ps(idx, step);
  }

  alignas(16) float abs_parts[4];
  alignas(16) float val_parts[4];
  alignas(16) float idx_parts[4];
  _mm_store_ps(abs_parts, best_abs);
  _mm_store_ps(val_parts, best_val);
  _mm_store_ps(idx_parts, best_idx);

  float amax = abs_parts[0];
  float max = val_parts[0];
  float min_idx = idx_parts[0];
  for (int lane = 1; lane < 4; ++lane)
  {
    if (abs_parts[lane] > amax || (abs_parts[lane] == amax && idx_parts[lane] < min_idx))
    {
      amax = abs_parts[lane];
      max = val_parts[lane];
      min_idx = idx_parts[lane];
    }
  }
  return max;
}

static inline __m128i q5_0_quantize_8_i16_sse2(const float *RESTRICT x, __m128 id, __m128 offset, __m128i max_q)
{
  const __m128i q0 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(_mm_loadu_ps(x), id), offset));
  const __m128i q1 = _mm_cvttps_epi32(_mm_add_ps(_mm_mul_ps(_mm_loadu_ps(x + 4), id), offset));
  return _mm_min_epi16(_mm_packs_epi32(q0, q1), max_q);
}
#endif

extern "C" void quantize_row_q5_0_sse2(const float *RESTRICT x, block_q5_0 *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_SSE2)
  assert(k % QK5_0 == 0);
  static_assert(QK5_0 == 32, "QK5_0 must be 32");

  const int nb = (int)(k / QK5_0);
  const __m128 offset = _mm_set1_ps(16.5f);
  const __m128i max_q = _mm_set1_epi16(31);
  for (int i = 0; i < nb; ++i)
  {
    const float *xb = x + i * QK5_0;
    const float max = q5_0_signed_absmax_sse2(xb);
    const float d = max / -16;
    const float id = d ? 1.0f / d : 0.0f;
    y[i].d = GGML_FP32_TO_FP16(d);

    const __m128 idv = _mm_set1_ps(id);
    uint32_t qh = 0;
    for (int j = 0; j < QK5_0 / 2; j += 8)
    {
      alignas(16) uint16_t lo[8];
      alignas(16) uint16_t hi[8];
      _mm_store_si128((__m128i *)lo, q5_0_quantize_8_i16_sse2(xb + j, idv, offset, max_q));
      _mm_store_si128((__m128i *)hi, q5_0_quantize_8_i16_sse2(xb + QK5_0 / 2 + j, idv, offset, max_q));
      for (int lane = 0; lane < 8; ++lane)
      {
        const uint8_t xi0 = (uint8_t)lo[lane];
        const uint8_t xi1 = (uint8_t)hi[lane];
        y[i].qs[j + lane] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + lane);
        qh |= ((xi1 & 0x10u) >> 4) << (j + lane + QK5_0 / 2);
      }
    }
    memcpy(&y[i].qh, &qh, sizeof(qh));
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
