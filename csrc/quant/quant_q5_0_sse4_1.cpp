#include "libgguf_common.h"

#if defined(__SSE4_1__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
#include <smmintrin.h>
#define LIBGGUF_BUILD_SSE4_1 1
#endif

#if defined(LIBGGUF_BUILD_SSE4_1)
static inline float q5_0_signed_absmax_sse4_1(const float *RESTRICT x)
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
    best_abs = _mm_blendv_ps(best_abs, av, mask);
    best_val = _mm_blendv_ps(best_val, v, mask);
    best_idx = _mm_blendv_ps(best_idx, idx, mask);
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

static inline __m128i q5_0_quantize_4_i32_sse4_1(const float *RESTRICT x, __m128 id, __m128 offset, __m128i max_q)
{
  const __m128 scaled = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(x), id), offset);
  return _mm_min_epi32(_mm_cvttps_epi32(scaled), max_q);
}
#endif

extern "C" void quantize_row_q5_0_sse4_1(const float *RESTRICT x, block_q5_0 *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_SSE4_1)
  assert(k % QK5_0 == 0);
  static_assert(QK5_0 == 32, "QK5_0 must be 32");

  const int nb = (int)(k / QK5_0);
  const __m128 offset = _mm_set1_ps(16.5f);
  const __m128i max_q = _mm_set1_epi32(31);
  for (int i = 0; i < nb; ++i)
  {
    const float *xb = x + i * QK5_0;
    const float max = q5_0_signed_absmax_sse4_1(xb);
    const float d = max / -16;
    const float id = d ? 1.0f / d : 0.0f;
    y[i].d = GGML_FP32_TO_FP16(d);

    const __m128 idv = _mm_set1_ps(id);
    uint32_t qh = 0;
    for (int j = 0; j < QK5_0 / 2; j += 4)
    {
      alignas(16) int32_t lo[4];
      alignas(16) int32_t hi[4];
      _mm_store_si128((__m128i *)lo, q5_0_quantize_4_i32_sse4_1(xb + j, idv, offset, max_q));
      _mm_store_si128((__m128i *)hi, q5_0_quantize_4_i32_sse4_1(xb + QK5_0 / 2 + j, idv, offset, max_q));
      for (int lane = 0; lane < 4; ++lane)
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
