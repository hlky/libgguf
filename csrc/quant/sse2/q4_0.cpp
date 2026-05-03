#include "libgguf_common.h"

#if defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(__SSE2__)
#include <emmintrin.h>
#include <xmmintrin.h>
#define LIBGGUF_BUILD_SSE2 1
#endif

#if defined(LIBGGUF_BUILD_SSE2)
static inline __m128 q4_0_blendv_ps_sse2(__m128 a, __m128 b, __m128 mask)
{
  return _mm_or_ps(_mm_and_ps(mask, b), _mm_andnot_ps(mask, a));
}

static inline float q4_0_signed_absmax_sse2(const float *RESTRICT x)
{
  const __m128 sign_mask = _mm_set1_ps(-0.0f);
  const __m128 step = _mm_set1_ps(4.0f);
  __m128 idx = _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f);
  __m128 best_idx = _mm_set1_ps(2147483648.0f);
  __m128 best_abs = _mm_setzero_ps();
  __m128 best_val = _mm_setzero_ps();

  for (int j = 0; j < QK4_0; j += 4)
  {
    const __m128 v = _mm_loadu_ps(x + j);
    const __m128 av = _mm_andnot_ps(sign_mask, v);
    const __m128 mask = _mm_cmpgt_ps(av, best_abs);
    best_abs = q4_0_blendv_ps_sse2(best_abs, av, mask);
    best_val = q4_0_blendv_ps_sse2(best_val, v, mask);
    best_idx = q4_0_blendv_ps_sse2(best_idx, idx, mask);
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

static inline __m128i q4_0_quantize_4_i32_sse2(const float *RESTRICT x, __m128 id, __m128 offset)
{
  const __m128 scaled = _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(x), id), offset);
  return _mm_cvttps_epi32(scaled);
}
#endif

extern "C" void quantize_row_q4_0_sse2(const float *RESTRICT x, block_q4_0 *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_SSE2)
  assert(k % QK4_0 == 0);
  static_assert(QK4_0 == 32, "QK4_0 must be 32");

  const int nb = k / QK4_0;
  const __m128 offset = _mm_set1_ps(8.5f);
  const __m128i max_q = _mm_set1_epi16(15);
  for (int i = 0; i < nb; ++i)
  {
    const float *xb = x + i * QK4_0;
    const float max = q4_0_signed_absmax_sse2(xb);
    const float d = max / -8;
    const float id = d ? 1.0f / d : 0.0f;
    const __m128 idv = _mm_set1_ps(id);
    y[i].d = GGML_FP32_TO_FP16(d);

    for (int j = 0; j < QK4_0 / 2; j += 8)
    {
      const __m128i lo0 = q4_0_quantize_4_i32_sse2(xb + j, idv, offset);
      const __m128i lo1 = q4_0_quantize_4_i32_sse2(xb + j + 4, idv, offset);
      const __m128i hi0 = q4_0_quantize_4_i32_sse2(xb + QK4_0 / 2 + j, idv, offset);
      const __m128i hi1 = q4_0_quantize_4_i32_sse2(xb + QK4_0 / 2 + j + 4, idv, offset);
      const __m128i lo = _mm_min_epi16(_mm_packs_epi32(lo0, lo1), max_q);
      const __m128i hi = _mm_min_epi16(_mm_packs_epi32(hi0, hi1), max_q);
      const __m128i packed = _mm_or_si128(lo, _mm_slli_epi16(hi, 4));
      const __m128i bytes = _mm_packus_epi16(packed, _mm_setzero_si128());
      _mm_storel_epi64((__m128i *)(y[i].qs + j), bytes);
    }
  }
#else
  GGML_UNUSED(x);
  GGML_UNUSED(y);
  GGML_UNUSED(k);
  GGML_UNREACHABLE();
#endif
}
