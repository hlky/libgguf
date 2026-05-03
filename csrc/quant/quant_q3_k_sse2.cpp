#include "libgguf_common.h"

extern "C" float libgguf_make_q3_quants_sse2(int n, int nmax, const float *RESTRICT x, int8_t *RESTRICT L,
                                              bool do_rmse);

#if defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(__SSE2__)
#include <emmintrin.h>
#include <xmmintrin.h>
#define LIBGGUF_BUILD_Q3_K_SSE2 1
#endif

#if defined(LIBGGUF_BUILD_Q3_K_SSE2)
static inline __m128i q3_k_nearest_i32_sse2(__m128 v)
{
  const __m128 magic = _mm_set1_ps(12582912.0f);
  return _mm_sub_epi32(
      _mm_and_si128(_mm_castps_si128(_mm_add_ps(v, magic)), _mm_set1_epi32(0x007fffff)),
      _mm_set1_epi32(0x00400000));
}

static inline void q3_k_quantize_16_sse2(const float *RESTRICT x, int8_t *RESTRICT L, float d)
{
  const __m128 dv = _mm_set1_ps(d);
  const __m128i min_q = _mm_set1_epi16(-4);
  const __m128i max_q = _mm_set1_epi16(3);
  const __m128i offset = _mm_set1_epi16(4);
  const __m128i zero = _mm_setzero_si128();

  for (int ii = 0; ii < 16; ii += 8)
  {
    const __m128 v0 = _mm_div_ps(_mm_loadu_ps(x + ii), dv);
    const __m128 v1 = _mm_div_ps(_mm_loadu_ps(x + ii + 4), dv);
    __m128i q = _mm_packs_epi32(q3_k_nearest_i32_sse2(v0), q3_k_nearest_i32_sse2(v1));
    q = _mm_add_epi16(_mm_min_epi16(_mm_max_epi16(q, min_q), max_q), offset);
    q = _mm_packus_epi16(q, zero);
    _mm_storel_epi64((__m128i *)(L + ii), q);
  }
}

static inline void q3_k_update_hmask_32_sse2(int8_t *RESTRICT L, uint8_t *RESTRICT hmask, uint8_t hm)
{
  const __m128i threshold = _mm_set1_epi8(3);
  const __m128i high_bit = _mm_set1_epi8((char)hm);
  const __m128i four = _mm_set1_epi8(4);
  for (int l = 0; l < 32; l += 16)
  {
    const __m128i q = _mm_loadu_si128((const __m128i *)(L + l));
    const __m128i high = _mm_cmpgt_epi8(q, threshold);
    const __m128i old_h = _mm_loadu_si128((const __m128i *)(hmask + l));
    _mm_storeu_si128((__m128i *)(hmask + l), _mm_or_si128(old_h, _mm_and_si128(high, high_bit)));
    _mm_storeu_si128((__m128i *)(L + l), _mm_sub_epi8(q, _mm_and_si128(high, four)));
  }
}

static inline void q3_k_pack_128_sse2(const int8_t *RESTRICT L, uint8_t *RESTRICT q)
{
  const __m128i mask = _mm_set1_epi8(0x03);
  for (int l = 0; l < 32; l += 16)
  {
    const __m128i q0 = _mm_and_si128(_mm_loadu_si128((const __m128i *)(L + l + 0)), mask);
    const __m128i q1 = _mm_slli_epi16(_mm_and_si128(_mm_loadu_si128((const __m128i *)(L + l + 32)), mask), 2);
    const __m128i q2 = _mm_slli_epi16(_mm_and_si128(_mm_loadu_si128((const __m128i *)(L + l + 64)), mask), 4);
    const __m128i q3 = _mm_slli_epi16(_mm_and_si128(_mm_loadu_si128((const __m128i *)(L + l + 96)), mask), 6);
    _mm_storeu_si128((__m128i *)(q + l), _mm_or_si128(_mm_or_si128(q0, q1), _mm_or_si128(q2, q3)));
  }
}
#endif

extern "C" void quantize_row_q3_K_sse2(const float *RESTRICT x, block_q3_K *RESTRICT y, int64_t k)
{
#if defined(LIBGGUF_BUILD_Q3_K_SSE2)
  assert(k % QK_K == 0);
  const int64_t nb = k / QK_K;

  int8_t L[QK_K];
  float scales[QK_K / 16];

  for (int64_t i = 0; i < nb; i++)
  {
    float max_scale = 0;
    float amax = 0;
    for (int j = 0; j < QK_K / 16; ++j)
    {
      scales[j] = libgguf_make_q3_quants_sse2(16, 4, x + 16 * j, L + 16 * j, true);
      const float scale = fabsf(scales[j]);
      if (scale > amax)
      {
        amax = scale;
        max_scale = scales[j];
      }
    }

    memset(y[i].scales, 0, 12);
    if (max_scale)
    {
      const float iscale = -32.f / max_scale;
      for (int j = 0; j < QK_K / 16; ++j)
      {
        int8_t l = (int8_t)nearest_int(iscale * scales[j]);
        l = (int8_t)(MAX(-32, MIN(31, l)) + 32);
        if (j < 8)
        {
          y[i].scales[j] = l & 0xF;
        }
        else
        {
          y[i].scales[j - 8] |= (uint8_t)((l & 0xF) << 4);
        }
        l >>= 4;
        y[i].scales[j % 4 + 8] |= (uint8_t)(l << (2 * (j / 4)));
      }
      y[i].d = GGML_FP32_TO_FP16(1 / iscale);
    }
    else
    {
      y[i].d = GGML_FP32_TO_FP16(0.f);
    }

    const float d_base = GGML_FP16_TO_FP32(y[i].d);
    for (int j = 0; j < QK_K / 16; ++j)
    {
      int8_t sc = j < 8 ? y[i].scales[j] & 0xF : y[i].scales[j - 8] >> 4;
      sc = (int8_t)((sc | (((y[i].scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) - 32);
      const float d = d_base * sc;
      if (!d)
      {
        continue;
      }
      q3_k_quantize_16_sse2(x + 16 * j, L + 16 * j, d);
    }

    memset(y[i].hmask, 0, QK_K / 8);
    uint8_t hm = 1;
    for (int j = 0; j < QK_K; j += 32)
    {
      q3_k_update_hmask_32_sse2(L + j, y[i].hmask, hm);
      hm <<= 1;
    }

    for (int j = 0; j < QK_K; j += 128)
    {
      q3_k_pack_128_sse2(L + j, y[i].qs + j / 4);
    }

    x += QK_K;
  }
#else
  quantize_row_q3_K_ref(x, y, k);
#endif
}
