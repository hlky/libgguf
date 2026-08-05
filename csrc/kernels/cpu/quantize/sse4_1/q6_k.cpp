#include "kernels/cpu/common/common.h"
#include "kernels/cpu/quantize/kernels.h"
#include "quantize/iq_tables.h"
#include <libgguf/libgguf.h>


#include <emmintrin.h>
#include <xmmintrin.h>

static inline __m128i q6_k_nearest_i32(__m128 v)
{
  const __m128 magic = _mm_set1_ps(12582912.0f);
  return _mm_sub_epi32(
      _mm_and_si128(_mm_castps_si128(_mm_add_ps(v, magic)), _mm_set1_epi32(0x007fffff)),
      _mm_set1_epi32(0x00400000));
}

static inline void q6_k_quantize_16(const float *LIBGGUF_RESTRICT x, int8_t *LIBGGUF_RESTRICT L, float d)
{
  const __m128 dv = _mm_set1_ps(d);
  const __m128i min_q = _mm_set1_epi16(-32);
  const __m128i max_q = _mm_set1_epi16(31);
  const __m128i offset = _mm_set1_epi16(32);
  const __m128i zero = _mm_setzero_si128();

  for (int ii = 0; ii < 16; ii += 8)
  {
    const __m128 v0 = _mm_div_ps(_mm_loadu_ps(x + ii), dv);
    const __m128 v1 = _mm_div_ps(_mm_loadu_ps(x + ii + 4), dv);
    __m128i q = _mm_packs_epi32(q6_k_nearest_i32(v0), q6_k_nearest_i32(v1));
    q = _mm_add_epi16(_mm_min_epi16(_mm_max_epi16(q, min_q), max_q), offset);
    q = _mm_packus_epi16(q, zero);
    _mm_storel_epi64((__m128i *)(L + ii), q);
  }
}

static inline void q6_k_pack_128(const int8_t *LIBGGUF_RESTRICT L, uint8_t *LIBGGUF_RESTRICT ql, uint8_t *LIBGGUF_RESTRICT qh)
{
  const __m128i low_mask = _mm_set1_epi8(0x0F);
  const __m128i high_mask = _mm_set1_epi8(0x30);
  for (int l = 0; l < 32; l += 16)
  {
    const __m128i q0 = _mm_loadu_si128((const __m128i *)(L + l + 0));
    const __m128i q1 = _mm_loadu_si128((const __m128i *)(L + l + 32));
    const __m128i q2 = _mm_loadu_si128((const __m128i *)(L + l + 64));
    const __m128i q3 = _mm_loadu_si128((const __m128i *)(L + l + 96));
    const __m128i ql0 = _mm_or_si128(_mm_and_si128(q0, low_mask),
                                     _mm_slli_epi16(_mm_and_si128(q2, low_mask), 4));
    const __m128i ql1 = _mm_or_si128(_mm_and_si128(q1, low_mask),
                                     _mm_slli_epi16(_mm_and_si128(q3, low_mask), 4));
    const __m128i h0 = _mm_srli_epi16(_mm_and_si128(q0, high_mask), 4);
    const __m128i h1 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(q1, high_mask), 4), 2);
    const __m128i h2 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(q2, high_mask), 4), 4);
    const __m128i h3 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(q3, high_mask), 4), 6);
    _mm_storeu_si128((__m128i *)(ql + l), ql0);
    _mm_storeu_si128((__m128i *)(ql + l + 32), ql1);
    _mm_storeu_si128((__m128i *)(qh + l), _mm_or_si128(_mm_or_si128(h0, h1), _mm_or_si128(h2, h3)));
  }
}

extern "C" void quantize_block_q6_k(const float *LIBGGUF_RESTRICT x, block_q6_K *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK_K == 0);
  const int64_t nb = k / QK_K;

  int8_t L[QK_K];
  float scales[QK_K / 16];

  for (int64_t i = 0; i < nb; i++)
  {
    float max_scale = 0;
    float max_abs_scale = 0;

    for (int ib = 0; ib < QK_K / 16; ++ib)
    {
      const float scale = make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib, 1, nullptr);
      scales[ib] = scale;
      const float abs_scale = fabsf(scale);
      if (abs_scale > max_abs_scale)
      {
        max_abs_scale = abs_scale;
        max_scale = scale;
      }
    }

    if (max_abs_scale < GROUP_MAX_EPS)
    {
      memset(&y[i], 0, sizeof(block_q6_K));
      y[i].d = LIBGGUF_FP32_TO_FP16(0.f);
      x += QK_K;
      continue;
    }

    const float iscale = -128.f / max_scale;
    y[i].d = LIBGGUF_FP32_TO_FP16(1 / iscale);
    for (int ib = 0; ib < QK_K / 16; ++ib)
    {
      y[i].scales[ib] = (int8_t)libgguf_min(127, nearest_int(iscale * scales[ib]));
    }

    const float d_base = LIBGGUF_FP16_TO_FP32(y[i].d);
    for (int j = 0; j < QK_K / 16; ++j)
    {
      const float d = d_base * y[i].scales[j];
      if (!d)
      {
        continue;
      }
      q6_k_quantize_16(x + 16 * j, L + 16 * j, d);
    }

    uint8_t *LIBGGUF_RESTRICT ql = y[i].ql;
    uint8_t *LIBGGUF_RESTRICT qh = y[i].qh;
    for (int j = 0; j < QK_K; j += 128)
    {
      q6_k_pack_128(L + j, ql, qh);
      ql += 64;
      qh += 32;
    }

    x += QK_K;
  }
}


static void quantize_weighted_q6_K(const float *LIBGGUF_RESTRICT x, block_q6_K *LIBGGUF_RESTRICT y, int64_t n_per_row, const float *quant_weights)
{
  assert(n_per_row % QK_K == 0);
  const int64_t nb = n_per_row / QK_K;

  int8_t L[QK_K];
  float scales[QK_K / 16];
  // float   weights[16];

  for (int i = 0; i < nb; i++)
  {

    // float sum_x2 = 0;
    // for (int j = 0; j < QK_K; ++j) sum_x2 += x[j]*x[j];
    // float sigma2 = sum_x2/QK_K;

    float max_scale = 0;
    float max_abs_scale = 0;

    for (int ib = 0; ib < QK_K / 16; ++ib)
    {

      float scale;
      if (quant_weights)
      {
        const float *qw = quant_weights + QK_K * i + 16 * ib;
        // for (int j = 0; j < 16; ++j) weights[j] = qw[j] * sqrtf(sigma2 + x[16*ib + j]*x[16*ib + j]);
        // scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, weights);
        scale = make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib, 1, qw);
      }
      else
      {
        scale = make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib, 1, nullptr);
      }
      scales[ib] = scale;

      const float abs_scale = fabsf(scale);
      if (abs_scale > max_abs_scale)
      {
        max_abs_scale = abs_scale;
        max_scale = scale;
      }
    }

    if (max_abs_scale < GROUP_MAX_EPS)
    {
      memset(&y[i], 0, sizeof(block_q6_K));
      y[i].d = LIBGGUF_FP32_TO_FP16(0.f);
      x += QK_K;
      continue;
    }

    float iscale = -128.f / max_scale;
    y[i].d = LIBGGUF_FP32_TO_FP16(1 / iscale);
    for (int ib = 0; ib < QK_K / 16; ++ib)
    {
      y[i].scales[ib] = libgguf_min(127, nearest_int(iscale * scales[ib]));
    }

    for (int j = 0; j < QK_K / 16; ++j)
    {
      float d = LIBGGUF_FP16_TO_FP32(y[i].d) * y[i].scales[j];
      if (!d)
      {
        continue;
      }
      for (int ii = 0; ii < 16; ++ii)
      {
        int l = nearest_int(x[16 * j + ii] / d);
        l = libgguf_max(-32, libgguf_min(31, l));
        L[16 * j + ii] = l + 32;
      }
    }

    uint8_t *LIBGGUF_RESTRICT ql = y[i].ql;
    uint8_t *LIBGGUF_RESTRICT qh = y[i].qh;
    for (int j = 0; j < QK_K; j += 128)
    {
      for (int l = 0; l < 32; ++l)
      {
        const uint8_t q1 = L[j + l + 0] & 0xF;
        const uint8_t q2 = L[j + l + 32] & 0xF;
        const uint8_t q3 = L[j + l + 64] & 0xF;
        const uint8_t q4 = L[j + l + 96] & 0xF;
        ql[l + 0] = q1 | (q3 << 4);
        ql[l + 32] = q2 | (q4 << 4);
        qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
      }
      ql += 64;
      qh += 32;
    }

    x += QK_K;
  }
}

size_t quantize_q6_k(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  size_t row_size = libgguf_row_size(GGML_TYPE_Q6_K, n_per_row);
  if (!quant_weights)
  {
    quantize_block_q6_k(src, (block_q6_K *)dst, (int64_t)nrow * n_per_row);
  }
  else
  {
    char *qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row)
    {
      quantize_weighted_q6_K(src, (block_q6_K *)qrow, n_per_row, quant_weights);
      src += n_per_row;
      qrow += row_size;
    }
  }
  return nrow * row_size;
}

