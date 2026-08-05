#include "kernels/cpu/common/common.h"
#include "kernels/cpu/quantize/kernels.h"
#include "quantize/iq_tables.h"
#include <libgguf/libgguf.h>


#include <emmintrin.h>
#include <xmmintrin.h>

static inline __m128i q2_k_nearest_i32(__m128 v)
{
  const __m128 magic = _mm_set1_ps(12582912.0f);
  return _mm_sub_epi32(
      _mm_and_si128(_mm_castps_si128(_mm_add_ps(v, magic)), _mm_set1_epi32(0x007fffff)),
      _mm_set1_epi32(0x00400000));
}

static inline void q2_k_fill_abs_16(const float *LIBGGUF_RESTRICT x, float *LIBGGUF_RESTRICT weights)
{
  const __m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
  for (int l = 0; l < 16; l += 4)
  {
    _mm_storeu_ps(weights + l, _mm_and_ps(_mm_loadu_ps(x + l), abs_mask));
  }
}

static inline void q2_k_quantize_16(const float *LIBGGUF_RESTRICT x, uint8_t *LIBGGUF_RESTRICT L, float d, float dm)
{
  const __m128 dv = _mm_set1_ps(d);
  const __m128 dmv = _mm_set1_ps(dm);
  const __m128i zero = _mm_setzero_si128();
  const __m128i max_q = _mm_set1_epi16(3);

  for (int ii = 0; ii < 16; ii += 8)
  {
    const __m128 v0 = _mm_div_ps(_mm_add_ps(_mm_loadu_ps(x + ii), dmv), dv);
    const __m128 v1 = _mm_div_ps(_mm_add_ps(_mm_loadu_ps(x + ii + 4), dmv), dv);
    __m128i q = _mm_packs_epi32(q2_k_nearest_i32(v0), q2_k_nearest_i32(v1));
    q = _mm_min_epi16(_mm_max_epi16(q, zero), max_q);
    q = _mm_packus_epi16(q, zero);
    _mm_storel_epi64((__m128i *)(L + ii), q);
  }
}

static inline void q2_k_pack_128(const uint8_t *LIBGGUF_RESTRICT L, uint8_t *LIBGGUF_RESTRICT q)
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

extern "C" void quantize_block_q2_k(const float *LIBGGUF_RESTRICT x, block_q2_K *LIBGGUF_RESTRICT y, int64_t k)
{
  assert(k % QK_K == 0);
  const int64_t nb = k / QK_K;

  uint8_t L[QK_K];
  uint8_t Laux[16];
  float weights[16];
  float mins[QK_K / 16];
  float scales[QK_K / 16];
  const float q4scale = 15.f;

  for (int64_t i = 0; i < nb; i++)
  {
    float max_scale = 0;
    float max_min = 0;
    for (int j = 0; j < QK_K / 16; ++j)
    {
      q2_k_fill_abs_16(x + 16 * j, weights);
      scales[j] = make_qkx2_quants(16, 3, x + 16 * j, weights, L + 16 * j, &mins[j], Laux,
                                                  -0.5f, 0.1f, 15, true);
      if (scales[j] > max_scale)
      {
        max_scale = scales[j];
      }
      if (mins[j] > max_min)
      {
        max_min = mins[j];
      }
    }

    if (max_scale > 0)
    {
      const float iscale = q4scale / max_scale;
      for (int j = 0; j < QK_K / 16; ++j)
      {
        y[i].scales[j] = (uint8_t)nearest_int(iscale * scales[j]);
      }
      y[i].d = LIBGGUF_FP32_TO_FP16(max_scale / q4scale);
    }
    else
    {
      memset(y[i].scales, 0, QK_K / 16);
      y[i].d = LIBGGUF_FP32_TO_FP16(0.f);
    }

    if (max_min > 0)
    {
      const float iscale = q4scale / max_min;
      for (int j = 0; j < QK_K / 16; ++j)
      {
        y[i].scales[j] |= (uint8_t)(nearest_int(iscale * mins[j]) << 4);
      }
      y[i].dmin = LIBGGUF_FP32_TO_FP16(max_min / q4scale);
    }
    else
    {
      y[i].dmin = LIBGGUF_FP32_TO_FP16(0.f);
    }

    const float d_base = LIBGGUF_FP16_TO_FP32(y[i].d);
    const float dm_base = LIBGGUF_FP16_TO_FP32(y[i].dmin);
    for (int j = 0; j < QK_K / 16; ++j)
    {
      const float d = d_base * (y[i].scales[j] & 0xF);
      if (!d)
      {
        continue;
      }
      const float dm = dm_base * (y[i].scales[j] >> 4);
      q2_k_quantize_16(x + 16 * j, L + 16 * j, d, dm);
    }

    for (int j = 0; j < QK_K; j += 128)
    {
      q2_k_pack_128(L + j, y[i].qs + j / 4);
    }

    x += QK_K;
  }
}


static void quantize_weighted_q2_K(const float *LIBGGUF_RESTRICT x, block_q2_K *LIBGGUF_RESTRICT y, int k, const float *LIBGGUF_RESTRICT quant_weights)
{
  assert(quant_weights);
  assert(k % QK_K == 0);
  const int nb = k / QK_K;
  const bool requantize = true;

  uint8_t L[QK_K];
  uint8_t Laux[16];
  float mins[QK_K / 16];
  float scales[QK_K / 16];
  float sw[QK_K / 16];
  float weight[16];
  uint8_t Ls[QK_K / 16], Lm[QK_K / 16];

  for (int i = 0; i < nb; i++)
  {
    memset(sw, 0, QK_K / 16 * sizeof(float));
    float sumx2 = 0;
    for (int j = 0; j < QK_K; ++j)
      sumx2 += x[j] * x[j];
    float sigma2 = sumx2 / QK_K;
    for (int j = 0; j < QK_K / 16; ++j)
    {
      const float *LIBGGUF_RESTRICT qw = quant_weights + QK_K * i + 16 * j;
      for (int l = 0; l < 16; ++l)
        weight[l] = qw[l] * sqrtf(sigma2 + x[16 * j + l] * x[16 * j + l]);
      for (int l = 0; l < QK_K / 16; ++l)
        sw[j] += weight[l];
      scales[j] = make_qkx3_quants(16, 3, x + 16 * j, weight, L + 16 * j, &mins[j], Laux, -0.9f, 0.05f, 36, false);
    }

    float dm, mm;
    dm = make_qp_quants(QK_K / 16, 15, scales, Ls, sw);
    mm = make_qp_quants(QK_K / 16, 15, mins, Lm, sw);

    y[i].d = LIBGGUF_FP32_TO_FP16(dm);
    y[i].dmin = LIBGGUF_FP32_TO_FP16(mm);
    dm = LIBGGUF_FP16_TO_FP32(y[i].d);
    mm = LIBGGUF_FP16_TO_FP32(y[i].dmin);

    for (int j = 0; j < QK_K / 16; ++j)
    {
      y[i].scales[j] = Ls[j] | (Lm[j] << 4);
    }

    if (requantize)
    {
      for (int j = 0; j < QK_K / 16; ++j)
      {
        const float d = dm * (y[i].scales[j] & 0xF);
        if (!d)
          continue;
        const float m = mm * (y[i].scales[j] >> 4);
        for (int ii = 0; ii < 16; ++ii)
        {
          int l = nearest_int((x[16 * j + ii] + m) / d);
          l = libgguf_max(0, libgguf_min(3, l));
          L[16 * j + ii] = l;
        }
      }
    }

    for (int j = 0; j < QK_K; j += 128)
    {
      for (int l = 0; l < 32; ++l)
      {
        y[i].qs[j / 4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
      }
    }

    x += QK_K;
  }
}

size_t quantize_q2_k(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  size_t row_size = libgguf_row_size(GGML_TYPE_Q2_K, n_per_row);
  if (!quant_weights)
  {
    quantize_block_q2_k(src, (block_q2_K *)dst, (int64_t)nrow * n_per_row);
  }
  else
  {
    char *qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row)
    {
      quantize_weighted_q2_K(src, (block_q2_K *)qrow, n_per_row, quant_weights);
      src += n_per_row;
      qrow += row_size;
    }
  }
  return nrow * row_size;
}

