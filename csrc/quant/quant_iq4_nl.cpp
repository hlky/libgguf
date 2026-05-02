#include "libgguf_common.h"
#include "libgguf_tables.h"

static void quantize_row_iq4_nl_impl(const int super_block_size, const int block_size, const float *RESTRICT x,
                                     ggml_fp16_t *dh, uint8_t *q4, uint16_t *scales_h, uint8_t *scales_l,
                                     float *scales, float *weight, uint8_t *L,
                                     const int8_t *values,
                                     const float *quant_weights,
                                     const int ntry)
{

  float sigma2 = 0;
  for (int j = 0; j < super_block_size; ++j)
    sigma2 += x[j] * x[j];
  sigma2 *= 2.f / super_block_size;

  memset(q4, 0, super_block_size / 2);
  dh[0] = GGML_FP32_TO_FP16(0.f);

  float max_scale = 0, amax_scale = 0;
  for (int ib = 0; ib < super_block_size / block_size; ++ib)
  {
    const float *xb = x + ib * block_size;
    uint8_t *Lb = L + ib * block_size;
    if (quant_weights)
    {
      const float *qw = quant_weights + ib * block_size;
      for (int j = 0; j < block_size; ++j)
        weight[j] = qw[j] * sqrtf(sigma2 + xb[j] * xb[j]);
    }
    else
    {
      for (int j = 0; j < block_size; ++j)
        weight[j] = xb[j] * xb[j];
    }
    float amax = 0, max = 0;
    for (int j = 0; j < block_size; ++j)
    {
      float ax = fabsf(xb[j]);
      if (ax > amax)
      {
        amax = ax;
        max = xb[j];
      }
    }
    if (amax < GROUP_MAX_EPS)
    {
      scales[ib] = 0;
      continue;
    }
    float d = ntry > 0 ? -max / values[0] : max / values[0];
    float id = 1 / d;
    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < block_size; ++j)
    {
      float al = id * xb[j];
      int l = best_index_int8(16, values, al);
      Lb[j] = l;
      float q = values[l];
      float w = weight[j];
      sumqx += w * q * xb[j];
      sumq2 += w * q * q;
    }
    d = sumq2 > 0 ? sumqx / sumq2 : 0.f;
    float best = d * sumqx;
    for (int itry = -ntry; itry <= ntry; ++itry)
    {
      id = (itry + values[0]) / max;
      sumqx = sumq2 = 0;
      for (int j = 0; j < block_size; ++j)
      {
        float al = id * xb[j];
        int l = best_index_int8(16, values, al);
        float q = values[l];
        float w = weight[j];
        sumqx += w * q * xb[j];
        sumq2 += w * q * q;
      }
      if (sumq2 > 0 && sumqx * sumqx > best * sumq2)
      {
        d = sumqx / sumq2;
        best = d * sumqx;
      }
    }
    scales[ib] = d;
    float abs_d = fabsf(d);
    if (abs_d > amax_scale)
    {
      amax_scale = abs_d;
      max_scale = d;
    }
  }

  if (super_block_size / block_size > 1)
  {
    int nb = super_block_size / block_size;
    memset(scales_h, 0, ((nb + 7) / 8) * sizeof(uint16_t));
    float d = -max_scale / 32;
    dh[0] = GGML_FP32_TO_FP16(d);
    float id = d ? 1 / d : 0.f;
    for (int ib = 0; ib < super_block_size / block_size; ++ib)
    {
      int l = nearest_int(id * scales[ib]);
      l = MAX(-32, MIN(31, l));
      float dl = d * l;
      float idl = dl ? 1 / dl : 0.f;
      uint8_t *Lb = L + ib * block_size;
      const float *xb = x + ib * block_size;
      for (int j = 0; j < block_size; ++j)
      {
        Lb[j] = best_index_int8(16, values, idl * xb[j]);
      }
      l += 32;
      uint8_t l_l = l & 0xf;
      uint8_t l_h = l >> 4;
      if (ib % 2 == 0)
        scales_l[ib / 2] = l_l;
      else
        scales_l[ib / 2] |= (l_l << 4);
      scales_h[ib / 8] |= (l_h << 2 * (ib % 8));
    }
  }
  else
  {
    dh[0] = GGML_FP32_TO_FP16(scales[0]);
    if (ntry > 0)
    {
      float id = scales[0] ? 1 / scales[0] : 0;
      for (int j = 0; j < super_block_size; ++j)
      {
        L[j] = best_index_int8(16, values, id * x[j]);
      }
    }
  }

  for (int i = 0; i < super_block_size / 32; ++i)
  {
    for (int j = 0; j < 16; ++j)
    {
      q4[16 * i + j] = L[32 * i + j] | (L[32 * i + 16 + j] << 4);
    }
  }
}

size_t quantize_iq4_nl(const float *RESTRICT src, void *RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  assert(n_per_row % QK4_NL == 0);
  int64_t nblock = n_per_row / QK4_NL;
  char *qrow = (char *)dst;
  uint8_t L[QK4_NL];
  float weight[QK4_NL];
  uint16_t unused_h;
  uint8_t *unused_l = nullptr;
  float scale;
  for (int64_t row = 0; row < nrow; ++row)
  {
    block_iq4_nl *iq4 = (block_iq4_nl *)qrow;
    for (int ibl = 0; ibl < nblock; ++ibl)
    {
      const float *qw = quant_weights ? quant_weights + QK4_NL * ibl : nullptr;
      quantize_row_iq4_nl_impl(QK4_NL, 32, src + QK4_NL * ibl, &iq4[ibl].d, iq4[ibl].qs, &unused_h, unused_l,
                               &scale, weight, L, kvalues_iq4nl, qw, 7);
    }
    src += n_per_row;
    qrow += nblock * sizeof(block_iq4_nl);
  }
  return nrow * nblock * sizeof(block_iq4_nl);
}

// void quantize_row_iq4_nl_ref(const float * RESTRICT x, void * RESTRICT vy, int64_t k) {
void quantize_row_iq4_nl_ref(const float *RESTRICT x, block_iq4_nl *RESTRICT y, int64_t k)
{
  assert(k % QK4_NL == 0);
  int64_t nblock = k / QK4_NL;
  uint8_t L[QK4_NL];
  float weight[QK4_NL];
  uint16_t unused_h;
  uint8_t *unused_l = nullptr;
  float scale;
  block_iq4_nl *iq4 = y;
  for (int ibl = 0; ibl < nblock; ++ibl)
  {
    quantize_row_iq4_nl_impl(QK4_NL, 32, x + QK4_NL * ibl, &iq4[ibl].d, iq4[ibl].qs, &unused_h, unused_l,
                             &scale, weight, L, kvalues_iq4nl, nullptr, -1);
  }
}

