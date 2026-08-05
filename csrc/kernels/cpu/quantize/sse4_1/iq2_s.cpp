#include "kernels/cpu/common/common.h"
#include "kernels/cpu/quantize/kernels.h"
#include "quantize/iq_tables.h"
#include <libgguf/libgguf.h>

#include "kernels/cpu/common/common.h"
#include "kernels/cpu/quantize/kernels.h"
#include "quantize/iq_tables.h"

void quantize_block_iq2_s(const float *LIBGGUF_RESTRICT x, void *LIBGGUF_RESTRICT vy, int64_t n, const float *LIBGGUF_RESTRICT quant_weights)
{

  const int gindex = iq2_data_index(GGML_TYPE_IQ2_S);

  const uint64_t *kgrid_q2xs = iq2_data[gindex].grid;
  const int *kmap_q2xs = iq2_data[gindex].map;
  const uint16_t *kneighbors_q2xs = iq2_data[gindex].neighbours;

  assert(kmap_q2xs && "IQ tables are not initialized");
  assert(kgrid_q2xs && "IQ tables are not initialized");
  assert(kneighbors_q2xs && "IQ tables are not initialized");
  assert(n % QK_K == 0);

  const int kMaxQ = 3;

  const int64_t nbl = n / QK_K;

  block_iq2_s *y = (block_iq2_s *)vy;

  float scales[QK_K / 16];
  float weight[16];
  float xval[16];
  int8_t L[16];
  int8_t Laux[16];
  float waux[16];
  bool is_on_grid[2];
  bool is_on_grid_aux[2];
  uint8_t block_signs[2];

  for (int ibl = 0; ibl < nbl; ++ibl)
  {

    memset(&y[ibl], 0, sizeof(block_iq2_s));
    y[ibl].d = LIBGGUF_FP32_TO_FP16(0.f);

    float max_scale = 0;

    const float *xbl = x + QK_K * ibl;
    float sumx2 = 0;
    for (int i = 0; i < QK_K; ++i)
      sumx2 += xbl[i] * xbl[i];
    float sigma2 = 2 * sumx2 / QK_K;

    for (int ib = 0; ib < QK_K / 16; ++ib)
    {
      const float *xb = xbl + 16 * ib;
      if (quant_weights)
      {
        const float *qw = quant_weights + QK_K * ibl + 16 * ib;
        for (int i = 0; i < 16; ++i)
          weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
      }
      else
      {
        for (int i = 0; i < 16; ++i)
          weight[i] = 0.25f * sigma2 + xb[i] * xb[i];
      }
      for (int i = 0; i < 16; ++i)
        waux[i] = sqrtf(weight[i]);
      for (int k = 0; k < 2; ++k)
      {
        uint8_t s = 0;
        for (int i = 0; i < 8; ++i)
        {
          if (xb[8 * k + i] >= 0)
            xval[8 * k + i] = xb[8 * k + i];
          else
          {
            xval[8 * k + i] = -xb[8 * k + i];
            s |= (1 << i);
          }
        }
        block_signs[k] = s;
      }
      float max = xval[0];
      for (int i = 1; i < 16; ++i)
        max = libgguf_max(max, xval[i]);
      memset(L, 0, 16);
      if (max < GROUP_MAX_EPS_IQ2_S)
      {
        scales[ib] = 0;
        continue;
      }
      float best = 0;
      float scale = max / (2 * kMaxQ - 1);
      is_on_grid[0] = is_on_grid[1] = true;
      for (int is = -9; is <= 9; ++is)
      {
        float id = (2 * kMaxQ - 1 + is * 0.1f) / max;
        float this_scale = 1 / id;
        for (int k = 0; k < 2; ++k)
        {
          for (int i = 0; i < 8; ++i)
          {
            int l = nearest_int(0.5f * (id * xval[8 * k + i] - 1));
            Laux[8 * k + i] = libgguf_max(0, libgguf_min(kMaxQ - 1, l));
          }
          uint16_t u = 0;
          for (int i = 0; i < 8; ++i)
            u |= (Laux[8 * k + i] << 2 * i);
          int grid_index = kmap_q2xs[u];
          is_on_grid_aux[k] = true;
          if (grid_index < 0)
          {
            is_on_grid_aux[k] = false;
            const uint16_t *neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
            grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8 * k, waux + 8 * k, this_scale, Laux + 8 * k);
          }
        }
        float sumqx = 0, sumq2 = 0;
        for (int i = 0; i < 16; ++i)
        {
          float w = weight[i];
          float q = 2 * Laux[i] + 1;
          sumqx += w * xval[i] * q;
          sumq2 += w * q * q;
        }
        if (sumq2 > 0 && sumqx * sumqx > best * sumq2)
        {
          scale = sumqx / sumq2;
          best = scale * sumqx;
          for (int i = 0; i < 16; ++i)
            L[i] = Laux[i];
          for (int k = 0; k < 2; ++k)
            is_on_grid[k] = is_on_grid_aux[k];
        }
      }
      int n_not_ongrid = 0;
      for (int k = 0; k < 2; ++k)
        if (!is_on_grid[k])
          ++n_not_ongrid;
      if (n_not_ongrid > 0 && scale > 0)
      {
        float id = 1 / scale;
        for (int k = 0; k < 2; ++k)
        {
          if (is_on_grid[k])
            continue;
          uint16_t u = 0;
          for (int i = 0; i < 8; ++i)
          {
            int l = nearest_int(0.5f * (id * xval[8 * k + i] - 1));
            l = libgguf_max(0, libgguf_min(kMaxQ - 1, l));
            u |= (l << 2 * i);
            L[8 * k + i] = l;
          }
          int grid_index = kmap_q2xs[u];
          if (grid_index < 0)
          {
            const uint16_t *neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
            grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8 * k, waux + 8 * k, scale, L + 8 * k);
          }
        }
        float sumqx = 0, sumq2 = 0;
        for (int i = 0; i < 16; ++i)
        {
          float w = weight[i];
          float q = 2 * L[i] + 1;
          sumqx += w * xval[i] * q;
          sumq2 += w * q * q;
        }
        if (sumq2 > 0)
          scale = sumqx / sumq2;
      }
      if (scale < 0)
      {
        scale = -scale;
        for (int k = 0; k < 2; ++k)
          block_signs[k] = ~block_signs[k];
      }
      for (int k = 0; k < 2; ++k)
      {
        uint16_t u = 0;
        for (int i = 0; i < 8; ++i)
          u |= (L[8 * k + i] << 2 * i);
        int grid_index = kmap_q2xs[u];
        if (grid_index < 0)
        {
          printf("Oops: found point %u not on grid:", u);
          for (int i = 0; i < 8; ++i)
            printf(" %d", L[8 * k + i]);
          printf("\n");
          printf("fatal error");
          abort();
        }
        const int i8 = 2 * ib + k;
        y[ibl].qs[i8] = grid_index & 255;
        y[ibl].qh[i8 / 4] |= ((grid_index >> 8) << 2 * (i8 % 4));
        y[ibl].qs[QK_K / 8 + i8] = block_signs[k];
      }
      assert(scale >= 0);
      scales[ib] = scale;
      max_scale = libgguf_max(max_scale, scale);
    }

    if (!max_scale)
    {
      continue;
    }

    float d = max_scale / 31;
    y[ibl].d = LIBGGUF_FP32_TO_FP16(d * 0.9875f);
    float id = 1 / d;
    for (int ib = 0; ib < QK_K / 16; ++ib)
    {
      int l = nearest_int(0.5f * (id * scales[ib] - 1));
      l = libgguf_max(0, libgguf_min(15, l));
      if (ib % 2 == 0)
        y[ibl].scales[ib / 2] = l;
      else
        y[ibl].scales[ib / 2] |= (l << 4);
    }
  }
}



size_t quantize_iq2_s(const float *LIBGGUF_RESTRICT src, void *LIBGGUF_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  assert(n_per_row % QK_K == 0);
  int64_t nblock = n_per_row / QK_K;
  char *qrow = (char *)dst;
  for (int64_t row = 0; row < nrow; ++row)
  {
    quantize_block_iq2_s(src, qrow, n_per_row, quant_weights);
    src += n_per_row;
    qrow += nblock * sizeof(block_iq2_s);
  }
  return nrow * nblock * sizeof(block_iq2_s);
}

