#include "libgguf_common.h"
#include "libgguf_iq_tables.h"

static void quantize_row_iq2_xxs_impl(const float *RESTRICT x, void *RESTRICT vy, int64_t n, const float *RESTRICT quant_weights)
{

  const int gindex = iq2_data_index(GGML_TYPE_IQ2_XXS);

  const uint64_t *kgrid_q2xs = iq2_data[gindex].grid;
  const int *kmap_q2xs = iq2_data[gindex].map;
  const uint16_t *kneighbors_q2xs = iq2_data[gindex].neighbours;

  assert(quant_weights && "missing quantization weights");
  assert(kgrid_q2xs && "forgot to call ggml_quantize_init()?");
  assert(kmap_q2xs && "forgot to call ggml_quantize_init()?");
  assert(kneighbors_q2xs && "forgot to call ggml_quantize_init()?");
  assert(n % QK_K == 0);

  const int kMaxQ = 3;

  const int64_t nbl = n / QK_K;

  block_iq2_xxs *y = (block_iq2_xxs *)vy;

  float scales[QK_K / 32];
  float weight[32];
  float xval[32];
  int8_t L[32];
  int8_t Laux[32];
  float waux[32];
  uint8_t block_signs[4];
  uint32_t q2[2 * (QK_K / 32)];

  for (int ibl = 0; ibl < nbl; ++ibl)
  {

    y[ibl].d = GGML_FP32_TO_FP16(0.f);
    memset(q2, 0, QK_K / 4);

    float max_scale = 0;

    const float *xbl = x + QK_K * ibl;
    float sumx2 = 0;
    for (int i = 0; i < QK_K; ++i)
      sumx2 += xbl[i] * xbl[i];
    float sigma2 = sumx2 / QK_K;

    for (int ib = 0; ib < QK_K / 32; ++ib)
    {
      const float *xb = xbl + 32 * ib;
      const float *qw = quant_weights + QK_K * ibl + 32 * ib;
      for (int i = 0; i < 32; ++i)
        weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
      for (int i = 0; i < 32; ++i)
        waux[i] = sqrtf(weight[i]);
      for (int k = 0; k < 4; ++k)
      {
        int nflip = 0;
        uint8_t s = 0;
        for (int i = 0; i < 8; ++i)
        {
          if (xb[8 * k + i] >= 0)
            xval[8 * k + i] = xb[8 * k + i];
          else
          {
            xval[8 * k + i] = -xb[8 * k + i];
            ++nflip;
            s |= (1 << i);
          }
        }
        if (nflip % 2)
        {
          int imin = 0;
          float min = weight[8 * k + imin] * xb[8 * k + imin] * xb[8 * k + imin];
          for (int i = 1; i < 8; ++i)
          {
            float ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
            if (ax < min)
            {
              min = ax;
              imin = i;
            }
          }
          xval[8 * k + imin] = -xval[8 * k + imin];
          s ^= (1 << imin);
        }
        block_signs[k] = s & 127;
      }
      float max = xval[0];
      for (int i = 1; i < 32; ++i)
        max = MAX(max, xval[i]);
      if (max < GROUP_MAX_EPS)
      {
        scales[ib] = 0;
        memset(L, 0, 32);
        continue;
      }
      float scale = make_qp_quants(32, kMaxQ + 1, xval, (uint8_t *)L, weight);
      float eff_max = scale * kMaxQ;
      if (eff_max <= 0)
      {
        scales[ib] = 0;
        memset(L, 0, 32);
        continue;
      }
      float best = 0;
      for (int is = -6; is <= 6; ++is)
      {
        float id = (2 * kMaxQ - 1 + is * 0.1f) / eff_max;
        float this_scale = 1 / id;
        for (int k = 0; k < 4; ++k)
        {
          for (int i = 0; i < 8; ++i)
          {
            int l = nearest_int(0.5f * (id * xval[8 * k + i] - 1));
            Laux[8 * k + i] = MAX(0, MIN(kMaxQ - 1, l));
          }
          uint16_t u = 0;
          for (int i = 0; i < 8; ++i)
            u |= (Laux[8 * k + i] << 2 * i);
          int grid_index = kmap_q2xs[u];
          if (grid_index < 0)
          {
            const uint16_t *neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
            grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8 * k, waux + 8 * k, this_scale, Laux + 8 * k);
          }
        }
        float sumqx = 0, sumq2 = 0;
        for (int i = 0; i < 32; ++i)
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
          memcpy(L, Laux, 32);
        }
      }
      if (scale > 0)
      {
        float id = 1 / scale;
        for (int k = 0; k < 4; ++k)
        {
          uint16_t u = 0;
          for (int i = 0; i < 8; ++i)
          {
            int l = nearest_int(0.5f * (id * xval[8 * k + i] - 1));
            l = MAX(0, MIN(kMaxQ - 1, l));
            u |= (l << 2 * i);
          }
          int grid_index = kmap_q2xs[u];
          if (grid_index < 0)
          {
            const uint16_t *neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
            grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8 * k, waux + 8 * k, scale, L + 8 * k);
          }
          const int8_t *pg = (const int8_t *)(kgrid_q2xs + grid_index);
          for (int i = 0; i < 8; ++i)
            L[8 * k + i] = (pg[i] - 1) / 2;
        }
        float sumqx = 0, sumq2 = 0;
        for (int i = 0; i < 32; ++i)
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
        // This should never happen, but just in case, flip scale so that it is positive (we use uint's to encode the scale)
        // and correspondingly flip quant signs.
        scale = -scale;
        for (int k = 0; k < 4; ++k)
          block_signs[k] = (~block_signs[k]) & 127;
      }
      for (int k = 0; k < 4; ++k)
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
        q2[2 * ib + 0] |= ((uint32_t)grid_index << 8 * k);
        q2[2 * ib + 1] |= (block_signs[k] << 7 * k);
      }
      assert(scale >= 0);
      scales[ib] = scale;
      max_scale = MAX(max_scale, scale);
    }

    if (!max_scale)
    {
      memset(y[ibl].qs, 0, QK_K / 4);
      continue;
    }

    float d = max_scale / 31;
    y[ibl].d = GGML_FP32_TO_FP16(d);
    float id = 1 / d;
    for (int ib = 0; ib < QK_K / 32; ++ib)
    {
      int l = nearest_int(0.5f * (id * scales[ib] - 1));
      l = MAX(0, MIN(15, l));
      q2[2 * ib + 1] |= ((uint32_t)l << 28);
    }
    memcpy(y[ibl].qs, q2, QK_K / 4);
  }
}

size_t quantize_iq2_xxs(const float *RESTRICT src, void *RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  assert(n_per_row % QK_K == 0);
  int64_t nblock = n_per_row / QK_K;
  char *qrow = (char *)dst;
  for (int64_t row = 0; row < nrow; ++row)
  {
    quantize_row_iq2_xxs_impl(src, qrow, n_per_row, quant_weights);
    src += n_per_row;
    qrow += nblock * sizeof(block_iq2_xxs);
  }
  return nrow * nblock * sizeof(block_iq2_xxs);
}

