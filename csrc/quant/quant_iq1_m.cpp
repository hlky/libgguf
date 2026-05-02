#include "libgguf_common.h"
#include "libgguf_iq_tables.h"

static void quantize_row_iq1_m_impl(const float *RESTRICT x, void *RESTRICT vy, int64_t n, const float *RESTRICT quant_weights,
                                    float *scales,
                                    float *weight,
                                    float *pairs,
                                    int8_t *L,
                                    uint16_t *index,
                                    int8_t *shifts)
{

  const int gindex = iq2_data_index(GGML_TYPE_IQ1_M);

  const uint64_t *kgrid_q2xs = iq2_data[gindex].grid;
  const int *kmap_q2xs = iq2_data[gindex].map;
  const uint16_t *kneighbors_q2xs = iq2_data[gindex].neighbours;

  // assert(quant_weights   && "missing quantization weights");
  assert(kgrid_q2xs && "forgot to call ggml_quantize_init()?");
  assert(kmap_q2xs && "forgot to call ggml_quantize_init()?");
  assert(kneighbors_q2xs && "forgot to call ggml_quantize_init()?");
  assert(n % QK_K == 0);

  block_iq1_m *y = (block_iq1_m *)vy;

  const int64_t nbl = n / QK_K;

  const int block_size = IQ1M_BLOCK_SIZE;

  const float x_p[3] = {-1 + IQ1M_DELTA, IQ1M_DELTA, 1 + IQ1M_DELTA};
  const float x_m[3] = {-1 - IQ1M_DELTA, -IQ1M_DELTA, 1 - IQ1M_DELTA};
  const uint8_t masks[4] = {0x00, 0x80, 0x08, 0x88};

  int *idx = (int *)(pairs + 1);

  float sumqx[4], sumq2[4];

  iq1m_scale_t s;
  const float *xx;

  for (int ibl = 0; ibl < nbl; ++ibl)
  {
    memset(y[ibl].qs, 0, QK_K / 8);
    memset(y[ibl].qh, 0, QK_K / 16);
    memset(y[ibl].scales, 0, QK_K / 32);

    float max_scale = 0;

    const float *xbl = x + QK_K * ibl;
    float sumx2 = 0;
    for (int i = 0; i < QK_K; ++i)
      sumx2 += xbl[i] * xbl[i];
    float sigma2 = 2 * sumx2 / QK_K;

    for (int ib = 0; ib < QK_K / block_size; ++ib)
    {
      const float *xb = xbl + block_size * ib;
      if (quant_weights)
      {
        const float *qw = quant_weights + QK_K * ibl + block_size * ib;
        for (int i = 0; i < block_size; ++i)
          weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
      }
      else
      {
        for (int i = 0; i < block_size; ++i)
          weight[i] = xb[i] * xb[i];
      }
      float max = fabsf(xb[0]);
      for (int i = 1; i < block_size; ++i)
        max = MAX(max, fabsf(xb[i]));
      if (max < GROUP_MAX_EPS_IQ1_M)
      {
        scales[ib] = 0;
        shifts[ib] = 0;
        memset(L, 1, block_size);
        continue;
      }
      // Here we solve exactly the sum of squared difference (SSD) weighted minimization problem.
      // With just 3 allowed quant values (-1, 0, 1), we can search exhaustively for the two
      // boundaries that split the weights xb[i] into 3 groups. To do so, we sort the weights
      // in ascending order, compute Si = sum[weight[j] xb[j], j = 0...i] and
      // Wi = sum[weight[j], j = 0...i], and use these to quckly get get the optimum scale
      // for each possible and score for each split.
      for (int j = 0; j < block_size; ++j)
      {
        pairs[2 * j] = xb[j];
        idx[2 * j] = j;
      }
      qsort(pairs, block_size, 2 * sizeof(float), iq1_sort_helper);
      float best_score = -FLT_MAX, scale = max;
      int besti1 = -1, besti2 = -1, best_k = -1;
      // 0: +, +
      // 1: +, -
      // 2: -, +
      // 3: -, -
      for (int i1 = 0; i1 <= block_size; ++i1)
      {
        for (int i2 = i1; i2 <= block_size; ++i2)
        {
          memset(sumqx, 0, 4 * sizeof(float));
          memset(sumq2, 0, 4 * sizeof(float));
          for (int j = 0; j < i1; ++j)
          {
            int i = idx[2 * j];
            if (i < block_size / 2)
            {
              sumqx[0] += weight[i] * x_p[0] * xb[i];
              sumqx[1] += weight[i] * x_p[0] * xb[i];
              sumqx[2] += weight[i] * x_m[0] * xb[i];
              sumqx[3] += weight[i] * x_m[0] * xb[i];
              sumq2[0] += weight[i] * x_p[0] * x_p[0];
              sumq2[1] += weight[i] * x_p[0] * x_p[0];
              sumq2[2] += weight[i] * x_m[0] * x_m[0];
              sumq2[3] += weight[i] * x_m[0] * x_m[0];
            }
            else
            {
              sumqx[0] += weight[i] * x_p[0] * xb[i];
              sumqx[2] += weight[i] * x_p[0] * xb[i];
              sumqx[1] += weight[i] * x_m[0] * xb[i];
              sumqx[3] += weight[i] * x_m[0] * xb[i];
              sumq2[0] += weight[i] * x_p[0] * x_p[0];
              sumq2[2] += weight[i] * x_p[0] * x_p[0];
              sumq2[1] += weight[i] * x_m[0] * x_m[0];
              sumq2[3] += weight[i] * x_m[0] * x_m[0];
            }
          }
          for (int j = i1; j < i2; ++j)
          {
            int i = idx[2 * j];
            if (i < block_size / 2)
            {
              sumqx[0] += weight[i] * x_p[1] * xb[i];
              sumqx[1] += weight[i] * x_p[1] * xb[i];
              sumqx[2] += weight[i] * x_m[1] * xb[i];
              sumqx[3] += weight[i] * x_m[1] * xb[i];
              sumq2[0] += weight[i] * x_p[1] * x_p[1];
              sumq2[1] += weight[i] * x_p[1] * x_p[1];
              sumq2[2] += weight[i] * x_m[1] * x_m[1];
              sumq2[3] += weight[i] * x_m[1] * x_m[1];
            }
            else
            {
              sumqx[0] += weight[i] * x_p[1] * xb[i];
              sumqx[2] += weight[i] * x_p[1] * xb[i];
              sumqx[1] += weight[i] * x_m[1] * xb[i];
              sumqx[3] += weight[i] * x_m[1] * xb[i];
              sumq2[0] += weight[i] * x_p[1] * x_p[1];
              sumq2[2] += weight[i] * x_p[1] * x_p[1];
              sumq2[1] += weight[i] * x_m[1] * x_m[1];
              sumq2[3] += weight[i] * x_m[1] * x_m[1];
            }
          }
          for (int j = i2; j < block_size; ++j)
          {
            int i = idx[2 * j];
            if (i < block_size / 2)
            {
              sumqx[0] += weight[i] * x_p[2] * xb[i];
              sumqx[1] += weight[i] * x_p[2] * xb[i];
              sumqx[2] += weight[i] * x_m[2] * xb[i];
              sumqx[3] += weight[i] * x_m[2] * xb[i];
              sumq2[0] += weight[i] * x_p[2] * x_p[2];
              sumq2[1] += weight[i] * x_p[2] * x_p[2];
              sumq2[2] += weight[i] * x_m[2] * x_m[2];
              sumq2[3] += weight[i] * x_m[2] * x_m[2];
            }
            else
            {
              sumqx[0] += weight[i] * x_p[2] * xb[i];
              sumqx[2] += weight[i] * x_p[2] * xb[i];
              sumqx[1] += weight[i] * x_m[2] * xb[i];
              sumqx[3] += weight[i] * x_m[2] * xb[i];
              sumq2[0] += weight[i] * x_p[2] * x_p[2];
              sumq2[2] += weight[i] * x_p[2] * x_p[2];
              sumq2[1] += weight[i] * x_m[2] * x_m[2];
              sumq2[3] += weight[i] * x_m[2] * x_m[2];
            }
          }
          for (int k = 0; k < 4; ++k)
          {
            if (sumq2[k] > 0 && sumqx[k] * sumqx[k] > best_score * sumq2[k])
            {
              scale = sumqx[k] / sumq2[k];
              best_score = scale * sumqx[k];
              besti1 = i1;
              besti2 = i2;
              best_k = k;
            }
          }
        }
      }
      if (besti1 < 0 || besti2 < 0 || best_k < 0)
      {
        scales[ib] = 0;
        shifts[ib] = 0;
        memset(L, 1, block_size);
        continue;
      }
      for (int j = 0; j < besti1; ++j)
        L[idx[2 * j]] = 0;
      for (int j = besti1; j < besti2; ++j)
        L[idx[2 * j]] = 1;
      for (int j = besti2; j < block_size; ++j)
        L[idx[2 * j]] = 2;
      if (scale < 0)
      {
        for (int j = 0; j < block_size; ++j)
          L[j] = 2 - L[j];
        scale = -scale;
        best_k = best_k == 0 ? 3 : best_k == 1 ? 2
                               : best_k == 2   ? 1
                                               : 0;
      }
      bool all_on_grid = true;
      for (int k = 0; k < block_size / 8; ++k)
      {
        if (k == 0)
          xx = best_k < 2 ? x_p : x_m;
        else
          xx = best_k % 2 == 0 ? x_p : x_m;
        uint16_t u = 0;
        for (int j = 0; j < 8; ++j)
          u |= (L[8 * k + j] << 2 * j);
        int grid_index = kmap_q2xs[u];
        if (grid_index < 0)
        {
          all_on_grid = false;
          const uint16_t *neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
          grid_index = iq1_find_best_neighbour2(neighbours, kgrid_q2xs, xb + 8 * k, weight + 8 * k, scale, xx, L + 8 * k, NGRID_IQ1S);
          assert(grid_index >= 0);
        }
        index[k] = grid_index;
      }
      if (!all_on_grid)
      {
        float sumqx_f = 0, sumq2_f = 0;
        for (int k = 0; k < block_size / 8; ++k)
        {
          if (k == 0)
            xx = best_k < 2 ? x_p : x_m;
          else
            xx = best_k % 2 == 0 ? x_p : x_m;
          const int8_t *pg = (const int8_t *)(kgrid_q2xs + index[k]);
          for (int j = 0; j < 8; ++j)
          {
            float w = weight[8 * k + j];
            float q = xx[(pg[j] - 1) / 2];
            sumqx_f += w * q * xb[8 * k + j];
            sumq2_f += w * q * q;
          }
        }
        if (sumqx_f > 0 && sumq2_f > 0)
          scale = sumqx_f / sumq2_f;
      }
      y[ibl].qs[2 * ib + 0] = index[0] & 255;
      y[ibl].qs[2 * ib + 1] = index[1] & 255;
      y[ibl].qh[ib] = (index[0] >> 8) | ((index[1] >> 8) << 4);
      assert(scale >= 0);
      scales[ib] = scale;
      shifts[ib] = best_k;
      max_scale = MAX(max_scale, scale);
    }

    if (!max_scale)
    {
      continue;
    }

    uint16_t *sc = (uint16_t *)y[ibl].scales;
    float d = max_scale / 15;
    float id = 1 / d;
    float sumqx_f = 0, sumq2_f = 0;
    for (int ib = 0; ib < QK_K / block_size; ++ib)
    {
      int l = nearest_int(0.5f * (id * scales[ib + 0] - 1));
      l = MAX(0, MIN(7, l));
      sc[ib / 4] |= (l << 3 * (ib % 4));
      y[ibl].qh[ib] |= masks[shifts[ib]];
      const float *xb = xbl + block_size * ib;
      if (quant_weights)
      {
        const float *qw = quant_weights + QK_K * ibl + block_size * ib;
        for (int i = 0; i < block_size; ++i)
          weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
      }
      else
      {
        for (int i = 0; i < block_size; ++i)
          weight[i] = xb[i] * xb[i];
      }
      for (int k = 0; k < block_size / 8; ++k)
      {
        if (k == 0)
          xx = shifts[ib] < 2 ? x_p : x_m;
        else
          xx = shifts[ib] % 2 == 0 ? x_p : x_m;
        const int8_t *pg = (const int8_t *)(kgrid_q2xs + y[ibl].qs[2 * ib + k] + ((y[ibl].qh[ib] << (8 - 4 * k)) & 0x700));
        for (int j = 0; j < 8; ++j)
        {
          float w = weight[8 * k + j];
          float q = xx[(pg[j] - 1) / 2] * (2 * l + 1);
          sumqx_f += w * q * xb[8 * k + j];
          sumq2_f += w * q * q;
        }
      }
    }
    if (sumq2_f > 0)
      d = sumqx_f / sumq2_f;
    s.f16 = GGML_FP32_TO_FP16(d * 1.1125f); // 1.1125f is another fudge factor. Don't ask me why it is needed.
    sc[0] |= ((s.u16 & 0x000f) << 12);
    sc[1] |= ((s.u16 & 0x00f0) << 8);
    sc[2] |= ((s.u16 & 0x0f00) << 4);
    sc[3] |= ((s.u16 & 0xf000) << 0);
  }
}

size_t quantize_iq1_m(const float *RESTRICT src, void *RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  assert(n_per_row % QK_K == 0);
  float scales[QK_K / IQ1M_BLOCK_SIZE];
  float weight[IQ1M_BLOCK_SIZE];
  int8_t L[IQ1M_BLOCK_SIZE];
  float pairs[2 * IQ1M_BLOCK_SIZE];
  uint16_t index[IQ1M_BLOCK_SIZE / 8];
  int8_t shifts[QK_K / IQ1M_BLOCK_SIZE];
  int64_t nblock = n_per_row / QK_K;
  char *qrow = (char *)dst;
  for (int64_t row = 0; row < nrow; ++row)
  {
    quantize_row_iq1_m_impl(src, qrow, n_per_row, quant_weights, scales, weight, pairs, L, index, shifts);
    src += n_per_row;
    qrow += nblock * sizeof(block_iq1_m);
  }
  return nrow * nblock * sizeof(block_iq1_m);
}

// ============================ 4-bit non-linear quants

