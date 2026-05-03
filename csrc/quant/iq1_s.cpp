#include "libgguf_common.h"
#include "libgguf_iq_tables.h"

#define IQ1S_BLOCK_SIZE 32
#define IQ1M_BLOCK_SIZE 16
static void quantize_row_iq1_s_impl(const float *RESTRICT x, void *RESTRICT vy, int64_t n, const float *RESTRICT quant_weights,
                                    float *scales,
                                    float *weight,
                                    float *sumx,
                                    float *sumw,
                                    float *pairs,
                                    int8_t *L,
                                    uint16_t *index,
                                    int8_t *shifts)
{

  const int gindex = iq2_data_index(GGML_TYPE_IQ1_S);

  const uint64_t *kgrid_q2xs = iq2_data[gindex].grid;
  const int *kmap_q2xs = iq2_data[gindex].map;
  const uint16_t *kneighbors_q2xs = iq2_data[gindex].neighbours;

  assert(quant_weights && "missing quantization weights");
  assert(kgrid_q2xs && "forgot to call ggml_quantize_init()?");
  assert(kmap_q2xs && "forgot to call ggml_quantize_init()?");
  assert(kneighbors_q2xs && "forgot to call ggml_quantize_init()?");
  assert(n % QK_K == 0);

  block_iq1_s *y = (block_iq1_s *)vy;

  const int64_t nbl = n / QK_K;

  const int block_size = IQ1S_BLOCK_SIZE;

  const float x_p[3] = {-1 + IQ1S_DELTA, IQ1S_DELTA, 1 + IQ1S_DELTA};
  const float x_m[3] = {-1 - IQ1S_DELTA, -IQ1S_DELTA, 1 - IQ1S_DELTA};

  int *idx = (int *)(pairs + 1);

  for (int ibl = 0; ibl < nbl; ++ibl)
  {

    y[ibl].d = GGML_FP32_TO_FP16(0.f);
    memset(y[ibl].qs, 0, QK_K / 8);
    memset(y[ibl].qh, 0, QK_K / 16);

    float max_scale = 0;

    const float *xbl = x + QK_K * ibl;
    float sumx2 = 0;
    for (int i = 0; i < QK_K; ++i)
      sumx2 += xbl[i] * xbl[i];
    float sigma2 = 2 * sumx2 / QK_K;

    for (int ib = 0; ib < QK_K / block_size; ++ib)
    {
      const float *xb = xbl + block_size * ib;
      const float *qw = quant_weights + QK_K * ibl + block_size * ib;
      for (int i = 0; i < block_size; ++i)
        weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
      float max = fabsf(xb[0]);
      for (int i = 1; i < block_size; ++i)
        max = MAX(max, fabsf(xb[i]));
      if (max < GROUP_MAX_EPS_IQ1_S)
      {
        scales[ib] = 0;
        shifts[ib] = 1;
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
      {
        sumx[0] = sumw[0] = 0;
        for (int j = 0; j < block_size; ++j)
        {
          int i = idx[2 * j];
          sumx[j + 1] = sumx[j] + weight[i] * xb[i];
          sumw[j + 1] = sumw[j] + weight[i];
        }
      }
      float best_score = -FLT_MAX, scale = max;
      int besti1 = -1, besti2 = -1, best_shift = 0;
      for (int i1 = 0; i1 <= block_size; ++i1)
      {
        for (int i2 = i1; i2 <= block_size; ++i2)
        {
          float sumqx = (sumx[i1] - sumx[0]) * x_p[0] + (sumx[i2] - sumx[i1]) * x_p[1] + (sumx[block_size] - sumx[i2]) * x_p[2];
          float sumq2 = (sumw[i1] - sumw[0]) * x_p[0] * x_p[0] + (sumw[i2] - sumw[i1]) * x_p[1] * x_p[1] + (sumw[block_size] - sumw[i2]) * x_p[2] * x_p[2];
          if (sumq2 > 0 && sumqx * sumqx > best_score * sumq2)
          {
            scale = sumqx / sumq2;
            best_score = scale * sumqx;
            besti1 = i1;
            besti2 = i2;
            best_shift = 1;
          }
          sumqx = (sumx[i1] - sumx[0]) * x_m[0] + (sumx[i2] - sumx[i1]) * x_m[1] + (sumx[block_size] - sumx[i2]) * x_m[2];
          sumq2 = (sumw[i1] - sumw[0]) * x_m[0] * x_m[0] + (sumw[i2] - sumw[i1]) * x_m[1] * x_m[1] + (sumw[block_size] - sumw[i2]) * x_m[2] * x_m[2];
          if (sumq2 > 0 && sumqx * sumqx > best_score * sumq2)
          {
            scale = sumqx / sumq2;
            best_score = scale * sumqx;
            besti1 = i1;
            besti2 = i2;
            best_shift = -1;
          }
        }
      }
      if (besti1 < 0 || besti2 < 0 || best_shift == 0)
      {
        scales[ib] = 0;
        shifts[ib] = 1;
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
        best_shift = -best_shift;
      }
      bool all_on_grid = true;
      const float *xx = best_shift == 1 ? x_p : x_m;
      for (int k = 0; k < block_size / 8; ++k)
      {
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
        float sumqx = 0, sumq2 = 0;
        for (int k = 0; k < block_size / 8; ++k)
        {
          const int8_t *pg = (const int8_t *)(kgrid_q2xs + index[k]);
          for (int j = 0; j < 8; ++j)
          {
            float w = weight[8 * k + j];
            float q = xx[(pg[j] - 1) / 2];
            sumqx += w * q * xb[8 * k + j];
            sumq2 += w * q * q;
          }
        }
        if (sumqx > 0 && sumq2 > 0)
          scale = sumqx / sumq2;
      }
      uint16_t h = 0;
      for (int k = 0; k < block_size / 8; ++k)
      {
        y[ibl].qs[(block_size / 8) * ib + k] = index[k] & 255;
        h |= (index[k] >> 8) << 3 * k;
      }
      y[ibl].qh[ib] = h;
      assert(scale >= 0);
      scales[ib] = scale;
      shifts[ib] = best_shift;
      max_scale = MAX(max_scale, scale);
    }

    if (!max_scale)
    {
      continue;
    }

    float d = max_scale / 15;
    y[ibl].d = GGML_FP32_TO_FP16(d * 1.125f); // 1.125f is another fudge factor. Don't ask me why it is needed.
    float id = 1 / d;
    for (int ib = 0; ib < QK_K / block_size; ++ib)
    {
      int l = nearest_int(0.5f * (id * scales[ib] - 1));
      l = MAX(0, MIN(7, l));
      if (shifts[ib] == -1)
        l |= 8;
      y[ibl].qh[ib] |= (l << 12);
    }
  }
}

size_t quantize_iq1_s(const float *RESTRICT src, void *RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  assert(n_per_row % QK_K == 0);
  float scales[QK_K / IQ1S_BLOCK_SIZE];
  float weight[IQ1S_BLOCK_SIZE];
  int8_t L[IQ1S_BLOCK_SIZE];
  float sumx[IQ1S_BLOCK_SIZE + 1];
  float sumw[IQ1S_BLOCK_SIZE + 1];
  float pairs[2 * IQ1S_BLOCK_SIZE];
  uint16_t index[IQ1S_BLOCK_SIZE / 8];
  int8_t shifts[QK_K / IQ1S_BLOCK_SIZE];
  int64_t nblock = n_per_row / QK_K;
  char *qrow = (char *)dst;
  for (int64_t row = 0; row < nrow; ++row)
  {
    quantize_row_iq1_s_impl(src, qrow, n_per_row, quant_weights, scales, weight, sumx, sumw, pairs, L, index, shifts);
    src += n_per_row;
    qrow += nblock * sizeof(block_iq1_s);
  }
  return nrow * nblock * sizeof(block_iq1_s);
}
