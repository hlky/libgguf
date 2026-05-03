#include "libgguf_common.h"
#include "libgguf_iq_tables.h"

#define IQ3S_BLOCK_SIZE 32
static void quantize_row_iq3_s_impl(int block_size, const float *RESTRICT x, void *RESTRICT vy, int n,
                                    const float *RESTRICT quant_weights,
                                    float *scales,
                                    float *weight,
                                    float *xval,
                                    int8_t *L,
                                    int8_t *Laux,
                                    float *waux,
                                    bool *is_on_grid,
                                    bool *is_on_grid_aux,
                                    uint8_t *block_signs)
{

  const int gindex = iq3_data_index(512);

  const uint32_t *kgrid_q3xs = iq3_data[gindex].grid;
  const int *kmap_q3xs = iq3_data[gindex].map;
  const uint16_t *kneighbors_q3xs = iq3_data[gindex].neighbours;

  // assert(quant_weights   && "missing quantization weights");
  assert(kgrid_q3xs && "forgot to call ggml_quantize_init()?");
  assert(kmap_q3xs && "forgot to call ggml_quantize_init()?");
  assert(kneighbors_q3xs && "forgot to call ggml_quantize_init()?");
  assert(n % QK_K == 0);

  const int kMaxQ = 8;

  const int64_t nbl = n / QK_K;

  block_iq3_s *y = (block_iq3_s *)vy;

  const int bs4 = block_size / 4;
  const int bs8 = block_size / 8;

  for (int ibl = 0; ibl < nbl; ++ibl)
  {

    memset(&y[ibl], 0, sizeof(block_iq3_s));
    y[ibl].d = GGML_FP32_TO_FP16(0.f);

    uint8_t *qs = y[ibl].qs;
    uint8_t *qh = y[ibl].qh;
    uint8_t *signs = y[ibl].signs;

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
      for (int i = 0; i < block_size; ++i)
        waux[i] = sqrtf(weight[i]);
      for (int k = 0; k < bs8; ++k)
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
      for (int i = 1; i < block_size; ++i)
        max = MAX(max, xval[i]);
      memset(L, 0, block_size);
      if (!max)
      {
        scales[ib] = 0;
        continue;
      }
      float best = 0;
      float scale = max / (2 * kMaxQ - 1);
      for (int k = 0; k < bs4; ++k)
        is_on_grid[k] = false;
      for (int is = -9; is <= 9; ++is)
      {
        float id = (2 * kMaxQ - 1 + is * 0.2f) / max;
        float this_scale = 1 / id;
        for (int k = 0; k < bs4; ++k)
        {
          for (int i = 0; i < 4; ++i)
          {
            int l = nearest_int(0.5f * (id * xval[4 * k + i] - 1));
            Laux[4 * k + i] = MAX(0, MIN(kMaxQ - 1, l));
          }
          uint16_t u = 0;
          for (int i = 0; i < 4; ++i)
            u |= (Laux[4 * k + i] << 3 * i);
          int grid_index = kmap_q3xs[u];
          is_on_grid_aux[k] = true;
          if (grid_index < 0)
          {
            is_on_grid_aux[k] = false;
            const uint16_t *neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
            grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4 * k, waux + 4 * k, this_scale, Laux + 4 * k);
          }
        }
        float sumqx = 0, sumq2 = 0;
        for (int i = 0; i < block_size; ++i)
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
          for (int i = 0; i < block_size; ++i)
            L[i] = Laux[i];
          for (int k = 0; k < bs4; ++k)
            is_on_grid[k] = is_on_grid_aux[k];
        }
      }
      int n_not_ongrid = 0;
      for (int k = 0; k < bs4; ++k)
        if (!is_on_grid[k])
          ++n_not_ongrid;
      if (n_not_ongrid > 0 && scale > 0)
      {
        float id = 1 / scale;
        for (int k = 0; k < bs4; ++k)
        {
          // if (is_on_grid[k]) continue;
          uint16_t u = 0;
          for (int i = 0; i < 4; ++i)
          {
            int l = nearest_int(0.5f * (id * xval[4 * k + i] - 1));
            l = MAX(0, MIN(kMaxQ - 1, l));
            u |= (l << 3 * i);
          }
          int grid_index = kmap_q3xs[u];
          if (grid_index < 0)
          {
            const uint16_t *neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
            grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4 * k, waux + 4 * k, scale, L + 4 * k);
          }
          const int8_t *pg = (const int8_t *)(kgrid_q3xs + grid_index);
          for (int i = 0; i < 4; ++i)
            L[4 * k + i] = (pg[i] - 1) / 2;
        }
        float sumqx = 0, sumq2 = 0;
        for (int i = 0; i < block_size; ++i)
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
        for (int k = 0; k < bs8; ++k)
          block_signs[k] = ~block_signs[k];
      }
      for (int k = 0; k < bs4; ++k)
      {
        uint16_t u = 0;
        for (int i = 0; i < 4; ++i)
          u |= (L[4 * k + i] << 3 * i);
        int grid_index = kmap_q3xs[u];
        if (grid_index < 0)
        {
          printf("Oops: found point %u not on grid:", u);
          for (int i = 0; i < 4; ++i)
            printf(" %d", L[4 * k + i]);
          printf("\n");
          printf("fatal error");
          abort();
        }
        qs[k] = grid_index & 255;
        qh[(ib * bs4 + k) / 8] |= ((grid_index >> 8) << ((ib * bs4 + k) % 8));
      }
      qs += bs4;
      for (int k = 0; k < bs8; ++k)
        signs[k] = block_signs[k];
      signs += bs8;
      assert(scale >= 0);
      scales[ib] = scale;
      max_scale = MAX(max_scale, scale);
    }

    if (!max_scale)
    {
      continue;
    }

    float d = max_scale / 31;
    y[ibl].d = GGML_FP32_TO_FP16(d * 1.033f);
    float id = 1 / d;
    for (int ib = 0; ib < QK_K / block_size; ib += 2)
    {
      int l1 = nearest_int(0.5f * (id * scales[ib + 0] - 1));
      l1 = MAX(0, MIN(15, l1));
      int l2 = nearest_int(0.5f * (id * scales[ib + 1] - 1));
      l2 = MAX(0, MIN(15, l2));
      y[ibl].scales[ib / 2] = l1 | (l2 << 4);
    }
  }
}

#define IQ3S_BLOCK_SIZE 32
size_t quantize_iq3_s(const float *RESTRICT src, void *RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  assert(n_per_row % QK_K == 0);
  int64_t nblock = n_per_row / QK_K;
  float scales[QK_K / IQ3S_BLOCK_SIZE];
  float weight[IQ3S_BLOCK_SIZE];
  float xval[IQ3S_BLOCK_SIZE];
  int8_t L[IQ3S_BLOCK_SIZE];
  int8_t Laux[IQ3S_BLOCK_SIZE];
  float waux[IQ3S_BLOCK_SIZE];
  bool is_on_grid[IQ3S_BLOCK_SIZE / 4];
  bool is_on_grid_aux[IQ3S_BLOCK_SIZE / 4];
  uint8_t block_signs[IQ3S_BLOCK_SIZE / 8];
  char *qrow = (char *)dst;
  for (int64_t row = 0; row < nrow; ++row)
  {
    quantize_row_iq3_s_impl(IQ3S_BLOCK_SIZE, src, qrow, n_per_row, quant_weights,
                            scales, weight, xval, L, Laux, waux, is_on_grid, is_on_grid_aux, block_signs);
    src += n_per_row;
    qrow += nblock * sizeof(block_iq3_s);
  }
  return nrow * nblock * sizeof(block_iq3_s);
}

void quantize_row_iq3_s(const float *RESTRICT x, block_iq3_s *RESTRICT y, int64_t k)
{
  assert(k % QK_K == 0);
  quantize_iq3_s(x, y, 1, k, nullptr);
}

// =================================== 1.5 bpw ===================================================

