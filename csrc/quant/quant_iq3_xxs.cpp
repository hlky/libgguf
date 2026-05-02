#include "libgguf_common.h"
#include "libgguf_iq_tables.h"

static void quantize_row_iq3_xxs_impl(int grid_size, const float *RESTRICT x, void *RESTRICT vy, int64_t n,
                                      const float *RESTRICT quant_weights)
{

  const int gindex = iq3_data_index(grid_size);

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

  ggml_fp16_t *dh;
  uint8_t *qs;
  int block_size;
  if (grid_size == 256)
  {
    block_iq3_xxs *y = (block_iq3_xxs *)vy;
    dh = &y->d;
    qs = y->qs;
    block_size = sizeof(block_iq3_xxs);
  }
  else
  {
    block_iq3_s *y = (block_iq3_s *)vy;
    dh = &y->d;
    qs = y->qs;
    block_size = sizeof(block_iq3_s);
  }
  int quant_size = block_size - sizeof(ggml_fp16_t);

  float scales[QK_K / 32];
  float weight[32];
  float xval[32];
  int8_t L[32];
  int8_t Laux[32];
  float waux[32];
  bool is_on_grid[8];
  bool is_on_grid_aux[8];
  uint8_t block_signs[8];
  uint8_t q3[3 * (QK_K / 8) + QK_K / 32];
  uint32_t *scales_and_signs = (uint32_t *)(q3 + QK_K / 4);
  uint8_t *qh = q3 + 3 * (QK_K / 8);

  for (int ibl = 0; ibl < nbl; ++ibl)
  {

    dh[0] = GGML_FP32_TO_FP16(0.f);
    memset(q3, 0, 3 * QK_K / 8 + QK_K / 32);

    float max_scale = 0;

    const float *xbl = x + QK_K * ibl;
    float sumx2 = 0;
    for (int i = 0; i < QK_K; ++i)
      sumx2 += xbl[i] * xbl[i];
    float sigma2 = 2 * sumx2 / QK_K;

    for (int ib = 0; ib < QK_K / 32; ++ib)
    {
      const float *xb = xbl + 32 * ib;
      if (quant_weights)
      {
        const float *qw = quant_weights + QK_K * ibl + 32 * ib;
        for (int i = 0; i < 32; ++i)
          weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
      }
      else
      {
        for (int i = 0; i < 32; ++i)
          weight[i] = xb[i] * xb[i];
      }
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
      memset(L, 0, 32);
      if (max < GROUP_MAX_EPS_IQ3_XXS)
      {
        scales[ib] = 0;
        continue;
      }
      float best = 0;
      float scale = max / (2 * kMaxQ - 1);
      for (int k = 0; k < 8; ++k)
        is_on_grid[k] = true;
      for (int is = -15; is <= 15; ++is)
      {
        float id = (2 * kMaxQ - 1 + is * 0.2f) / max;
        float this_scale = 1 / id;
        for (int k = 0; k < 8; ++k)
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
          for (int i = 0; i < 32; ++i)
            L[i] = Laux[i];
          for (int k = 0; k < 8; ++k)
            is_on_grid[k] = is_on_grid_aux[k];
        }
      }
      int n_not_ongrid = 0;
      for (int k = 0; k < 8; ++k)
        if (!is_on_grid[k])
          ++n_not_ongrid;
      if (n_not_ongrid > 0 && scale > 0)
      {
        float id = 1 / scale;
        for (int k = 0; k < 8; ++k)
        {
          if (is_on_grid[k])
            continue;
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
      for (int k = 0; k < 8; ++k)
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
        if (grid_size == 256)
        {
          q3[8 * ib + k] = grid_index;
        }
        else
        {
          q3[8 * ib + k] = grid_index & 255;
          qh[ib] |= ((grid_index >> 8) << k);
        }
      }
      scales_and_signs[ib] = block_signs[0] | (block_signs[1] << 7) | (block_signs[2] << 14) | (block_signs[3] << 21);
      assert(scale >= 0);
      scales[ib] = scale;
      max_scale = MAX(max_scale, scale);
    }

    if (!max_scale)
    {
      memset(qs, 0, quant_size);
      dh += block_size / sizeof(ggml_fp16_t);
      qs += block_size;
      continue;
    }

    float d = max_scale / 31;
    dh[0] = GGML_FP32_TO_FP16(d * 1.0125f); // small improvement via this fudge factor
    float id = 1 / d;
    for (int ib = 0; ib < QK_K / 32; ++ib)
    {
      int l = nearest_int(0.5f * (id * scales[ib] - 1));
      l = MAX(0, MIN(15, l));
      scales_and_signs[ib] |= ((uint32_t)l << 28);
    }
    memcpy(qs, q3, quant_size);

    dh += block_size / sizeof(ggml_fp16_t);
    qs += block_size;
  }
}

size_t quantize_iq3_xxs(const float *RESTRICT src, void *RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  assert(n_per_row % QK_K == 0);
  int64_t nblock = n_per_row / QK_K;
  char *qrow = (char *)dst;
  for (int64_t row = 0; row < nrow; ++row)
  {
    quantize_row_iq3_xxs_impl(256, src, qrow, n_per_row, quant_weights);
    src += n_per_row;
    qrow += nblock * sizeof(block_iq3_xxs);
  }
  return nrow * nblock * sizeof(block_iq3_xxs);
}

void quantize_row_iq3_xxs_ref(const float *RESTRICT x, block_iq3_xxs *RESTRICT y, int64_t k)
{
  assert(k % QK_K == 0);
  quantize_row_iq3_xxs_impl(256, x, y, k, nullptr);
}

