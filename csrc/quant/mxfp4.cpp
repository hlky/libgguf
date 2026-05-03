#include "libgguf_common.h"
#include "common/libgguf_cpu.h"
#include "libgguf_tables.h"

#include <cstring>

typedef void (*libgguf_mxfp4_kernel_fn)(const float *RESTRICT, block_mxfp4 *RESTRICT, int64_t);

struct libgguf_mxfp4_selection
{
  const char *backend;
  libgguf_mxfp4_kernel_fn kernel;
};

extern "C" void quantize_row_mxfp4_sse2(const float *RESTRICT x, block_mxfp4 *RESTRICT y, int64_t k);
extern "C" void quantize_row_mxfp4_sse4_1(const float *RESTRICT x, block_mxfp4 *RESTRICT y, int64_t k);
extern "C" void quantize_row_mxfp4_avx2(const float *RESTRICT x, block_mxfp4 *RESTRICT y, int64_t k);

void quantize_row_mxfp4(const float *RESTRICT x, block_mxfp4 *RESTRICT y, int64_t k)
{
  static const int qk = QK_MXFP4;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++)
  {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < qk; j++)
    {
      const float v = x[i * qk + j];

      if (amax < fabsf(v))
      {
        amax = fabsf(v);
      }
    }

    const uint8_t e = amax > 0.0f ? (uint8_t)(floorf(log2f(amax)) - 2 + 127) : 0;

    const float d = GGML_E8M0_TO_FP32_HALF(e);

    y[i].e = e;

    for (int j = 0; j < qk / 2; ++j)
    {
      const uint8_t x0 = best_index_mxfp4(x[i * qk + 0 + j], d);
      const uint8_t x1 = best_index_mxfp4(x[i * qk + qk / 2 + j], d);

      y[i].qs[j] = x0;
      y[i].qs[j] |= x1 << 4;
    }
  }
}

static libgguf_mxfp4_selection libgguf_mxfp4_select_kernel()
{
  const libgguf_cpu_features &features = libgguf_get_cpu_features();

  if (features.avx2)
    return {"avx2", quantize_row_mxfp4_avx2};
  if (features.sse4_1)
    return {"sse4_1", quantize_row_mxfp4_sse4_1};
  if (features.sse2)
    return {"sse2", quantize_row_mxfp4_sse2};
  return {"ref", quantize_row_mxfp4};
}

static const libgguf_mxfp4_selection &libgguf_mxfp4_selected()
{
  static const libgguf_mxfp4_selection selected = libgguf_mxfp4_select_kernel();
  return selected;
}

extern "C" const char *libgguf_mxfp4_backend(void)
{
  return libgguf_mxfp4_selected().backend;
}

extern "C" int libgguf_mxfp4_cpu_supports_backend(const char *backend)
{
  const libgguf_cpu_features &features = libgguf_get_cpu_features();
  if (backend == nullptr || std::strcmp(backend, "ref") == 0)
    return 1;
  if (std::strcmp(backend, "sse2") == 0)
    return features.sse2 ? 1 : 0;
  if (std::strcmp(backend, "sse4_1") == 0)
    return features.sse4_1 ? 1 : 0;
  if (std::strcmp(backend, "avx2") == 0)
    return features.avx2 ? 1 : 0;
  return 0;
}

static libgguf_mxfp4_kernel_fn libgguf_mxfp4_kernel_for_backend(const char *backend)
{
  if (!libgguf_mxfp4_cpu_supports_backend(backend))
    return nullptr;
  if (backend == nullptr || std::strcmp(backend, "ref") == 0)
    return quantize_row_mxfp4;
  if (std::strcmp(backend, "sse2") == 0)
    return quantize_row_mxfp4_sse2;
  if (std::strcmp(backend, "sse4_1") == 0)
    return quantize_row_mxfp4_sse4_1;
  if (std::strcmp(backend, "avx2") == 0)
    return quantize_row_mxfp4_avx2;
  return nullptr;
}

extern "C" size_t libgguf_quantize_mxfp4_for_backend(
    const char *backend,
    const float *RESTRICT src,
    void *RESTRICT dst,
    int64_t nrows,
    int64_t n_per_row)
{
  libgguf_mxfp4_kernel_fn kernel = libgguf_mxfp4_kernel_for_backend(backend);
  if (!kernel)
    return 0;
  kernel(src, (block_mxfp4 *)dst, nrows * n_per_row);
  return nrows * libgguf_row_size(GGML_TYPE_MXFP4, n_per_row);
}

size_t quantize_mxfp4(const float *RESTRICT src, void *RESTRICT dst, int64_t nrow, int64_t n_per_row, const float *quant_weights)
{
  GGML_UNUSED(quant_weights);
  libgguf_mxfp4_selected().kernel(src, (block_mxfp4 *)dst, (int64_t)nrow * n_per_row);
  return nrow * libgguf_row_size(GGML_TYPE_MXFP4, n_per_row);
}
